"""Training loops for the SST-2 LoRA experiments using ``transformers.Trainer``.

The structure mirrors ``messy_random_code/Experiments_seq_rank1.ipynb``
(and ``NOplots.ipynb``): for each strategy we train one epoch at a time so
we can record the per-epoch validation accuracy without restarting training.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .nlp_model import (
    LoRALayer,
    add_next_sequential_component,
    count_parameters,
    enable_standard_lora_training,
    prepare_lora_model,
)


def _compute_metrics_factory():
    """Build the ``compute_metrics`` callback used by HuggingFace ``Trainer``."""
    import evaluate as hf_evaluate

    accuracy_metric = hf_evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    return compute_metrics


def _per_epoch_loop(
    model,
    train_dataset,
    eval_dataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    output_root: str,
    label: str,
    precision: str = "fp32",
) -> tuple[list[dict], float]:
    """Train ``model`` one epoch at a time, evaluating after each epoch.

    ``precision`` controls mixed-precision training (``"fp32"``, ``"fp16"`` or
    ``"bf16"``).  When CUDA is unavailable we transparently fall back to fp32
    so the same code path works on CPU.
    """
    from transformers import Trainer, TrainingArguments
    import torch

    compute_metrics = _compute_metrics_factory()
    history: list[dict] = []
    cumulative_seconds = 0.0

    use_fp16 = precision == "fp16" and torch.cuda.is_available()
    use_bf16 = precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    for epoch in range(1, epochs + 1):
        out_dir = os.path.join(output_root, f"{label}_epoch_{epoch}")
        os.makedirs(out_dir, exist_ok=True)

        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            logging_steps=200,
            report_to="none",
            save_strategy="no",
            disable_tqdm=True,
            fp16=use_fp16,
            bf16=use_bf16,
            dataloader_num_workers=2,
            dataloader_pin_memory=torch.cuda.is_available(),
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        t0 = time.time()
        trainer.train()
        cumulative_seconds += time.time() - t0

        metrics = trainer.evaluate()
        history.append(
            {
                "epoch": epoch,
                "validation_loss": float(metrics["eval_loss"]),
                "accuracy": float(metrics["eval_accuracy"]),
            }
        )

    return history, cumulative_seconds


# --------------------------------------------------------------------------- #
# Public training entry points
# --------------------------------------------------------------------------- #


@dataclass
class StandardLoRARun:
    rank: int
    epochs: int
    history: list[dict] = field(default_factory=list)
    final_accuracy: float = float("nan")
    trainable_params: int = 0
    total_params: int = 0
    training_time_seconds: float = 0.0


def train_standard_lora_sst2(
    train_dataset,
    eval_dataset,
    *,
    rank: int = 3,
    epochs: int = 6,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    model_name: str = "distilbert-base-uncased",
    output_root: Optional[str] = None,
    precision: str = "fp32",
) -> StandardLoRARun:
    """Standard (jointly-optimised) LoRA on SST-2."""
    output_root = output_root or tempfile.mkdtemp(prefix="standard_lora_")
    model = prepare_lora_model(model_name=model_name, rank=rank, sequential=False)
    enable_standard_lora_training(model)
    trainable, total = count_parameters(model)

    history, seconds = _per_epoch_loop(
        model,
        train_dataset,
        eval_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        output_root=output_root,
        label="standard_lora",
        precision=precision,
    )

    return StandardLoRARun(
        rank=rank,
        epochs=epochs,
        history=history,
        final_accuracy=history[-1]["accuracy"] if history else float("nan"),
        trainable_params=trainable,
        total_params=total,
        training_time_seconds=seconds,
    )


@dataclass
class SequentialPathRun:
    name: str
    epoch_allocation: List[int]
    component_history: list[dict] = field(default_factory=list)
    final_accuracy: float = float("nan")
    trainable_params: int = 0
    total_params: int = 0
    training_time_seconds: float = 0.0

    @property
    def total_epochs(self) -> int:
        return sum(self.epoch_allocation)


def train_sequential_lora_sst2(
    train_dataset,
    eval_dataset,
    *,
    epoch_allocation: Sequence[int],
    name: Optional[str] = None,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    model_name: str = "distilbert-base-uncased",
    output_root: Optional[str] = None,
    precision: str = "fp32",
) -> SequentialPathRun:
    """Train a single sequential rank-1 LoRA path on SST-2.

    The same model object is reused across components — only the freshly
    added rank-1 component is trainable at any one time.
    """
    name = name or "-".join(str(x) for x in epoch_allocation)
    output_root = output_root or tempfile.mkdtemp(prefix=f"seq_{name}_")

    model = prepare_lora_model(model_name=model_name, rank=len(epoch_allocation), sequential=True)
    component_history: list[dict] = []
    cumulative_seconds = 0.0

    for k, n_epochs in enumerate(epoch_allocation):
        add_next_sequential_component(model, k)
        history, seconds = _per_epoch_loop(
            model,
            train_dataset,
            eval_dataset,
            epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            output_root=output_root,
            label=f"seq_{name}_comp_{k}",
            precision=precision,
        )
        cumulative_seconds += seconds
        last = history[-1]
        component_history.append(
            {
                "component_index": k + 1,
                "epochs": n_epochs,
                "accuracy_after": last["accuracy"],
                "validation_loss_after": last["validation_loss"],
            }
        )

    trainable, total = count_parameters(model)
    return SequentialPathRun(
        name=name,
        epoch_allocation=list(epoch_allocation),
        component_history=component_history,
        final_accuracy=component_history[-1]["accuracy_after"] if component_history else float("nan"),
        trainable_params=trainable,
        total_params=total,
        training_time_seconds=cumulative_seconds,
    )


def run_sst2_sweep(
    sequential_paths: Dict[str, Sequence[int]],
    *,
    standard_lora_rank: int = 3,
    standard_lora_epochs: int = 6,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    model_name: str = "distilbert-base-uncased",
    output_root: Optional[str] = None,
    max_length: int = 128,
    precision: str = "fp32",
) -> dict:
    """End-to-end SST-2 sweep used by ``run_nlp_experiment.py``.

    Returns a JSON-serialisable dict describing the standard LoRA baseline
    plus every sequential path.
    """
    from .nlp_data import load_sst2

    print("Loading SST-2...")
    train_ds, eval_ds, _ = load_sst2(tokenizer_name=model_name, max_length=max_length)

    print(f"Standard LoRA r={standard_lora_rank} ({standard_lora_epochs} epochs, precision={precision})...")
    standard = train_standard_lora_sst2(
        train_ds,
        eval_ds,
        rank=standard_lora_rank,
        epochs=standard_lora_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        model_name=model_name,
        output_root=output_root,
        precision=precision,
    )
    print(f"  -> final acc: {standard.final_accuracy:.4f}")

    sequential_results: Dict[str, dict] = {}
    for name, alloc in sequential_paths.items():
        print(f"Sequential path {name} = {list(alloc)}...")
        run = train_sequential_lora_sst2(
            train_ds,
            eval_ds,
            epoch_allocation=alloc,
            name=name,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            model_name=model_name,
            output_root=output_root,
            precision=precision,
        )
        sequential_results[name] = {
            "epoch_allocation": run.epoch_allocation,
            "component_history": run.component_history,
            "final_accuracy": run.final_accuracy,
            "total_epochs": run.total_epochs,
            "trainable_params": run.trainable_params,
            "training_time_seconds": run.training_time_seconds,
        }
        print(f"  -> final acc: {run.final_accuracy:.4f}")

    return {
        "dataset": "sst2",
        "standard_lora": {
            "rank": standard.rank,
            "epochs": standard.epochs,
            "final_accuracy": standard.final_accuracy,
            "history": standard.history,
            "trainable_params": standard.trainable_params,
            "total_params": standard.total_params,
            "training_time_seconds": standard.training_time_seconds,
        },
        "sequential_paths": sequential_results,
        "config": {
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_length": max_length,
            "precision": precision,
        },
        "flops": {
            "architecture": "distilbert_qv",
            # 6 transformer blocks × {q_lin, v_lin}, all 768×768.
            "layers": [[768, 768]] * 12,
            # Per-token cost for transformer attention; one "sample" in the
            # FLOPs sense is one token, so use train_examples × max_length.
            "samples_per_epoch": len(train_ds) * max_length,
            "num_train_examples": len(train_ds),
            "max_length": max_length,
        },
    }
