"""Training/evaluation utilities for the vision LoRA experiments.

These helpers wrap a small, deterministic-ish PyTorch training loop around
:class:`lora_lib.vision_model.SimpleLoRAModel`.  They intentionally avoid
PyTorch Lightning so the experiment runner stays readable and easy to debug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .flops import vision_mlp_layers
from .vision_model import SimpleLoRAModel


def _device() -> torch.device:
    if torch.cuda.is_available():
        # Free speedup on Ampere/Hopper, no measurable accuracy impact for
        # the small MLPs we train here.
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


def _train_one_epoch(
    model: SimpleLoRAModel,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_accuracy(model: SimpleLoRAModel, loader: DataLoader, device: Optional[torch.device] = None) -> float:
    """Return classification accuracy of ``model`` on ``loader``."""
    device = device or _device()
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_model(
    model: SimpleLoRAModel,
    train_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> SimpleLoRAModel:
    """Train ``model`` for ``epochs`` epochs with Adam and the given lr."""
    device = device or _device()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        # Nothing to train; the model is fully frozen.
        return model
    optim = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        _train_one_epoch(model, train_loader, optim, device)
    return model


# --------------------------------------------------------------------------- #
# Higher-level helpers used by run_vision_experiment.py
# --------------------------------------------------------------------------- #


def train_base_model(
    input_size: int,
    num_classes: int,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    seed: int = 42,
) -> SimpleLoRAModel:
    """Train the unadapted base classifier on the old-class subset."""
    torch.manual_seed(seed)
    model = SimpleLoRAModel(input_size=input_size, num_classes=num_classes, rank=0)
    return train_model(model, train_loader, epochs=epochs, lr=lr)


def train_standard_lora(
    base_model: SimpleLoRAModel,
    train_loader: DataLoader,
    rank: int,
    epochs: int,
    lr: float = 1e-3,
    seed: int = 42,
) -> SimpleLoRAModel:
    """Train a standard (jointly-optimised) LoRA adapter of the given rank."""
    torch.manual_seed(seed)
    model = SimpleLoRAModel(
        input_size=base_model.input_size,
        num_classes=base_model.num_classes,
        rank=rank,
        base_network=base_model,
    )
    return train_model(model, train_loader, epochs=epochs, lr=lr)


@dataclass
class SequentialPathResult:
    """Outcome of training one α-β-γ sequential rank-1 path."""

    epoch_allocation: List[int]
    component_accuracies: List[float] = field(default_factory=list)
    cumulative_epochs: List[int] = field(default_factory=list)
    trainable_params_per_component: List[int] = field(default_factory=list)

    @property
    def total_epochs(self) -> int:
        return sum(self.epoch_allocation)

    @property
    def final_accuracy(self) -> float:
        return self.component_accuracies[-1] if self.component_accuracies else float("nan")


def train_sequential_lora_path(
    base_model: SimpleLoRAModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epoch_allocation: Sequence[int],
    lr: float = 1e-3,
    seed: int = 42,
) -> SequentialPathResult:
    """Train one sequential rank-1 LoRA path with the given per-component epochs.

    For an allocation ``(α, β, γ)`` we train:

        rank-1 model → α epochs → freeze, snapshot components
            ↓
        rank-2 model (k=0 frozen) → β epochs → freeze, snapshot
            ↓
        rank-3 model (k=0,1 frozen) → γ epochs

    The accuracy on the *new* class test set is recorded after each
    component finishes training.
    """
    torch.manual_seed(seed)
    result = SequentialPathResult(epoch_allocation=list(epoch_allocation))

    fixed_components: List = []
    cumulative = 0
    current_model: Optional[SimpleLoRAModel] = None
    for k, n_epochs in enumerate(epoch_allocation, start=1):
        current_model = SimpleLoRAModel(
            input_size=base_model.input_size,
            num_classes=base_model.num_classes,
            rank=k,
            base_network=base_model,
            fixed_components=fixed_components,
        )
        trainable_count = current_model.num_trainable_params()
        train_model(current_model, train_loader, epochs=n_epochs, lr=lr)

        acc = evaluate_accuracy(current_model, test_loader)
        cumulative += n_epochs

        result.component_accuracies.append(acc)
        result.cumulative_epochs.append(cumulative)
        result.trainable_params_per_component.append(trainable_count)

        # Snapshot the components from the just-trained model so the next
        # iteration can build on top with the previous components frozen.
        fixed_components = current_model.get_components()

    return result


def evaluate_sweep(
    *,
    dataset_name: str,
    input_size: int,
    num_classes: int,
    train_old_loader: DataLoader,
    train_new_loader: DataLoader,
    test_new_loader: DataLoader,
    base_epochs: int,
    standard_lora_rank: int,
    standard_lora_epochs: int,
    sequential_paths: Dict[str, Sequence[int]],
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """End-to-end vision sweep used by ``run_vision_experiment.py``.

    Returns a JSON-serialisable dict with one entry per experimental run.
    """
    torch.manual_seed(seed)

    device = _device()
    print(f"[{dataset_name}] using device: {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")
    print(f"[{dataset_name}] training base model ({base_epochs} epochs)...")
    base_model = train_base_model(
        input_size=input_size,
        num_classes=num_classes,
        train_loader=train_old_loader,
        epochs=base_epochs,
        lr=lr,
        seed=seed,
    )
    base_acc_new = evaluate_accuracy(base_model, test_new_loader)
    print(f"  base accuracy on new classes: {base_acc_new:.4f}")

    print(f"[{dataset_name}] training standard LoRA r={standard_lora_rank} "
          f"({standard_lora_epochs} epochs)...")
    std_lora = train_standard_lora(
        base_model=base_model,
        train_loader=train_new_loader,
        rank=standard_lora_rank,
        epochs=standard_lora_epochs,
        lr=lr,
        seed=seed,
    )
    std_acc = evaluate_accuracy(std_lora, test_new_loader)
    std_trainable = std_lora.num_trainable_params()
    print(f"  standard LoRA accuracy: {std_acc:.4f}")

    sequential_results: Dict[str, dict] = {}
    for name, alloc in sequential_paths.items():
        print(f"[{dataset_name}] sequential path {name} = {alloc}...")
        path_result = train_sequential_lora_path(
            base_model=base_model,
            train_loader=train_new_loader,
            test_loader=test_new_loader,
            epoch_allocation=alloc,
            lr=lr,
            seed=seed,
        )
        sequential_results[name] = {
            "epoch_allocation": list(alloc),
            "component_accuracies": path_result.component_accuracies,
            "cumulative_epochs": path_result.cumulative_epochs,
            "trainable_params_per_component": path_result.trainable_params_per_component,
            "final_accuracy": path_result.final_accuracy,
            "total_epochs": path_result.total_epochs,
        }
        print(f"  -> final acc: {path_result.final_accuracy:.4f}")

    return {
        "dataset": dataset_name,
        "base_model": {
            "epochs": base_epochs,
            "test_acc_new_classes": base_acc_new,
            "num_total_params": base_model.num_total_params(),
        },
        "standard_lora": {
            "rank": standard_lora_rank,
            "epochs": standard_lora_epochs,
            "test_acc": std_acc,
            "trainable_params": std_trainable,
        },
        "sequential_paths": sequential_results,
        "config": {
            "lr": lr,
            "seed": seed,
        },
        "flops": {
            "architecture": "vision_mlp",
            "layers": vision_mlp_layers(input_size, num_classes),
            "samples_per_epoch": len(train_new_loader.dataset),
        },
    }
