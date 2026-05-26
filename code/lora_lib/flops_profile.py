"""Empirical FLOPs measurement via :class:`torch.utils.flop_counter.FlopCounterMode`.

The analytical estimator in :mod:`lora_lib.flops` is convenient because it
needs nothing but layer shapes — but it ignores activation sizes, attention
softmax, layer-norm, biases, and the host of other small ops that happen
during a real training step.  This module complements it by *measuring*
FLOPs for one fwd+bwd step using PyTorch's dispatcher-level counter
(``torch.utils.flop_counter.FlopCounterMode``).

Two measurement primitives are exposed:

* :func:`profile_vision_step` — builds a fresh ``SimpleLoRAModel`` whose
  active LoRA components and ``requires_grad`` flags match a (joint or
  sequential) configuration and runs a single batch through it.
* :func:`profile_nlp_step` — same idea for the LoRA-adapted DistilBERT
  (random ``input_ids`` are fed; we do not need the real tokenizer).

Both return *per-sample* FLOPs (vision) or *per-token* FLOPs (NLP) so the
numbers are directly comparable to the analytical formulas in
:mod:`lora_lib.flops`.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn.functional as F


def _device(explicit: Optional[str]) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _freeze(params: Iterable[torch.nn.Parameter]) -> None:
    for p in params:
        p.requires_grad = False


@contextlib.contextmanager
def _flop_counter():
    """Context manager that yields a fresh ``FlopCounterMode`` instance."""
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with counter:
        yield counter


# --------------------------------------------------------------------------- #
# Vision profiler
# --------------------------------------------------------------------------- #


def _build_vision_model(
    input_size: int,
    num_classes: int,
    r_active: int,
    r_train: int,
    device: torch.device,
):
    """Construct a :class:`SimpleLoRAModel` whose grad flags match (r_active, r_train)."""
    from .vision_model import SimpleLoRAModel

    model = SimpleLoRAModel(input_size, num_classes, rank=r_active).to(device)

    # Always freeze the base MLP (LoRA is the only thing being trained).
    _freeze(model.fc1.parameters())
    _freeze(model.fc2.parameters())
    _freeze(model.fc3.parameters())

    # In sequential mode we freeze the (r_active - r_train) earliest components
    # and leave the last r_train trainable.  Joint LoRA ⇒ r_train == r_active
    # ⇒ nothing extra to freeze.
    n_freeze = max(0, r_active - r_train)
    if r_active > 0 and n_freeze > 0:
        for plist in (
            model.lora_A1, model.lora_B1,
            model.lora_A2, model.lora_B2,
            model.lora_A3, model.lora_B3,
        ):
            _freeze(plist[i] for i in range(n_freeze))

    # Special case: rank=0 has zero trainable params, so a real backward
    # would crash.  Re-enable the FIRST layer's bias — that forces autograd
    # to backpropagate through every hidden layer to reach it (just like
    # a real LoRA component attached at the bottom of the stack would).
    # Picking the *last* layer's bias would be wrong here: autograd would
    # skip the backward through the hidden layers entirely.
    if r_active == 0:
        model.fc1.bias.requires_grad = True
    return model


def profile_vision_step(
    *,
    input_size: int,
    num_classes: int,
    r_active: int,
    r_train: int,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Measure per-sample FLOPs for one fwd+bwd step of the vision MLP.

    Returns a dict with ``per_sample_flops`` (forward + backward) and
    ``batch_size``.  The model is freshly built; no real data is needed.
    """
    dev = _device(device)
    model = _build_vision_model(input_size, num_classes, r_active, r_train, dev)
    x = torch.randn(batch_size, input_size, device=dev)
    y = torch.randint(0, num_classes, (batch_size,), device=dev)

    # One un-counted warmup pass to build any kernel caches.
    out = model(x)
    F.cross_entropy(out, y).backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    with _flop_counter() as counter:
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
    total = counter.get_total_flops()
    return {"per_sample_flops": total / batch_size, "batch_size": batch_size, "total_flops": total}


# --------------------------------------------------------------------------- #
# NLP profiler
# --------------------------------------------------------------------------- #


def _build_nlp_model(
    *,
    r_active: int,
    r_train: int,
    model_name: str,
    device: torch.device,
):
    from .nlp_model import (
        add_next_sequential_component,
        enable_standard_lora_training,
        prepare_lora_model,
    )

    if r_active == 0:
        # No LoRA at all — purely the frozen base.  Still useful as a baseline.
        model = prepare_lora_model(model_name=model_name, rank=0, sequential=False)
    elif r_train == r_active:
        # Joint LoRA: the standard A/B matrices of rank r_active are the trainable knobs.
        model = prepare_lora_model(model_name=model_name, rank=r_active, sequential=False)
        enable_standard_lora_training(model)
    else:
        # Sequential: r_active total rank-1 components, only the last is trainable.
        model = prepare_lora_model(model_name=model_name, rank=0, sequential=True)
        for k in range(r_active):
            add_next_sequential_component(model, k)
    model = model.to(device)

    # Special case: rank=0 has zero trainable params.  We need to force
    # backward to flow through *exactly* the same depth as a real LoRA run
    # would — that's down to the q_lin/v_lin of the *first* transformer
    # layer.  Going deeper (e.g. to the embedding LayerNorm) would
    # over-charge the base; going shallower would under-charge it.
    if r_active == 0:
        first_q_lin = model.distilbert.transformer.layer[0].attention.q_lin
        # `q_lin` is wrapped by LoRALayer; the underlying Linear is .original_layer.
        target = getattr(first_q_lin, "original_layer", first_q_lin)
        target.bias.requires_grad = True
    return model


def profile_nlp_step(
    *,
    r_active: int,
    r_train: int,
    batch_size: int = 8,
    max_length: int = 128,
    model_name: str = "distilbert-base-uncased",
    device: Optional[str] = None,
    vocab_size: int = 30522,
    num_labels: int = 2,
) -> Dict[str, float]:
    """Measure per-token FLOPs for one fwd+bwd step of LoRA-adapted DistilBERT.

    Returns a dict with ``per_token_flops``, ``per_sample_flops``,
    ``batch_size`` and ``max_length``.  Uses random token IDs so it does
    not need the SST-2 dataset to be downloaded.
    """
    dev = _device(device)
    model = _build_nlp_model(r_active=r_active, r_train=r_train, model_name=model_name, device=dev)
    input_ids = torch.randint(0, vocab_size, (batch_size, max_length), device=dev)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, num_labels, (batch_size,), device=dev)

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    out.loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    with _flop_counter() as counter:
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
    total = counter.get_total_flops()
    return {
        "per_token_flops": total / (batch_size * max_length),
        "per_sample_flops": total / batch_size,
        "batch_size": batch_size,
        "max_length": max_length,
        "total_flops": total,
    }


# --------------------------------------------------------------------------- #
# Convenience: profile every (rank, joint/seq) pair we need for the figure
# --------------------------------------------------------------------------- #


def profile_per_rank_grid_vision(
    *,
    input_size: int,
    num_classes: int,
    max_rank: int,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """For each rank ``r`` in ``1..max_rank`` profile both joint and sequential.

    Also profiles ``r=0`` (no LoRA) which gives the *empirical* frozen-base
    cost of the whole model — useful for NLP where the analytical formula
    only accounts for the LoRA-adapted layers (q_lin, v_lin) and ignores
    the rest (FFN, attention K/O, classifier head, layer norms, ...).

    Returns ``{"base_per_sample", "joint": {r: per_sample}, "sequential": {r: per_sample}}``.
    """
    base = profile_vision_step(
        input_size=input_size, num_classes=num_classes,
        r_active=0, r_train=0, batch_size=batch_size, device=device,
    )["per_sample_flops"]
    out: Dict[str, Any] = {"base_per_sample": base, "joint": {}, "sequential": {}}
    for r in range(1, max_rank + 1):
        out["joint"][r] = profile_vision_step(
            input_size=input_size, num_classes=num_classes,
            r_active=r, r_train=r, batch_size=batch_size, device=device,
        )["per_sample_flops"]
        out["sequential"][r] = profile_vision_step(
            input_size=input_size, num_classes=num_classes,
            r_active=r, r_train=1, batch_size=batch_size, device=device,
        )["per_sample_flops"]
    return out


def profile_per_rank_grid_nlp(
    *,
    max_rank: int,
    batch_size: int = 8,
    max_length: int = 128,
    model_name: str = "distilbert-base-uncased",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Same as :func:`profile_per_rank_grid_vision` but per-token for DistilBERT."""
    base = profile_nlp_step(
        r_active=0, r_train=0, batch_size=batch_size, max_length=max_length,
        model_name=model_name, device=device,
    )["per_token_flops"]
    out: Dict[str, Any] = {"base_per_sample": base, "joint": {}, "sequential": {}}
    for r in range(1, max_rank + 1):
        out["joint"][r] = profile_nlp_step(
            r_active=r, r_train=r, batch_size=batch_size, max_length=max_length,
            model_name=model_name, device=device,
        )["per_token_flops"]
        out["sequential"][r] = profile_nlp_step(
            r_active=r, r_train=1, batch_size=batch_size, max_length=max_length,
            model_name=model_name, device=device,
        )["per_token_flops"]
    return out
