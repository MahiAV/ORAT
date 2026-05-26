"""LoRA layer + helpers for adapting DistilBERT to SST-2.

This is a cleaned-up version of the ``LoRALayer`` / ``prepare_lora_model``
pair from ``messy_random_code/Experiments_seq_rank1.ipynb``.  Two modes are
supported:

* ``is_sequential=False`` — a single (A, B) matrix pair of the requested
  rank is added on top of the original layer (standard LoRA).
* ``is_sequential=True`` — rank-1 ``(a_k, b_k)`` components are appended one
  at a time via :meth:`LoRALayer.add_rank_one_component`.  Older components
  can be frozen so only the newest one is trained.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """LoRA wrapper around a frozen ``nn.Linear``."""

    def __init__(self, original_layer: nn.Linear, rank: int) -> None:
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Standard LoRA (jointly optimised) parameters.  Initialised so the
        # adapter is a no-op at the start of training (A random, B zero).
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_features))

        # Sequential rank-1 components, populated lazily.
        self.a_components = nn.ParameterList()
        self.b_components = nn.ParameterList()

        self.is_sequential = False

    def add_rank_one_component(self) -> None:
        """Append a fresh rank-1 component (only the newest one is trainable)."""
        in_features = self.original_layer.in_features
        out_features = self.original_layer.out_features
        self.a_components.append(nn.Parameter(torch.randn(in_features, 1) * 0.01))
        self.b_components.append(nn.Parameter(torch.zeros(1, out_features)))

    def set_sequential_mode(self, mode: bool = True) -> None:
        self.is_sequential = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original_layer(x)
        if self.is_sequential:
            for a, b in zip(self.a_components, self.b_components):
                result = result + (x @ a) @ b
        else:
            result = result + (x @ self.A) @ self.B
        return result


def prepare_lora_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    rank: int = 3,
    sequential: bool = False,
):
    """Load DistilBERT and wrap its q_lin/v_lin with :class:`LoRALayer`s.

    All original parameters (including the classifier head) are frozen — only
    the LoRA parameters that are explicitly enabled by the caller are
    trainable.  In sequential mode no components exist yet; the training loop
    is responsible for calling :meth:`LoRALayer.add_rank_one_component`.
    """
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    for p in model.parameters():
        p.requires_grad = False

    for layer in model.distilbert.transformer.layer:
        layer.attention.q_lin = LoRALayer(layer.attention.q_lin, rank)
        layer.attention.v_lin = LoRALayer(layer.attention.v_lin, rank)
        if sequential:
            layer.attention.q_lin.set_sequential_mode(True)
            layer.attention.v_lin.set_sequential_mode(True)

    # In sequential mode the standard A/B matrices are never used in the
    # forward pass; freeze them so the trainable-parameter count reflects
    # what is actually being optimised.
    if sequential:
        for module in model.modules():
            if isinstance(module, LoRALayer):
                module.A.requires_grad = False
                module.B.requires_grad = False

    return model


def enable_standard_lora_training(model) -> None:
    """Mark the ``A`` / ``B`` matrices on every :class:`LoRALayer` as trainable."""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.A.requires_grad = True
            module.B.requires_grad = True


def add_next_sequential_component(model, k: int) -> None:
    """Add the (k+1)-th rank-1 component to every :class:`LoRALayer` in ``model``.

    Components 0..k-1 are frozen, component k is left trainable.
    """
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.add_rank_one_component()
            for i in range(k):
                module.a_components[i].requires_grad = False
                module.b_components[i].requires_grad = False
            module.a_components[k].requires_grad = True
            module.b_components[k].requires_grad = True


def count_parameters(model) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
