"""Simple feedforward classifier with optional sequential rank-1 LoRA components.

This is a cleaned-up version of the ``SimpleLoRAModel`` from
``messy_random_code/Sequential_rank_1.ipynb`` (Cell 15).  The only
substantive differences are:

* the input dimension and number of output classes are configurable so the
  same code works for MNIST (784 / 10) and CIFAR-10/100 (3072 / 10 or 100);
* the model is a plain ``nn.Module`` (no PyTorch Lightning dependency); the
  training loop in :mod:`lora_lib.vision_train` drives it directly.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# A "fixed component" for a single LoRA-adapted layer is a (a_vec, b_vec) pair
# with shapes (in_features, 1) and (1, out_features).
LoRAComponent = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class SimpleLoRAModel(nn.Module):
    """Three-layer MLP (input -> 512 -> 512 -> num_classes) with optional LoRA.

    Each linear layer can have ``rank`` sequential rank-1 components added to
    it.  The semantics match Section 3 of the paper:

        h = (W_0 + sum_k a_k @ b_k) x

    Components passed in via ``fixed_components`` are copied in and frozen so
    only the freshly-added k-th component is trained.  ``rank=0`` reproduces
    the unadapted base network.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 512,
        rank: int = 0,
        base_network: Optional["SimpleLoRAModel"] = None,
        fixed_components: Optional[Sequence[LoRAComponent]] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.lora_rank = rank

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

        if base_network is not None:
            with torch.no_grad():
                self.fc1.weight.copy_(base_network.fc1.weight)
                self.fc1.bias.copy_(base_network.fc1.bias)
                self.fc2.weight.copy_(base_network.fc2.weight)
                self.fc2.bias.copy_(base_network.fc2.bias)
                if base_network.fc3.weight.shape == self.fc3.weight.shape:
                    self.fc3.weight.copy_(base_network.fc3.weight)
                    self.fc3.bias.copy_(base_network.fc3.bias)
            for p in (self.fc1.weight, self.fc1.bias,
                      self.fc2.weight, self.fc2.bias,
                      self.fc3.weight, self.fc3.bias):
                p.requires_grad = False

        if rank > 0:
            self.lora_A1 = nn.ParameterList(
                [nn.Parameter(torch.randn(input_size, 1) * 0.01) for _ in range(rank)]
            )
            self.lora_B1 = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, hidden_size)) for _ in range(rank)]
            )
            self.lora_A2 = nn.ParameterList(
                [nn.Parameter(torch.randn(hidden_size, 1) * 0.01) for _ in range(rank)]
            )
            self.lora_B2 = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, hidden_size)) for _ in range(rank)]
            )
            self.lora_A3 = nn.ParameterList(
                [nn.Parameter(torch.randn(hidden_size, 1) * 0.01) for _ in range(rank)]
            )
            self.lora_B3 = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, num_classes)) for _ in range(rank)]
            )

            if fixed_components is not None:
                for i, comp in enumerate(fixed_components):
                    if i >= rank:
                        break
                    a1, b1, a2, b2, a3, b3 = comp
                    with torch.no_grad():
                        self.lora_A1[i].copy_(a1)
                        self.lora_B1[i].copy_(b1)
                        self.lora_A2[i].copy_(a2)
                        self.lora_B2[i].copy_(b2)
                        if b3.shape[1] == num_classes:
                            self.lora_A3[i].copy_(a3)
                            self.lora_B3[i].copy_(b3)
                    for p in (self.lora_A1[i], self.lora_B1[i],
                              self.lora_A2[i], self.lora_B2[i],
                              self.lora_A3[i], self.lora_B3[i]):
                        p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)

        h1 = self.fc1(x)
        if self.lora_rank > 0:
            for i in range(self.lora_rank):
                h1 = h1 + (x @ self.lora_A1[i]) @ self.lora_B1[i]
        h1 = F.relu(h1)

        h2 = self.fc2(h1)
        if self.lora_rank > 0:
            for i in range(self.lora_rank):
                h2 = h2 + (h1 @ self.lora_A2[i]) @ self.lora_B2[i]
        h2 = F.relu(h2)

        out = self.fc3(h2)
        if self.lora_rank > 0:
            for i in range(self.lora_rank):
                out = out + (h2 @ self.lora_A3[i]) @ self.lora_B3[i]
        return out

    def get_components(self) -> List[LoRAComponent]:
        """Return a frozen snapshot of the current rank-1 components."""
        comps: List[LoRAComponent] = []
        for i in range(self.lora_rank):
            comps.append(
                (
                    self.lora_A1[i].detach().clone(),
                    self.lora_B1[i].detach().clone(),
                    self.lora_A2[i].detach().clone(),
                    self.lora_B2[i].detach().clone(),
                    self.lora_A3[i].detach().clone(),
                    self.lora_B3[i].detach().clone(),
                )
            )
        return comps

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
