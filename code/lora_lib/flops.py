"""Analytical FLOPs estimator for LoRA-adapted linear layers.

Conventions
-----------
We count multiply-adds as 2 FLOPs each and use the standard "matmul of
shape (B, K) @ (K, M) costs 2·B·K·M FLOPs" approximation.  For a single
``nn.Linear`` of shape ``(d_in, d_out)`` evaluated on a batch of size ``B``:

* **Forward (frozen base only)** — ``2·B·d_in·d_out``
* **Forward LoRA addition (rank ``r``)** — decomposed as ``(x A) B`` with
  ``A ∈ ℝ^{d_in × r}`` and ``B ∈ ℝ^{r × d_out}``:
  ``2·B·d_in·r + 2·B·r·d_out  =  2·B·r·(d_in + d_out)``
* **Backward through frozen base** — input-gradient ``W^T dy`` only:
  ``2·B·d_in·d_out``
* **Backward through LoRA (input gradient, ``r`` active)** — same shape
  as forward LoRA: ``2·B·r·(d_in + d_out)``
* **LoRA parameter gradients** (only for the ``r_train`` components that
  have ``requires_grad=True``) — ``2·B·r_train·(d_in + d_out)``
  (the cost of computing ``dA`` plus ``dB`` together)

Putting it together for one linear layer with ``r_active`` LoRA components
in the forward path and ``r_train`` of them being optimised this step:

    fwd          = 2·d_in·d_out + 2·r_active·(d_in + d_out)
    bwd_input    = 2·d_in·d_out + 2·r_active·(d_in + d_out)
    bwd_params   = 2·r_train·(d_in + d_out)

Per sample total:
    base_part = 4·d_in·d_out
    lora_part = (4·r_active + 2·r_train)·(d_in + d_out)

Joint LoRA of rank ``r`` has ``r_active = r_train = r`` ⇒ LoRA coefficient
``6·r``.  Sequential at component ``k`` (with components ``0..k-2`` frozen
and only the new one trainable) has ``r_active = k`` and ``r_train = 1``
⇒ LoRA coefficient ``4·k + 2``.

The same formula is reused below for each LoRA-adapted layer.  For
DistilBERT/SST-2 we count the LoRA additions on every transformer block's
``q_lin`` and ``v_lin`` (see :func:`distilbert_lora_layers`); the rest of
the model (frozen embedding + FFN + classifier head) contributes a
constant base cost that cancels out across configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


# Per-sample LoRA coefficient — multiplied by ``Σ (d_in + d_out)`` over the
# adapted layers to get LoRA-related FLOPs per sample.
def _lora_coeff(r_active: int, r_train: int) -> int:
    return 4 * r_active + 2 * r_train


def per_sample_flops(
    layers: Sequence[Tuple[int, int]],
    *,
    r_active: int,
    r_train: int,
    include_base: bool = True,
    first_layer_input_grad: bool = False,
) -> int:  # noqa: D401 — see __doc__ above
    """FLOPs per sample for one fwd+bwd pass through a stack of LoRA-adapted layers.

    ``layers`` is an iterable of ``(d_in, d_out)`` tuples.  ``include_base``
    toggles the frozen base contribution.  ``first_layer_input_grad``
    controls whether the first layer pays the cost of backpropagating an
    input gradient — in practice this is ``False`` (the model input is a
    tensor with ``requires_grad=False``), which is what
    :class:`torch.utils.flop_counter.FlopCounterMode` actually measures.
    """
    total = 0
    for idx, (d_in, d_out) in enumerate(layers):
        bd = per_sample_flops_breakdown(
            [(d_in, d_out)],
            r_active=r_active,
            r_train=r_train,
            input_has_grad=first_layer_input_grad if idx == 0 else True,
        )
        total += (bd.base if include_base else 0) + bd.lora_total
    return total


@dataclass
class StepFlops:
    """Per-sample FLOPs for one training step, broken into base / LoRA fwd / LoRA bwd."""

    base: int
    lora_forward: int
    lora_backward_input: int
    lora_backward_params: int

    @property
    def lora_total(self) -> int:
        return self.lora_forward + self.lora_backward_input + self.lora_backward_params

    @property
    def total(self) -> int:
        return self.base + self.lora_total


def per_sample_flops_breakdown(
    layers: Sequence[Tuple[int, int]],
    *,
    r_active: int,
    r_train: int,
    input_has_grad: bool = False,
) -> StepFlops:
    """Same as :func:`per_sample_flops` but returns the four sub-components.

    ``input_has_grad`` is a single flag for the first layer.  Defaults to
    ``False`` because in practice the model input is a leaf tensor without
    a gradient — that's what ``FlopCounterMode`` actually measures, and
    that's the only configuration where analytical and profiled numbers
    agree.  Pass ``True`` for an upper-bound estimate when the input
    really does need a gradient (e.g. adversarial training).
    """
    base = 0
    lora_fwd = 0
    lora_bwd_input = 0
    lora_bwd_params = 0
    for idx, (d_in, d_out) in enumerate(layers):
        is_first = idx == 0
        # Frozen base linear:
        #   fwd      : 2·d_in·d_out (always)
        #   bwd input: 2·d_in·d_out (only if upstream needs a gradient,
        #              i.e. not the very first layer when input has no grad)
        if is_first and not input_has_grad:
            base += 2 * d_in * d_out
        else:
            base += 4 * d_in * d_out
        # LoRA additive branch shares the input gradient of its host layer.
        lora_fwd += 2 * r_active * (d_in + d_out)
        if not (is_first and not input_has_grad):
            lora_bwd_input += 2 * r_active * (d_in + d_out)
        lora_bwd_params += 2 * r_train * (d_in + d_out)
    return StepFlops(
        base=base,
        lora_forward=lora_fwd,
        lora_backward_input=lora_bwd_input,
        lora_backward_params=lora_bwd_params,
    )


# --------------------------------------------------------------------------- #
# Per-experiment helpers
# --------------------------------------------------------------------------- #


def epoch_flops(
    layers: Sequence[Tuple[int, int]],
    samples_per_epoch: int,
    *,
    r_active: int,
    r_train: int,
) -> int:
    """FLOPs spent on one full pass through the training set."""
    return samples_per_epoch * per_sample_flops(layers, r_active=r_active, r_train=r_train)


def standard_lora_total_flops(
    layers: Sequence[Tuple[int, int]],
    samples_per_epoch: int,
    rank: int,
    epochs: int,
) -> int:
    """Total FLOPs for jointly-trained LoRA of given rank and epoch budget."""
    return epochs * epoch_flops(layers, samples_per_epoch, r_active=rank, r_train=rank)


def sequential_path_total_flops(
    layers: Sequence[Tuple[int, int]],
    samples_per_epoch: int,
    epoch_allocation: Sequence[int],
) -> List[int]:
    """Per-component FLOPs for one α-β-γ sequential path.

    Returns a list with one entry per component: ``flops[k]`` is the cost
    of training the ``(k+1)``-th component for ``epoch_allocation[k]``
    epochs while components ``0..k-1`` are frozen.
    """
    out: List[int] = []
    for k, n_epochs in enumerate(epoch_allocation, start=1):
        # k-th component is being trained; total active rank in fwd = k
        out.append(n_epochs * epoch_flops(layers, samples_per_epoch, r_active=k, r_train=1))
    return out


# --------------------------------------------------------------------------- #
# Convenience constructors for the two architectures we use
# --------------------------------------------------------------------------- #


def vision_mlp_layers(input_size: int, num_classes: int, hidden_size: int = 512) -> List[Tuple[int, int]]:
    """Return the per-layer (d_in, d_out) for ``SimpleLoRAModel``."""
    return [(input_size, hidden_size), (hidden_size, hidden_size), (hidden_size, num_classes)]


def distilbert_lora_layers(
    num_layers: int = 6,
    hidden_size: int = 768,
    adapt_modules: Sequence[str] = ("q_lin", "v_lin"),
) -> List[Tuple[int, int]]:
    """Return the per-layer (d_in, d_out) for the DistilBERT layers we adapt."""
    return [(hidden_size, hidden_size)] * (num_layers * len(adapt_modules))


# --------------------------------------------------------------------------- #
# Result-JSON-driven summarisers used by the figure script
# --------------------------------------------------------------------------- #


def summarise_results_flops(results: dict, samples_per_epoch: int) -> dict:
    """Compute per-step + total FLOPs for every run in a results JSON.

    The JSON must contain a ``"flops"`` block (written by ``vision_train``
    / ``nlp_train``).  ``samples_per_epoch`` should reflect the *new*-class
    training set for vision results and the SST-2 train split for NLP.
    """
    flops_meta = results.get("flops", {})
    layers = [tuple(pair) for pair in flops_meta["layers"]]
    samples = samples_per_epoch or flops_meta.get("samples_per_epoch")
    if samples is None:
        raise ValueError("samples_per_epoch is required (no fallback in JSON)")

    summary = {
        "layers": layers,
        "samples_per_epoch": samples,
        "configurations": [],
    }

    # Joint baseline.
    std = results["standard_lora"]
    rank = std["rank"]
    epochs = std["epochs"]
    breakdown = per_sample_flops_breakdown(layers, r_active=rank, r_train=rank)
    summary["configurations"].append(
        {
            "label": f"Joint LoRA r={rank}",
            "kind": "joint",
            "rank": rank,
            "epochs": epochs,
            "per_step_flops": breakdown.total,
            "per_step_breakdown": breakdown.__dict__,
            "total_flops": breakdown.total * samples * epochs,
        }
    )

    # Sequential paths.
    for name, run in results.get("sequential_paths", {}).items():
        alloc = run["epoch_allocation"]
        components: List[dict] = []
        path_total = 0
        for k, n_epochs in enumerate(alloc, start=1):
            bd = per_sample_flops_breakdown(layers, r_active=k, r_train=1)
            comp_total = bd.total * samples * n_epochs
            path_total += comp_total
            components.append(
                {
                    "component": k,
                    "epochs": n_epochs,
                    "per_step_flops": bd.total,
                    "per_step_breakdown": bd.__dict__,
                    "total_flops": comp_total,
                }
            )
        summary["configurations"].append(
            {
                "label": f"Seq {name}",
                "kind": "sequential",
                "epoch_allocation": list(alloc),
                "components": components,
                "total_flops": path_total,
            }
        )

    return summary
