"""FLOPs comparison: joint LoRA vs sequential rank-1 LoRA.

This figure addresses the question "does training a single rank-1 component
take the same FLOPs per epoch as training a joint LoRA?"  Short answer:
*almost* — the frozen base dominates total compute, but in pure LoRA terms
sequential is meaningfully cheaper at every rank ``> 1``.

Three panels:

* **Left — per-sample total FLOPs by rank (with frozen base).**  For each
  rank ``r`` we compare the per-sample cost of (a) jointly training a
  rank-``r`` LoRA versus (b) sequentially training the ``r``-th rank-1
  component with the previous ``r-1`` frozen.  Bars are stacked by
  sub-cost (frozen base / LoRA forward / LoRA parameter gradient).  The
  base is enormous compared to LoRA so the visible difference is small.

* **Middle — per-sample LoRA-only FLOPs by rank (base subtracted).**
  Same comparison but the frozen base is removed so the actual joint vs
  sequential difference is visible.  Joint LoRA scales as ``6·r``, while
  sequential scales as ``4·r + 2`` — savings grow with rank.

* **Right — total experiment FLOPs.**  Full training cost across every
  configuration that was actually run in the supplied results JSON
  (jointly trained rank-3 baseline plus every sequential α→β→γ schedule).

When ``--profile`` is enabled the script *also* runs PyTorch's
``FlopCounterMode`` on a freshly-built model for each (rank, joint/seq)
pair and overlays the measured totals as black tick marks on the
per-step panels.  Analytical and profiled numbers usually agree to
within a few percent — the analytical formula is the easy-to-read
ground truth, the profiled numbers verify it on a real model.

Usage
-----
    # Analytical only (instant)
    python figure_flops_comparison.py \\
        --results results/vision_mnist_10-10-10.json \\
        --output figures/figure_flops_mnist.png

    # Analytical + profiled (vision profile takes ~3s, NLP profile ~30s)
    python figure_flops_comparison.py \\
        --results results/vision_cifar10_10-10-10.json \\
        --output figures/figure_flops_cifar10.png \\
        --profile auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from lora_lib.flops import (
    distilbert_lora_layers,
    per_sample_flops_breakdown,
    summarise_results_flops,
    vision_mlp_layers,
)
from lora_lib.plot_style import xkcd_style


# Standard SST-2 sizes (used as fallback when the JSON does not carry FLOPs metadata).
_SST2_DEFAULT_TRAIN_SIZE = 67349
_SST2_DEFAULT_MAX_LENGTH = 128


def _infer_flops_metadata(results: dict) -> dict:
    """Inject a ``flops`` block if the JSON was generated before that field existed."""
    if "flops" in results:
        return results

    dataset = results.get("dataset", "").lower()
    if dataset in ("mnist", "fashion_mnist"):
        layers = vision_mlp_layers(input_size=28 * 28, num_classes=10)
        # 5 "new" classes in MNIST, ~6k examples each → ~30k samples per epoch.
        samples = 30_000
        results["flops"] = {"architecture": "vision_mlp", "layers": layers, "samples_per_epoch": samples}
    elif dataset == "cifar10":
        layers = vision_mlp_layers(input_size=3 * 32 * 32, num_classes=10)
        # 5 "new" classes × 5,000 train images each.
        results["flops"] = {"architecture": "vision_mlp", "layers": layers, "samples_per_epoch": 25_000}
    elif dataset == "cifar100":
        layers = vision_mlp_layers(input_size=3 * 32 * 32, num_classes=100)
        # 50 "new" classes × 500 train images each.
        results["flops"] = {"architecture": "vision_mlp", "layers": layers, "samples_per_epoch": 25_000}
    elif dataset == "sst2":
        results["flops"] = {
            "architecture": "distilbert_qv",
            "layers": distilbert_lora_layers(),
            "samples_per_epoch": _SST2_DEFAULT_TRAIN_SIZE * _SST2_DEFAULT_MAX_LENGTH,
            "num_train_examples": _SST2_DEFAULT_TRAIN_SIZE,
            "max_length": _SST2_DEFAULT_MAX_LENGTH,
        }
    else:
        raise ValueError(
            f"Cannot infer FLOPs metadata for dataset={dataset!r}; please rerun the "
            "experiment with the updated runner so the results JSON carries a 'flops' block."
        )
    return results


# --------------------------------------------------------------------------- #
# Per-step FLOPs panels
# --------------------------------------------------------------------------- #


_BASE_COLOR = "#cfcfcf"
_FWD_COLOR = "#7BAFD4"
_BWD_COLOR = "#F18F01"


def _gather_per_step(layers, max_rank: int, *, base_override: float | None = None):
    """Return joint and sequential per-sample breakdowns for ranks 1..max_rank.

    Each returned tuple is ``(base, lora_forward+input_grad, lora_param_grad)``
    in raw FLOPs (not Giga).  ``base_override`` (if given) replaces the
    analytical base — useful when we have a profiled rank=0 measurement
    that captures parts of the model the analytical formula ignores
    (FFN, attention K/O, classifier head, ...).
    """
    joint, seq = [], []
    for r in range(1, max_rank + 1):
        bd_joint = per_sample_flops_breakdown(layers, r_active=r, r_train=r)
        bd_seq = per_sample_flops_breakdown(layers, r_active=r, r_train=1)
        base = base_override if base_override is not None else bd_joint.base
        joint.append(
            (
                base,
                bd_joint.lora_forward + bd_joint.lora_backward_input,
                bd_joint.lora_backward_params,
            )
        )
        seq.append(
            (
                base,
                bd_seq.lora_forward + bd_seq.lora_backward_input,
                bd_seq.lora_backward_params,
            )
        )
    return joint, seq


def _draw_grouped_bars(
    ax,
    *,
    ranks,
    joint_breakdown,
    seq_breakdown,
    include_base: bool,
    unit_scale: float,
    unit_label: str,
    title: str,
    show_legend: bool,
    profiled_joint=None,
    profiled_seq=None,
    base_per_sample: float = 0.0,
) -> None:
    """Render the grouped bar chart.

    ``profiled_joint`` / ``profiled_seq`` are optional lists of measured
    per-sample FLOPs (one entry per rank).  When provided they are drawn
    as black tick marks on top of each bar, so the user can compare the
    analytical stack with the real measurement.  When ``include_base`` is
    False we strip ``base_per_sample`` from the profiled totals so the
    overlay matches the LoRA-only view.
    """
    bar_width = 0.38
    base_x = np.arange(len(ranks))

    def _scaled(idx):
        return np.asarray([row[idx] for row in joint_breakdown], dtype=float) / unit_scale

    def _scaled_seq(idx):
        return np.asarray([row[idx] for row in seq_breakdown], dtype=float) / unit_scale

    j_base = _scaled(0) if include_base else np.zeros(len(ranks))
    j_fwd = _scaled(1)
    j_bwd = _scaled(2)
    s_base = _scaled_seq(0) if include_base else np.zeros(len(ranks))
    s_fwd = _scaled_seq(1)
    s_bwd = _scaled_seq(2)

    if include_base:
        ax.bar(base_x - bar_width / 2, j_base, bar_width, color=_BASE_COLOR, edgecolor="white",
               label="Frozen base (fwd+bwd)")
        ax.bar(base_x + bar_width / 2, s_base, bar_width, color=_BASE_COLOR, edgecolor="white", hatch="//")

    ax.bar(base_x - bar_width / 2, j_fwd, bar_width, bottom=j_base, color=_FWD_COLOR, edgecolor="white",
           label="LoRA forward + input grad")
    ax.bar(base_x - bar_width / 2, j_bwd, bar_width, bottom=j_base + j_fwd, color=_BWD_COLOR, edgecolor="white",
           label="LoRA parameter grad")
    ax.bar(base_x + bar_width / 2, s_fwd, bar_width, bottom=s_base, color=_FWD_COLOR, edgecolor="white", hatch="//")
    ax.bar(base_x + bar_width / 2, s_bwd, bar_width, bottom=s_base + s_fwd, color=_BWD_COLOR, edgecolor="white",
           hatch="//")

    j_total = j_base + j_fwd + j_bwd
    s_total = s_base + s_fwd + s_bwd

    # Overlay measured (profiled) totals as horizontal tick marks.
    if profiled_joint is not None and profiled_seq is not None:
        scale_to_unit = 1.0 / unit_scale
        offset = (base_per_sample if not include_base else 0.0) * scale_to_unit
        prof_j = np.asarray(profiled_joint, dtype=float) * scale_to_unit - offset
        prof_s = np.asarray(profiled_seq, dtype=float) * scale_to_unit - offset

        for i, _ in enumerate(ranks):
            for x_center, y_val in (
                (base_x[i] - bar_width / 2, prof_j[i]),
                (base_x[i] + bar_width / 2, prof_s[i]),
            ):
                ax.hlines(y_val, x_center - bar_width * 0.45, x_center + bar_width * 0.45,
                          colors="black", linewidth=2.0, zorder=5)
                ax.scatter([x_center], [y_val], color="black", s=20, zorder=6)
        # Single legend entry for the profiled marker.
        ax.plot([], [], color="black", marker="o", linestyle="-", linewidth=2.0,
                markersize=5, label="Profiled (PyTorch)")

    y_top = max(j_total.max(), s_total.max()) * 1.20
    if profiled_joint is not None:
        scale = 1.0 / unit_scale
        offset = (base_per_sample if not include_base else 0.0) * scale
        prof_max = max(
            max(profiled_joint) * scale - offset,
            max(profiled_seq) * scale - offset,
        )
        y_top = max(y_top, prof_max * 1.20)

    for i, _ in enumerate(ranks):
        ax.text(base_x[i] - bar_width / 2, j_total[i] * 1.02, "joint", ha="center", va="bottom",
                fontsize=9, color="#333")
        ax.text(base_x[i] + bar_width / 2, s_total[i] * 1.02, "seq", ha="center", va="bottom",
                fontsize=9, color="#333")
        if j_total[i] > 0:
            savings = (j_total[i] - s_total[i]) / j_total[i] * 100
            ax.text(
                base_x[i],
                y_top * 0.92,
                f"-{savings:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                color=_BWD_COLOR,
                fontweight="bold",
            )

    ax.set_xticks(base_x)
    ax.set_xticklabels([f"rank {r}" for r in ranks])
    ax.set_ylabel(f"FLOPs per sample ({unit_label})", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(top=y_top)
    ax.grid(True, axis="y", linestyle="-", alpha=0.18)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=4,
            fontsize=8,
            frameon=True,
            framealpha=0.95,
        )


def _base_only_analytical(layers) -> int:
    """Per-sample FLOPs of the LoRA-adapted layers' frozen base (analytical)."""
    return per_sample_flops_breakdown(layers, r_active=0, r_train=0).base


def _draw_per_step_total(ax, layers, max_rank: int, *, profiled=None) -> None:
    base_override = profiled["base_per_sample"] if profiled is not None else None
    joint, seq = _gather_per_step(layers, max_rank, base_override=base_override)
    pj = ps = None
    if profiled is not None:
        pj = [profiled["joint"][r] for r in range(1, max_rank + 1)]
        ps = [profiled["sequential"][r] for r in range(1, max_rank + 1)]
    title = "Total per-sample FLOPs (incl. frozen base)"
    if profiled is not None:
        title += " — base from profile"
    _draw_grouped_bars(
        ax,
        ranks=list(range(1, max_rank + 1)),
        joint_breakdown=joint,
        seq_breakdown=seq,
        include_base=True,
        unit_scale=1e9,
        unit_label="GFLOPs",
        title=title,
        show_legend=True,
        profiled_joint=pj,
        profiled_seq=ps,
        base_per_sample=0.0,  # don't shift profiled markers in the total view
    )


def _draw_per_step_lora_only(ax, layers, max_rank: int, *, profiled=None) -> None:
    joint, seq = _gather_per_step(layers, max_rank)
    pj = ps = None
    if profiled is not None:
        pj = [profiled["joint"][r] for r in range(1, max_rank + 1)]
        ps = [profiled["sequential"][r] for r in range(1, max_rank + 1)]
    # Subtract the *empirical* base from profiled markers in the LoRA-only view
    # — for vision this matches the analytical base (~ exact agreement);
    # for NLP it removes the ~140 MFLOPs/token spent on the FFN, attention K/O,
    # classifier head etc. that are constant across configurations.
    base_offset = profiled["base_per_sample"] if profiled is not None else 0.0
    _draw_grouped_bars(
        ax,
        ranks=list(range(1, max_rank + 1)),
        joint_breakdown=joint,
        seq_breakdown=seq,
        include_base=False,
        unit_scale=1e6,
        unit_label="MFLOPs",
        title="LoRA-only per-sample FLOPs (base subtracted)",
        show_legend=True,
        profiled_joint=pj,
        profiled_seq=ps,
        base_per_sample=base_offset,
    )


# --------------------------------------------------------------------------- #
# Panel B: total experiment FLOPs
# --------------------------------------------------------------------------- #


def _draw_total_panel(ax, summary: dict) -> None:
    configs = summary["configurations"]
    labels = [c["label"] for c in configs]
    totals_g = np.array([c["total_flops"] for c in configs], dtype=float) / 1e12  # show in TFLOPs
    colors = ["#F0B96A" if c["kind"] == "joint" else "#8F8FE3" for c in configs]

    order = np.argsort(totals_g)
    labels = [labels[i] for i in order]
    totals_g = totals_g[order]
    colors = [colors[i] for i in order]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, totals_g, color=colors, edgecolor="white")
    for i, v in enumerate(totals_g):
        ax.text(v, i, f" {v:.1f} TF", va="center", ha="left", fontsize=9, color="#222")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Total training FLOPs (TFLOPs)", fontsize=13, fontweight="bold")
    ax.set_title("Total experiment FLOPs", fontsize=14, fontweight="bold")
    ax.set_xlim(right=totals_g.max() * 1.18)
    ax.grid(True, axis="x", linestyle="-", alpha=0.18)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_total_lora_only_panel(ax, summary: dict) -> None:
    """Same as _draw_total_panel but with base FLOPs subtracted."""
    configs = summary["configurations"]
    labels = [c["label"] for c in configs]
    layers = summary["layers"]
    samples_per_epoch = summary["samples_per_epoch"]
    
    # Calculate base-only FLOPs per sample (rank=0, train=0)
    base_per_sample = per_sample_flops_breakdown(layers, r_active=0, r_train=0).base
    
    # Extract LoRA-only totals by subtracting base FLOPs from each configuration
    lora_only_totals = []
    for c in configs:
        if c["kind"] == "joint":
            # Joint: total_flops - (base_per_sample * samples * epochs)
            base_total = base_per_sample * samples_per_epoch * c["epochs"]
            lora_only = c["total_flops"] - base_total
        else:
            # Sequential: sum over components of (component_total - base_for_that_component)
            lora_only = 0
            for comp in c["components"]:
                base_total = base_per_sample * samples_per_epoch * comp["epochs"]
                lora_only += comp["total_flops"] - base_total
        lora_only_totals.append(lora_only)
    
    lora_only_g = np.array(lora_only_totals, dtype=float) / 1e9  # show in GFLOPs for better scale
    colors = ["#F0B96A" if c["kind"] == "joint" else "#8F8FE3" for c in configs]

    order = np.argsort(lora_only_g)
    labels = [labels[i] for i in order]
    lora_only_g = lora_only_g[order]
    colors = [colors[i] for i in order]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, lora_only_g, color=colors, edgecolor="white")
    for i, v in enumerate(lora_only_g):
        ax.text(v, i, f" {v:.0f} GF", va="center", ha="left", fontsize=9, color="#222")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("LoRA-only training FLOPs (GFLOPs)", fontsize=13, fontweight="bold")
    ax.set_title("Total experiment FLOPs (base subtracted)", fontsize=14, fontweight="bold")
    ax.set_xlim(right=lora_only_g.max() * 1.18 if lora_only_g.size > 0 else 1)
    ax.grid(True, axis="x", linestyle="-", alpha=0.18)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def _maybe_profile(data: dict, mode: str, max_rank: int):
    """Run PyTorch's FlopCounterMode if requested.  Returns None if not.

    Supported ``mode`` values:
      * ``"none"``  — no profiling (default).
      * ``"vision"`` — use the vision MLP profiler.
      * ``"nlp"``    — use the DistilBERT profiler.
      * ``"auto"``   — pick based on the JSON's architecture.
    """
    if mode == "none":
        return None

    arch = data["flops"].get("architecture", "")
    if mode == "auto":
        mode = "nlp" if arch == "distilbert_qv" else "vision"

    print(f"[flops] profiling per-step FLOPs (mode={mode})...")
    if mode == "vision":
        from lora_lib.flops_profile import profile_per_rank_grid_vision

        first_layer = data["flops"]["layers"][0]
        last_layer = data["flops"]["layers"][-1]
        return profile_per_rank_grid_vision(
            input_size=first_layer[0],
            num_classes=last_layer[1],
            max_rank=max_rank,
        )
    if mode == "nlp":
        from lora_lib.flops_profile import profile_per_rank_grid_nlp

        max_length = data["flops"].get("max_length", 128)
        return profile_per_rank_grid_nlp(max_rank=max_rank, max_length=max_length)
    raise ValueError(f"Unknown --profile mode: {mode}")


def render(
    results_path: str,
    out_path: str,
    *,
    max_rank: int = 4,
    profile: str = "none",
    subtitle_extra: str = "",
) -> str:
    with open(results_path) as f:
        data = json.load(f)
    data = _infer_flops_metadata(data)

    layers = [tuple(pair) for pair in data["flops"]["layers"]]
    summary = summarise_results_flops(data, samples_per_epoch=data["flops"]["samples_per_epoch"])
    profiled = _maybe_profile(data, profile, max_rank)

    with xkcd_style(scale=0.9, length=80, randomness=2):
        fig, (ax_total, ax_lora, ax_exp, ax_exp_lora) = plt.subplots(
            1, 4, figsize=(26, 6.5), gridspec_kw={"width_ratios": [1.0, 1.0, 1.4, 1.4]}
        )
        _draw_per_step_total(ax_total, layers, max_rank=max_rank, profiled=profiled)
        _draw_per_step_lora_only(ax_lora, layers, max_rank=max_rank, profiled=profiled)
        _draw_total_panel(ax_exp, summary)
        _draw_total_lora_only_panel(ax_exp_lora, summary)

        dataset = data.get("dataset", "?")
        suptitle = f"FLOPs: joint LoRA vs sequential rank-1 LoRA — {dataset}"
        if profiled is not None:
            suptitle += "  (analytical bars + profiled tick marks)"
        if subtitle_extra:
            suptitle += f"  {subtitle_extra}"
        fig.suptitle(suptitle, fontsize=15, fontweight="bold")
        plt.tight_layout(rect=(0, 0.04, 1, 0.94))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path)
        plt.close(fig)
    return out_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", required=True, help="Path to a results/*.json file.")
    parser.add_argument("--output", required=True, help="Where to save the PNG figure.")
    parser.add_argument("--max-rank", type=int, default=4, help="Largest rank shown in the per-step panel.")
    parser.add_argument(
        "--profile",
        choices=["none", "vision", "nlp", "auto"],
        default="none",
        help="Also profile per-step FLOPs with torch.utils.flop_counter and overlay them.",
    )
    parser.add_argument(
        "--subtitle-extra",
        default="",
        help="Optional text appended to the figure suptitle (e.g. seed-run note).",
    )
    args = parser.parse_args(argv)
    out = render(
        args.results,
        args.output,
        max_rank=args.max_rank,
        profile=args.profile,
        subtitle_extra=args.subtitle_extra,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
