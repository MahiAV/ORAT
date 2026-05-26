"""Independent figure: per-sample LoRA-only FLOPs — joint LoRA vs sequential
rank-1 LoRA — extended to ``--max-rank`` ranks (default 6).

For each rank ``r`` the figure shows two stacked bars:

  - **joint**:  cost per sample of training a rank-``r`` LoRA jointly
                 (all ``r`` factors learnable together).
  - **seq**:    cost per sample of training the ``r``-th rank-1 component
                 with the previous ``r-1`` frozen.

Both bars are broken down into:

  - LoRA forward + input gradient  (light blue)
  - LoRA parameter gradient        (orange)

The frozen base is **always subtracted** so the joint vs sequential gap
is clearly visible.  The savings ``-X%`` between the two bars is shown
above each rank pair.  Optional ``--profile`` overlays the corresponding
PyTorch ``FlopCounterMode`` measurements as black tick marks.

Examples
--------
    python figure_lora_only_per_sample.py \\
        --results results/vision_cifar100_equiv_J12_E8-8-8.json \\
        --output figures/figure_lora_only_per_sample_cifar100.png \\
        --max-rank 6

    # Add profiled overlay (vision profile takes ~3s, NLP ~30s):
    python figure_lora_only_per_sample.py \\
        --results results/vision_cifar10_equiv_J18_E12-12-12.json \\
        --output figures/figure_lora_only_per_sample_cifar10.png \\
        --max-rank 6 --profile auto
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

from figure_flops_comparison import _infer_flops_metadata, _maybe_profile
from lora_lib.flops import per_sample_flops_breakdown
from lora_lib.plot_style import xkcd_style


_FWD_COLOR = "#7BAFD4"
_BWD_COLOR = "#F18F01"


def _gather_breakdowns(
    layers, max_rank: int
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Return per-rank ``(fwd+input_grad, param_grad)`` for joint and seq."""
    joint, seq = [], []
    for r in range(1, max_rank + 1):
        bd_j = per_sample_flops_breakdown(layers, r_active=r, r_train=r)
        bd_s = per_sample_flops_breakdown(layers, r_active=r, r_train=1)
        joint.append((bd_j.lora_forward + bd_j.lora_backward_input,
                      bd_j.lora_backward_params))
        seq.append((bd_s.lora_forward + bd_s.lora_backward_input,
                    bd_s.lora_backward_params))
    return joint, seq


def render(
    results_path: str,
    out_path: str,
    *,
    max_rank: int = 6,
    profile: str = "none",
) -> str:
    with open(results_path) as f:
        data = json.load(f)
    data = _infer_flops_metadata(data)

    layers = [tuple(pair) for pair in data["flops"]["layers"]]
    joint_bd, seq_bd = _gather_breakdowns(layers, max_rank)
    profiled = _maybe_profile(data, profile, max_rank)
    base_per_sample = (
        profiled["base_per_sample"]
        if profiled is not None
        else per_sample_flops_breakdown(layers, r_active=0, r_train=0).base
    )

    j_fwd = np.array([row[0] for row in joint_bd], dtype=float)
    j_bwd = np.array([row[1] for row in joint_bd], dtype=float)
    s_fwd = np.array([row[0] for row in seq_bd], dtype=float)
    s_bwd = np.array([row[1] for row in seq_bd], dtype=float)
    j_tot = j_fwd + j_bwd
    s_tot = s_fwd + s_bwd

    # Auto-pick unit so the bar labels are 1–3 digit numbers.
    max_total = float(max(j_tot.max(), s_tot.max())) if max_rank else 0.0
    if max_total >= 5e8:
        unit_scale, unit_label = 1e9, "GFLOPs"
    elif max_total >= 5e5:
        unit_scale, unit_label = 1e6, "MFLOPs"
    else:
        unit_scale, unit_label = 1e3, "KFLOPs"

    j_fwd_u = j_fwd / unit_scale
    j_bwd_u = j_bwd / unit_scale
    s_fwd_u = s_fwd / unit_scale
    s_bwd_u = s_bwd / unit_scale
    j_tot_u = j_tot / unit_scale
    s_tot_u = s_tot / unit_scale

    if profiled is not None:
        prof_j = np.array(
            [profiled["joint"][r] - base_per_sample for r in range(1, max_rank + 1)],
            dtype=float,
        ) / unit_scale
        prof_s = np.array(
            [profiled["sequential"][r] - base_per_sample for r in range(1, max_rank + 1)],
            dtype=float,
        ) / unit_scale
    else:
        prof_j = prof_s = None

    if max_total / unit_scale >= 100:
        fmt = "{:.0f}"
    elif max_total / unit_scale >= 10:
        fmt = "{:.1f}"
    else:
        fmt = "{:.2f}"

    dataset = data.get("dataset", "?").upper()
    arch = data["flops"].get("architecture", "")
    arch_pretty = "DistilBERT (Q,V)" if arch == "distilbert_qv" else "Vision MLP"

    with xkcd_style(scale=0.9, length=80, randomness=2):
        fig, ax = plt.subplots(figsize=(11.0, 6.0))

        x = np.arange(max_rank)
        bar_w = 0.40

        ax.bar(x - bar_w / 2, j_fwd_u, bar_w, color=_FWD_COLOR, edgecolor="white",
               linewidth=1.0, label="LoRA forward + input grad")
        ax.bar(x - bar_w / 2, j_bwd_u, bar_w, bottom=j_fwd_u, color=_BWD_COLOR,
               edgecolor="white", linewidth=1.0, label="LoRA parameter grad")
        ax.bar(x + bar_w / 2, s_fwd_u, bar_w, color=_FWD_COLOR, edgecolor="white",
               linewidth=1.0, hatch="//")
        ax.bar(x + bar_w / 2, s_bwd_u, bar_w, bottom=s_fwd_u, color=_BWD_COLOR,
               edgecolor="white", linewidth=1.0, hatch="//")

        # Profiled overlay (per pair)
        if prof_j is not None and prof_s is not None:
            for i in range(max_rank):
                for x_c, y_v in (
                    (x[i] - bar_w / 2, prof_j[i]),
                    (x[i] + bar_w / 2, prof_s[i]),
                ):
                    ax.hlines(y_v, x_c - bar_w * 0.45, x_c + bar_w * 0.45,
                              colors="black", linewidth=2.0, zorder=5)
                    ax.scatter([x_c], [y_v], color="black", s=22, zorder=6)
            ax.plot([], [], color="black", marker="o", linestyle="-",
                    linewidth=2.0, markersize=5, label="Profiled (PyTorch)")

        # Headroom large enough to fit both the bar number labels *and* the
        # savings %, with the legend tucked above the data area.
        bar_max = max(float(j_tot_u.max()), float(s_tot_u.max()))
        y_top = bar_max * 1.55
        if prof_j is not None and prof_j.size > 0:
            y_top = max(y_top, max(float(prof_j.max()), float(prof_s.max())) * 1.40)

        for i in range(max_rank):
            # numerical totals (top of each bar)
            ax.text(x[i] - bar_w / 2, j_tot_u[i] + bar_max * 0.018,
                    fmt.format(j_tot_u[i]), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#222")
            ax.text(x[i] + bar_w / 2, s_tot_u[i] + bar_max * 0.018,
                    fmt.format(s_tot_u[i]), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#222")

            # bar role labels (joint / seq) just above the value
            ax.text(x[i] - bar_w / 2, j_tot_u[i] + bar_max * 0.085,
                    "joint", ha="center", va="bottom", fontsize=8.5, color="#555")
            ax.text(x[i] + bar_w / 2, s_tot_u[i] + bar_max * 0.085,
                    "seq",   ha="center", va="bottom", fontsize=8.5, color="#555")

            # Savings %: above the *taller* bar (joint), centered between the pair
            if j_tot_u[i] > 0:
                savings = (j_tot_u[i] - s_tot_u[i]) / j_tot_u[i] * 100.0
                pair_top = max(j_tot_u[i], s_tot_u[i]) + bar_max * 0.18
                ax.text(x[i], pair_top, f"-{savings:.1f}%",
                        ha="center", va="bottom",
                        fontsize=10.5, color=_BWD_COLOR, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([rf"$r = {r}$" for r in range(1, max_rank + 1)],
                           fontsize=12)
        ax.set_xlabel("Rank", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"FLOPs per sample ({unit_label})",
                      fontsize=12, fontweight="bold")

        # No title to match the rebuttal layout.

        ax.set_ylim(top=y_top)
        ax.grid(True, axis="y", linestyle="-", alpha=0.18)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Build legend with extra entry for the hatched "sequential" pattern.
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        handles, labels = ax.get_legend_handles_labels()
        # Add hatch-only patch to indicate the sequential bars.
        seq_patch = Patch(facecolor="white", edgecolor="black", hatch="//",
                          label="Sequential (hatched)")
        handles = handles + [seq_patch]
        labels = labels + ["Sequential (hatched)"]

        # Bars grow with rank, so the upper-left of the axes is empty.
        # Savings % is now anchored to each pair (low for r=1) so it never
        # collides with a top-left legend.
        ax.legend(handles, labels,
                  loc="upper left", fontsize=9.5, frameon=True,
                  framealpha=0.95, edgecolor="#cccccc",
                  ncol=1, handletextpad=0.6, labelspacing=0.35)

        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
    return out_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", required=True, help="Path to results/*.json")
    parser.add_argument("--output", required=True, help="Where to save the PNG figure.")
    parser.add_argument("--max-rank", type=int, default=6,
                        help="Largest rank shown (default: 6).")
    parser.add_argument("--profile", choices=["none", "vision", "nlp", "auto"],
                        default="none",
                        help="Overlay PyTorch FlopCounterMode measurements.")
    args = parser.parse_args(argv)
    out = render(args.results, args.output,
                 max_rank=args.max_rank, profile=args.profile)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
