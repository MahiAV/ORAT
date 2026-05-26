"""Figure 2 — bar plot of test accuracy on the *new* classes for each
sequential rank (Rank-1, Rank-2, Rank-3) plus jointly trained LoRA, for
MNIST / CIFAR-10 / CIFAR-100.

For each rank ``k``, the bar height is the best accuracy across all
sequential paths after their ``k``-th component finished training.  This
mirrors the "best of stage" view shown in the paper.

Usage
-----
    python figure2_vision_barplots.py \
        --mnist results/vision_mnist.json \
        --cifar10 results/vision_cifar10.json \
        --cifar100 results/vision_cifar100.json \
        --output figures/xkcd_lora_plot.png

Any of ``--mnist`` / ``--cifar10`` / ``--cifar100`` may be omitted; the
corresponding panel is then skipped (handy for partial runs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from lora_lib.plot_style import percent_formatter, xkcd_style


_BAR_COLORS = ["#62A4D1", "#D66A6A", "#8F8FE3", "#F0B96A"]
# Mathtext is used to keep "LoRA" bold while the rest stays normal weight.
_BAR_LABELS = [
    r"$r=1$",
    r"$r=2$",
    r"$r=3$",
    r"$\mathbf{LoRA}$ - $r=3$",
]


def best_of_each_stage(results: dict) -> List[float]:
    """Return [best Rank-1, best Rank-2, best Rank-3, Standard LoRA] accuracies."""
    paths = results["sequential_paths"]
    by_rank: Dict[int, List[float]] = {1: [], 2: [], 3: []}
    for run in paths.values():
        for k, acc in enumerate(run["component_accuracies"], start=1):
            if k in by_rank:
                by_rank[k].append(acc)

    best = [max(by_rank[k]) if by_rank[k] else float("nan") for k in (1, 2, 3)]
    best.append(results["standard_lora"]["test_acc"])
    return best


def best_of_each_stage_stats(paths: List[str]) -> tuple[List[float], List[float]]:
    """Mean and sample standard deviation across JSON runs (same sweep, different seeds)."""
    rows: List[List[float]] = []
    for p in paths:
        with open(p) as f:
            rows.append(best_of_each_stage(json.load(f)))
    arr = np.asarray(rows, dtype=float)
    mean = np.nanmean(arr, axis=0).tolist()
    if arr.shape[0] < 2:
        return mean, [0.0] * len(mean)
    std = np.nanstd(arr, axis=0, ddof=1).tolist()
    std = [0.0 if (isinstance(s, float) and np.isnan(s)) else s for s in std]
    return mean, std


def _ylimits(values: List[float]) -> tuple[float, float]:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return 0.0, 1.0
    lo = max(0.0, min(finite) - 0.05)
    hi = min(1.0, max(finite) + 0.05)
    if hi - lo < 0.1:  # avoid an over-zoomed plot
        mid = 0.5 * (lo + hi)
        lo, hi = max(0.0, mid - 0.05), min(1.0, mid + 0.05)
    return lo, hi


def render_panel(
    ax,
    results: dict | None,
    title: str,
    *,
    multi_paths: Optional[List[str]] = None,
) -> None:
    if multi_paths is not None and len(multi_paths) > 0:
        accuracies, yerr = best_of_each_stage_stats(multi_paths)
        n_seeds = len(multi_paths)
    else:
        assert results is not None
        accuracies = best_of_each_stage(results)
        yerr = None
        n_seeds = 1

    bars = ax.bar(
        _BAR_LABELS,
        accuracies,
        yerr=yerr if yerr is not None and any(e > 0 for e in yerr) else None,
        capsize=5 if yerr is not None else 0,
        color=_BAR_COLORS,
        edgecolor="black",
        linewidth=1.4,
        error_kw={"elinewidth": 1.6, "capthick": 1.6, "ecolor": "#333333"},
    )

    lo, hi = _ylimits(accuracies)
    if yerr is not None:
        hi = min(1.0, hi + (max(yerr) if yerr else 0) * 1.2)
        lo = max(0.0, lo - (max(yerr) if yerr else 0) * 0.3)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Test Accuracy", fontsize=14)
    ax.yaxis.set_major_formatter(percent_formatter(decimals=1))
    ax.grid(axis="y", linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    for bar, acc, err in zip(
        bars,
        accuracies,
        yerr if yerr is not None else [0.0] * len(accuracies),
    ):
        if not np.isfinite(acc):
            continue
        if n_seeds > 1 and err > 0:
            label = f"{acc:.3f}\n±{err:.3f}"
        else:
            label = f"{acc:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            acc + (hi - lo) * 0.015 + (err if err else 0),
            label,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    for tick in ax.get_xticklabels():
        tick.set_rotation(15)


def render(
    results_paths: Dict[str, Optional[str]],
    out_path: str,
    *,
    multi_paths: Optional[Dict[str, Optional[List[str]]]] = None,
) -> str:
    """``results_paths`` holds single-file paths; ``multi_paths`` (if set)
    holds lists of JSON paths per dataset for mean±SD bars."""
    panels: List[Tuple[str, Optional[str], Optional[List[str]]]] = []
    if multi_paths:
        for name in ("mnist", "cifar10", "cifar100"):
            mp = multi_paths.get(name)
            if mp:
                panels.append((name, None, mp))
    else:
        for name, path in results_paths.items():
            if path:
                panels.append((name, path, None))

    if not panels:
        raise SystemExit(
            "Nothing to plot — provide --mnist/--cifar10/--cifar100 "
            "or --*-multi with JSON paths."
        )

    with xkcd_style(scale=0.9, length=80, randomness=2):
        fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6))
        if len(panels) == 1:
            axes = [axes]
        roman = ["(I)", "(II)", "(III)", "(IV)"]
        for idx, (ax, (name, path, mpaths)) in enumerate(zip(axes, panels)):
            dataset_name = {
                "mnist": "MNIST",
                "cifar10": "CIFAR-10",
                "cifar100": "CIFAR-100",
            }.get(name, name.upper())
            if mpaths:
                render_panel(ax, None, dataset_name, multi_paths=mpaths)
            else:
                with open(path) as f:  # type: ignore[arg-type]
                    data = json.load(f)
                render_panel(ax, data, dataset_name)
            ax.set_xlabel(f"{roman[idx]} {dataset_name}", fontsize=18, fontweight="bold", labelpad=15)
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path)
        plt.close(fig)
    return out_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mnist", default="results/vision_mnist.json", help="MNIST results JSON (or omit)")
    parser.add_argument("--cifar10", default="results/vision_cifar10.json", help="CIFAR-10 results JSON (or omit)")
    parser.add_argument("--cifar100", default="results/vision_cifar100.json", help="CIFAR-100 results JSON (or omit)")
    parser.add_argument(
        "--mnist-multi",
        nargs="+",
        default=None,
        metavar="JSON",
        help="Several MNIST JSONs (e.g. different --seed); bar heights = mean, error = sample SD.",
    )
    parser.add_argument("--cifar10-multi", nargs="+", default=None, metavar="JSON")
    parser.add_argument("--cifar100-multi", nargs="+", default=None, metavar="JSON")
    parser.add_argument("--output", default="figures/figure2_vision_barplots.png")
    args = parser.parse_args(argv)

    multi = {
        "mnist": [p for p in (args.mnist_multi or []) if os.path.exists(p)] or None,
        "cifar10": [p for p in (args.cifar10_multi or []) if os.path.exists(p)] or None,
        "cifar100": [p for p in (args.cifar100_multi or []) if os.path.exists(p)] or None,
    }
    if any(multi.values()):
        out = render({}, args.output, multi_paths=multi)
    else:
        paths = {
            "mnist": args.mnist if os.path.exists(args.mnist) else None,
            "cifar10": args.cifar10 if os.path.exists(args.cifar10) else None,
            "cifar100": args.cifar100 if os.path.exists(args.cifar100) else None,
        }
        out = render(paths, args.output)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
