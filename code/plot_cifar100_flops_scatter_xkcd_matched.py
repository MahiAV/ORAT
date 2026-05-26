"""XKCD-style scatter + **smoothed** trend curves: LoRA-only FLOPs vs accuracy.

Uses ``efficient_ablation_cifar100_fixed`` (or ``--results-dir``) JSON outputs.

**Sequential r=1 / r=2 / r=3:** one point per *front-loaded* run (``front_*.json``,
``ablation_config.alpha > 0`` only).  Each point is cumulative LoRA-only FLOPs
after that stage vs ``component_accuracies`` at that stage.

**Joint LoRA:** points from ``joint_<epochs>.json`` (``standard_lora``).  By default
the **two** highest-FLOPs joint points are dropped (``--joint-tail-drop 2``).

**α filter:** only ``front_*`` runs with ``alpha_min ≤ α ≤ alpha_max`` (defaults
``0.2``–``2.0`` to match the usual sweep grid).

**Markers:** one fixed Matplotlib ``s`` (area in pt²) for every scatter point and
matching legend glyphs (``--marker-size``, default 165).

Curves use a **smoothing spline** (``UnivariateSpline``), not an interpolating
PCHIP through every scatter — visually closer to a soft “fit” than
connect-the-dots.

Visual layout mirrors the 1200×800 reference (axes box, legend, colors).

Examples
--------
    python plot_cifar100_flops_scatter_xkcd_matched.py \\
        --results-dir ../efficient_ablation_cifar100_fixed \\
        --alpha-min 0.4 --alpha-max 1.8 \\
        -o figures/cifar100_flops_scatter_xkcd.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import UnivariateSpline

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lora_lib.flops import per_sample_flops_breakdown

_JOINT_FILE = re.compile(r"^joint_(\d+)\.json$")


COLORS = {
    "r1_marker": "#75C179",
    "r1_line": "#A3D6A5",
    "r2_marker": "#E76772",
    "r2_line": "#EF99A1",
    "r3_marker": "#B3A6DE",
    "r3_line": "#CCC3E9",
    "lora": "#F0A22E",
    "legend_edge": "#C3C3C3",
}


def _layers_and_spe(data: dict) -> Tuple[List[Tuple[int, int]], int]:
    fm = data["flops"]
    layers = [tuple(pair) for pair in fm["layers"]]
    return layers, int(fm["samples_per_epoch"])


def _cumulative_lora_flops_gf(
    layers: List[Tuple[int, int]],
    samples_per_epoch: int,
    schedule: List[int],
) -> List[float]:
    """Cumulative LoRA-only GFLOPs after each sequential stage (1..r)."""
    base = per_sample_flops_breakdown(layers, r_active=0, r_train=0).base
    cum_gf: List[float] = []
    total = 0.0
    for k, ep in enumerate(schedule, start=1):
        tot_ps = per_sample_flops_breakdown(layers, r_active=k, r_train=1).total
        lora_ps = tot_ps - base
        total += lora_ps * samples_per_epoch * ep
        cum_gf.append(total / 1e9)
    return cum_gf


def _joint_lora_flops_gf(
    layers: List[Tuple[int, int]],
    samples_per_epoch: int,
    epochs: int,
    rank: int = 3,
) -> float:
    base = per_sample_flops_breakdown(layers, r_active=0, r_train=0).base
    tot_ps = per_sample_flops_breakdown(layers, r_active=rank, r_train=rank).total
    lora_ps = tot_ps - base
    return float(lora_ps * samples_per_epoch * epochs / 1e9)


def collect_points(
    results_dir: Path,
    *,
    alpha_min: float,
    alpha_max: float,
    marker_size: float,
) -> Tuple[
    List[float], List[float], List[float],
    List[float], List[float], List[float],
    List[float], List[float], List[float],
    List[float], List[float], List[float],
    List[float], List[float],
]:
    """Returns (x_r1,y_r1,s_r1), (x_r2,y_r2,s_r2), (x_r3,y_r3,s_r3), (x_l,y_l,s_l)
    with y as **percent** (0–100) for plotting to match reference style."""
    x1, y1, s1 = [], [], []
    x2, y2, s2 = [], [], []
    x3, y3, s3 = [], [], []
    xl, yl, sl = [], [], []

    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json"):
            continue
        path = results_dir / name

        if name.startswith("front_"):
            with open(path) as f:
                data = json.load(f)
            cfg = data.get("ablation_config", {})
            if cfg.get("type") != "sequential":
                continue
            alpha = float(cfg.get("alpha", 0.0))
            if alpha <= 0:
                continue
            if alpha < alpha_min or alpha > alpha_max:
                continue
            seq = data.get("sequential_paths") or {}
            if not seq:
                continue
            run = next(iter(seq.values()))
            sched = run["epoch_allocation"]
            accs = run["component_accuracies"]
            layers, spe = _layers_and_spe(data)
            flops_cum = _cumulative_lora_flops_gf(layers, spe, sched)
            for stage in range(3):
                gf = flops_cum[stage]
                pct = accs[stage] * 100.0
                if stage == 0:
                    x1.append(gf)
                    y1.append(pct)
                    s1.append(marker_size)
                elif stage == 1:
                    x2.append(gf)
                    y2.append(pct)
                    s2.append(marker_size)
                else:
                    x3.append(gf)
                    y3.append(pct)
                    s3.append(marker_size)
            continue

        m = _JOINT_FILE.match(name)
        if m:
            with open(path) as f:
                data = json.load(f)
            slora = data.get("standard_lora") or {}
            ep = int(m.group(1))
            acc = slora.get("final_accuracy")
            if acc is None:
                continue
            layers, spe = _layers_and_spe(data)
            gf = _joint_lora_flops_gf(layers, spe, ep, rank=int(slora.get("rank", 3)))
            xl.append(gf)
            yl.append(float(acc) * 100.0)
            sl.append(marker_size)

    return x1, y1, s1, x2, y2, s2, x3, y3, s3, xl, yl, sl


def _aggregate_xy(xs: List[float], ys: List[float]) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """Sort by x; average y at duplicate x (rounded)."""
    if len(xs) < 2:
        return None, None
    bucket: Dict[float, List[float]] = defaultdict(list)
    for x, y in zip(xs, ys):
        key = round(float(x), 6)
        bucket[key].append(float(y))
    xu = np.array(sorted(bucket), dtype=float)
    yu = np.array([float(np.mean(bucket[k])) for k in xu], dtype=float)
    if xu.size < 2:
        return None, None
    order = np.argsort(xu)
    return xu[order], yu[order]


def _smooth_fit(
    xs: List[float],
    ys: List[float],
    *,
    smoothing: float,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """Smoothing spline: does *not* interpolate every point when smoothing > 0."""
    xu, yu = _aggregate_xy(xs, ys)
    if xu is None:
        return None, None
    n = int(xu.size)
    if n < 2:
        return None, None

    y_var = float(np.var(yu)) if n > 1 else 0.0
    y_span = float(np.max(yu) - np.min(yu)) if n > 1 else 1.0
    # ``s`` = target residual sum of squares; larger => smoother / less wiggly.
    s = max(n * 1e-8, smoothing * n * max(y_var, (0.02 * y_span) ** 2))

    k = min(3, n - 1)
    k = max(1, k)
    try:
        spl = UnivariateSpline(xu, yu, k=k, s=s)
    except (ValueError, TypeError):
        spl = UnivariateSpline(xu, yu, k=1, s=s * 10.0)

    xf = np.linspace(float(xu.min()), float(xu.max()), 500)
    yf = np.asarray(spl(xf), dtype=float)
    yf = np.clip(yf, 0.0, 100.0)
    return xf, yf


def _drop_top_joint_by_flops(
    xl: List[float],
    yl: List[float],
    sl: List[float],
    n_drop: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Remove the ``n_drop`` joint LoRA points with largest LoRA-only GFLOPs.

    Keeps at least one joint point when any exist: ``n_drop`` is capped at
    ``len(xl) - 1``.
    """
    if n_drop <= 0 or not xl:
        return xl, yl, sl
    n = min(n_drop, len(xl) - 1)
    if n <= 0:
        return xl, yl, sl
    xa = np.asarray(xl, dtype=float)
    drop_idx = set(int(i) for i in np.argsort(-xa)[:n])
    return (
        [v for i, v in enumerate(xl) if i not in drop_idx],
        [v for i, v in enumerate(yl) if i not in drop_idx],
        [v for i, v in enumerate(sl) if i not in drop_idx],
    )


def plot_figure(
    results_dir: Path,
    out_path: Path,
    *,
    xlabel: str = "Adaptation training FLOPs (GFLOPs)",
    ylabel: str = "Test Accuracy",
    smoothing: float = 4.0,
    alpha_min: float = 0.2,
    alpha_max: float = 2.0,
    joint_tail_drop: int = 2,
    marker_size: float = 165.0,
) -> None:
    x1, y1, s1, x2, y2, s2, x3, y3, s3, xl, yl, sl = collect_points(
        results_dir,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        marker_size=marker_size,
    )
    if joint_tail_drop > 0:
        xl, yl, sl = _drop_top_joint_by_flops(xl, yl, sl, joint_tail_drop)

    with plt.xkcd(scale=1.0, length=100, randomness=1.15):
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "mathtext.fontset": "dejavusans",
                "axes.linewidth": 1.75,
            }
        )

        fig = plt.figure(figsize=(12, 8), dpi=100, facecolor="white")
        ax = fig.add_axes([205 / 1200, (800 - 665) / 800, (1180 - 205) / 1200, (665 - 21) / 800])
        ax.set_facecolor("white")

        for xs, ys, color in [
            (x1, y1, COLORS["r1_line"]),
            (x2, y2, COLORS["r2_line"]),
            (x3, y3, COLORS["r3_line"]),
            (xl, yl, COLORS["lora"]),
        ]:
            xfine, yfine = _smooth_fit(xs, ys, smoothing=smoothing)
            if xfine is None:
                continue
            ax.plot(
                xfine,
                yfine,
                color=color,
                lw=5.15,
                solid_capstyle="round",
                zorder=2,
            )

        ax.scatter(x1, y1, s=s1, marker="s", color=COLORS["r1_marker"], edgecolors="none", zorder=3)
        ax.scatter(x2, y2, s=s2, marker="^", color=COLORS["r2_marker"], edgecolors="none", zorder=3)
        ax.scatter(x3, y3, s=s3, marker="d", color=COLORS["r3_marker"], edgecolors="none", zorder=3)
        ax.scatter(xl, yl, s=sl, marker="o", color=COLORS["lora"], edgecolors="none", zorder=4)

        all_y = list(y1) + list(y2) + list(y3) + list(yl)
        all_x = list(x1) + list(x2) + list(x3) + list(xl)
        if not all_y:
            raise SystemExit(f"No front-loaded + joint points found under {results_dir}")

        y_lo = max(0.0, min(all_y) - 3.0)
        y_hi = min(100.0, max(all_y) + 3.0)
        x_lo = max(0.0, min(all_x) * 0.85)
        x_hi = max(all_x) * 1.08

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
        ax.tick_params(axis="both", labelsize=22, width=2.2, length=8, pad=2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.75)
        ax.spines["bottom"].set_linewidth(1.75)
        ax.set_xlabel("")
        ax.set_ylabel("")

        fig.text(
            691.5 / 1200,
            (800 - 758) / 800,
            xlabel,
            ha="center",
            va="center",
            fontsize=30,
            fontweight="bold",
        )
        fig.text(
            42.5 / 1200,
            (800 - 354) / 800,
            ylabel,
            ha="center",
            va="center",
            rotation=90,
            fontsize=32,
            fontweight="bold",
        )

        leg_x = (734 - 205) / (1180 - 205)
        leg_y = (665 - 646) / (665 - 21)
        leg_w = (1159 - 734) / (1180 - 205)
        leg_h = (646 - 326) / (665 - 21)
        legend_box = FancyBboxPatch(
            (leg_x, leg_y),
            leg_w,
            leg_h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=COLORS["legend_edge"],
            linewidth=1.15,
            zorder=1.5,
        )
        ax.add_patch(legend_box)

        ax.text(
            (751 - 205) / (1180 - 205),
            (665 - 361) / (665 - 21),
            "Model Architecture",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=26,
            fontweight="bold",
            zorder=5,
        )

        marker_x = (809.0 - 205) / (1180 - 205)
        text_x = (884.0 - 205) / (1180 - 205)
        ms = marker_size
        marker_entries = [
            ((665 - 420.1) / (665 - 21), "s", ms, COLORS["r1_marker"]),
            ((665 - 486.0) / (665 - 21), "^", ms, COLORS["r2_marker"]),
            ((665 - 544.6) / (665 - 21), "d", ms, COLORS["r3_marker"]),
            ((665 - 609.0) / (665 - 21), "o", ms, COLORS["lora"]),
        ]
        for y, marker, size, color in marker_entries:
            ax.scatter(
                [marker_x],
                [y],
                transform=ax.transAxes,
                clip_on=False,
                s=size,
                marker=marker,
                color=color,
                edgecolors="none",
                zorder=5,
            )

        text_entries = [
            ((665 - 419.5) / (665 - 21), r"$r = 1$"),
            ((665 - 482.0) / (665 - 21), r"$r = 2$"),
            ((665 - 544.5) / (665 - 21), r"$r = 3$"),
            ((665 - 607.5) / (665 - 21), r"$\mathbf{LoRA}$ - $r = 3$"),
        ]
        for y, label in text_entries:
            ax.text(text_x, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=30, zorder=5)

        os.makedirs(out_path.parent or Path("."), exist_ok=True)
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("efficient_ablation_cifar100_fixed"),
        help="Directory with joint_*.json and front_*.json ablation outputs.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("figures/cifar100_flops_scatter_xkcd_front_only.png"),
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=10.0,
        help="Spline smoothness (larger = smoother trend, less hugging of each point). Default: 4.0",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.2,
        metavar="A",
        help="Include only front-loaded runs with α ≥ this (inclusive). Default: 0.2",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=2.0,
        metavar="A",
        help="Include only front-loaded runs with α ≤ this (inclusive). Default: 2.0",
    )
    parser.add_argument(
        "--joint-tail-drop",
        type=int,
        default=2,
        help="Remove this many highest-FLOPs joint LoRA points (default: 2 = last + penultimate).",
    )
    parser.add_argument(
        "--no-drop-joint-tail",
        action="store_true",
        help="Keep all joint LoRA points (ignore --joint-tail-drop).",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=165.0,
        metavar="S",
        help="Matplotlib scatter marker area (``s``) for every series—same for "
             "r=1/2/3 and joint LoRA. Default: 165.",
    )
    args = parser.parse_args()
    if args.alpha_min > args.alpha_max:
        raise SystemExit("--alpha-min must be ≤ --alpha-max")
    if not args.results_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.results_dir}")
    jdrop = 0 if args.no_drop_joint_tail else max(0, args.joint_tail_drop)
    plot_figure(
        args.results_dir,
        args.out,
        smoothing=args.smoothing,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        joint_tail_drop=jdrop,
        marker_size=args.marker_size,
    )
    print(
        f"Wrote {args.out} (front α∈[{args.alpha_min:g},{args.alpha_max:g}]; "
        f"joint_tail_drop={jdrop}; smoothing={args.smoothing}; marker_size={args.marker_size})"
    )


if __name__ == "__main__":
    main()
