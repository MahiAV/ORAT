"""Final, publication-ready accuracy-vs-Adaptation-FLOPs plot for schedule ablations.

Differences vs ``plot_schedule_ablation_filtered.py``:

1. **Top-k alpha selection**. We rank every front-loaded (positive alpha)
   schedule by win-rate against Joint LoRA (linearly interpolated at matched
   FLOPs); ties broken by mean accuracy advantage. The top ``--top-k`` alphas
   are kept along with their back-loaded counterparts. ``Equal Schedule`` is
   shown as a green baseline.
2. **X-axis truncation**. The x-axis is cut at the *second-to-last* Joint LoRA
   point (i.e. the very last Joint LoRA point and any same-or-larger-FLOPs
   points from other schedules are removed).
3. **Compact aspect**. Default figsize is 8 x 4.8 (less wide than the
   exploratory views).
4. **Legend inside the figure**, lower-right by default.
5. **Schedule-type colors** (matching ``reproduce_fig_combined.py``):
   ``Joint LoRA`` is black, ``Equal Schedule`` is green, all
   ``More first`` schedules are shades of orange, all ``Less first``
   schedules are shades of blue. Larger ``|α|`` => darker shade.
6. **Plain-language legend**: ``Joint LoRA``, ``Equal schedule (α=0.0)``,
   ``More first (α=X)``, ``Less first (α=−X)``.

Usage:
    python plot_schedule_ablation_final.py \\
        --results-dir efficient_ablation_cifar100_fixed \\
        --output figures_efficient_ablation/accuracy_vs_flops_cifar100_final.png \\
        --title "CIFAR-100" \\
        --top-k 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_schedule_ablation import (
    _parse_alpha,
    _set_lightness,
    extract_accuracy_flops_points,
    load_ablation_results,
)


# Base palette borrowed from ``reproduce_fig_combined.py`` so that the two sets
# of figures look like they belong to the same paper.
JOINT_COLOR = "#000000"           # Joint LoRA -- black, never confused with a schedule.
EQUAL_COLOR = "#2ca02c"           # Equal schedule -- green (matches "uniform").
MORE_FIRST_BASE = "#ff7f0e"       # Front-loaded -- orange (matches "early_heavy").
LESS_FIRST_BASE = "#1f77b4"       # Back-loaded  -- blue   (matches "late_heavy").

# Lightness range used to spread several alpha values within a single base hue.
# Larger ``|α|`` => darker shade (more visually emphasized).
DARK_L = 0.34
LIGHT_L = 0.66


def _shade_palette(base_color: str, n: int) -> List[Tuple[float, float, float]]:
    """Return ``n`` shades of ``base_color`` from light to dark (LIGHT_L -> DARK_L).

    With ``n == 1`` we just return the base color so the single-alpha case stays
    visually identical to ``reproduce_fig_combined.py``.
    """
    if n <= 0:
        return []
    if n == 1:
        return [tuple(plt.matplotlib.colors.to_rgb(base_color))]  # type: ignore[return-value]
    levels = np.linspace(LIGHT_L, DARK_L, n)
    return [_set_lightness(base_color, float(l)) for l in levels]


def _joint_curve(joint_points: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not joint_points:
        return np.array([]), np.array([])
    flops, acc = zip(*sorted(joint_points))
    return np.array(flops, dtype=float), np.array(acc, dtype=float)


def _alpha_score(
    points: List[Tuple[float, float]],
    joint_flops: np.ndarray,
    joint_acc: np.ndarray,
) -> Tuple[float, float, int]:
    """Return ``(win_rate, mean_advantage, n_in_range)`` against the Joint LoRA curve.

    Points outside Joint LoRA's FLOPs range are excluded from both
    numerator and denominator -- we cannot fairly compare them.
    """
    if len(joint_flops) < 2 or not points:
        return 0.0, 0.0, 0

    f_min, f_max = float(joint_flops[0]), float(joint_flops[-1])
    advantages: List[float] = []
    wins = 0
    for f, a in points:
        if f < f_min or f > f_max:
            continue
        joint_a = float(np.interp(f, joint_flops, joint_acc))
        adv = a - joint_a
        advantages.append(adv)
        if adv > 0:
            wins += 1

    n = len(advantages)
    if n == 0:
        return 0.0, 0.0, 0
    return wins / n, float(np.mean(advantages)), n


def select_top_alphas(
    grouped_points: Dict[str, List[Tuple[float, float]]],
    top_k: int,
    verbose: bool = True,
) -> List[float]:
    """Pick the top-k positive alphas by (win-rate desc, mean-advantage desc)."""
    joint = grouped_points.get("Joint LoRA", [])
    jf, ja = _joint_curve(joint)

    scored: List[Tuple[float, float, int, float]] = []  # (winrate, mean_adv, n, alpha)
    for name, points in grouped_points.items():
        if not name.startswith("Front-loaded"):
            continue
        a = _parse_alpha(name)
        if a is None or a <= 0:
            continue
        winrate, mean_adv, n = _alpha_score(points, jf, ja)
        if n == 0:
            continue
        scored.append((winrate, mean_adv, n, a))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    if verbose:
        if jf.size:
            print(f"\n[final] Joint LoRA reference: {len(joint)} points "
                  f"({jf[0]:.2f} – {jf[-1]:.2f} GF)")
        else:
            print("[final] No Joint LoRA reference")
        print(f"\n  {'rank':<5} {'alpha':<7} {'winrate':>8} {'meanΔacc':>10} {'n':>4}")
        print("  " + "-" * 40)
        for i, (wr, ma, n, a) in enumerate(scored):
            mark = " *" if i < top_k else ""
            print(f"  {i+1:<5} {a:<7.1f} {wr:>8.2%} {ma:>+10.4f} {n:>4d}{mark}")

    return [a for _, _, _, a in scored[:top_k]]


def select_groups(
    grouped_points: Dict[str, List[Tuple[float, float]]],
    top_alphas: List[float],
) -> Dict[str, List[Tuple[float, float]]]:
    """Build the final group dict: Joint + Equal + Front/Back for each top alpha."""
    out: Dict[str, List[Tuple[float, float]]] = {}
    if grouped_points.get("Joint LoRA"):
        out["Joint LoRA"] = list(grouped_points["Joint LoRA"])
    if grouped_points.get("Equal Schedule"):
        out["Equal Schedule"] = list(grouped_points["Equal Schedule"])
    for a in top_alphas:
        front_name = f"Front-loaded (α={a:.1f})"
        back_name = f"Back-loaded (α={a:.1f})"
        if grouped_points.get(front_name):
            out[front_name] = list(grouped_points[front_name])
        if grouped_points.get(back_name):
            out[back_name] = list(grouped_points[back_name])
    return out


def truncate_x(
    grouped_points: Dict[str, List[Tuple[float, float]]],
    cutoff: float,
) -> Dict[str, List[Tuple[float, float]]]:
    return {
        name: [(f, a) for (f, a) in pts if f <= cutoff]
        for name, pts in grouped_points.items()
    }


def render(
    grouped_points: Dict[str, List[Tuple[float, float]]],
    top_alphas: List[float],
    title: str,
    output_path: str,
    figsize: Tuple[float, float] = (8.0, 4.8),
    compact_legend: bool = False,
) -> None:
    """Render the final figure.

    With ``compact_legend=True`` the legend is split into a baselines block
    (Joint LoRA + Equal) and a separate two-row ``More first (\u03b1)`` block
    that lists only the \u03b1 values, so we never repeat the words ``More first``.
    """
    # Sort selected alphas ascending so we can paint by ``|α|``.
    sorted_alphas = sorted(top_alphas)
    n_alphas = len(sorted_alphas)
    orange_shades = _shade_palette(MORE_FIRST_BASE, n_alphas)
    blue_shades = _shade_palette(LESS_FIRST_BASE, n_alphas)

    colors: Dict[str, Tuple[float, float, float] | str] = {
        "Joint LoRA": JOINT_COLOR,
        "Equal Schedule": EQUAL_COLOR,
    }
    markers: Dict[str, str] = {
        "Joint LoRA": "o",
        "Equal Schedule": "s",
    }
    if compact_legend:
        labels: Dict[str, str] = {
            "Joint LoRA": "Joint LoRA",
            "Equal Schedule": r"Equal ($\alpha = 0$)",
        }
    else:
        labels = {
            "Joint LoRA": "Joint LoRA",
            "Equal Schedule": r"Equal schedule ($\alpha = 0$)",
        }

    for i, alpha in enumerate(sorted_alphas):
        front = f"Front-loaded (α={alpha:.1f})"
        back = f"Back-loaded (α={alpha:.1f})"
        colors[front] = orange_shades[i]
        colors[back] = blue_shades[i]
        markers[front] = "^"
        markers[back] = "v"
        if compact_legend:
            labels[front] = rf"$\alpha = {alpha:g}$"
            labels[back] = rf"$\alpha = -{alpha:g}$"
        else:
            labels[front] = rf"More first ($\alpha = {alpha:g}$)"
            labels[back] = rf"Less first ($\alpha = -{alpha:g}$)"

    # Plot order: Joint LoRA first (lowest zorder among baselines), then Equal,
    # then alpha pairs (front above back, ascending |alpha|).
    order: List[str] = ["Joint LoRA", "Equal Schedule"]
    for a in sorted_alphas:
        order.append(f"Front-loaded (α={a:.1f})")
        order.append(f"Back-loaded (α={a:.1f})")

    # Axis labels / ticks: 1.5× the original sizes for readability in print.
    _axis_label_fs = 12.5 * 1.5   # 18.75
    _tick_fs = 10.5 * 1.5         # 15.75
    _title_fs = 13 * 1.5          # 19.5
    _legend_fs = 9.5 * 1.5        # ~1.5× default legend text

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": _axis_label_fs,
        "axes.titlesize": _title_fs,
        "xtick.labelsize": _tick_fs,
        "ytick.labelsize": _tick_fs,
        "legend.fontsize": _legend_fs,
        "lines.linewidth": 2.0,
        "axes.linewidth": 0.8,
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=figsize)

    for name in order:
        if name not in grouped_points:
            continue
        pts = grouped_points[name]
        if not pts:
            continue
        flops, acc = zip(*sorted(pts))
        color = colors[name]
        marker = markers[name]
        label = labels[name]
        is_baseline = name in ("Joint LoRA", "Equal Schedule")
        is_joint = name == "Joint LoRA"

        ax.plot(
            flops, acc,
            color=color,
            linewidth=2.4 if is_baseline else 1.8,
            alpha=0.95 if is_baseline else 0.90,
            linestyle="-",
            zorder=3 if is_baseline else 2,
        )
        ax.scatter(
            flops, acc,
            color=color,
            marker=marker,
            s=80 if is_baseline else 55,
            edgecolor="white",
            linewidth=0.9,
            label=label,
            zorder=5 if is_joint else (4 if is_baseline else 3),
        )

    ax.set_xlabel(
        "Adaptation Training FLOPs (GFLOPs)",
        fontsize=_axis_label_fs,
        fontweight="bold",
    )
    ax.set_ylabel("Test Accuracy", fontsize=_axis_label_fs, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=_title_fs, fontweight="bold")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(colors="#444444", axis="both", labelsize=_tick_fs)

    if compact_legend:
        # Two compact legends: baseline curves at upper-left, "More first"
        # alphas in a two-row grid at lower-right with a single header.
        all_handles, all_labels = ax.get_legend_handles_labels()
        baseline_keys = {"Joint LoRA", labels["Joint LoRA"], labels.get("Equal Schedule", "")}
        baseline_set = {labels["Joint LoRA"]}
        if "Equal Schedule" in labels:
            baseline_set.add(labels["Equal Schedule"])
        front_keys = {labels[f"Front-loaded (α={a:.1f})"] for a in sorted_alphas
                      if f"Front-loaded (α={a:.1f})" in labels}
        back_keys = {labels[f"Back-loaded (α={a:.1f})"] for a in sorted_alphas
                     if f"Back-loaded (α={a:.1f})" in labels}

        baseline_handles, baseline_labels_list = [], []
        front_handles, front_labels_list = [], []
        back_handles, back_labels_list = [], []
        for h, l in zip(all_handles, all_labels):
            if l in baseline_set:
                baseline_handles.append(h); baseline_labels_list.append(l)
            elif l in front_keys:
                front_handles.append(h); front_labels_list.append(l)
            elif l in back_keys:
                back_handles.append(h); back_labels_list.append(l)

        compact_fs = _legend_fs * 0.78
        leg_baseline = ax.legend(
            baseline_handles, baseline_labels_list,
            loc="upper left",
            fontsize=compact_fs,
            markerscale=1.2,
            labelspacing=0.35,
            handletextpad=0.5,
            borderpad=0.45,
            frameon=True,
            framealpha=0.96,
            edgecolor="#cccccc",
            fancybox=False,
            borderaxespad=0.45,
        )
        leg_baseline.get_frame().set_linewidth(0.6)
        ax.add_artist(leg_baseline)

        if front_handles:
            front_ncol = max(1, (len(front_handles) + 1) // 2)
            leg_front = ax.legend(
                front_handles, front_labels_list,
                loc="lower right",
                title=r"More first ($\alpha$)",
                title_fontsize=compact_fs,
                fontsize=compact_fs,
                markerscale=1.2,
                labelspacing=0.35,
                handletextpad=0.4,
                columnspacing=0.9,
                borderpad=0.45,
                frameon=True,
                framealpha=0.96,
                edgecolor="#cccccc",
                fancybox=False,
                borderaxespad=0.45,
                ncol=front_ncol,
            )
            leg_front.get_frame().set_linewidth(0.6)
            if back_handles:
                ax.add_artist(leg_front)

        if back_handles:
            back_ncol = max(1, (len(back_handles) + 1) // 2)
            leg_back = ax.legend(
                back_handles, back_labels_list,
                loc="lower left",
                title=r"Less first ($\alpha$)",
                title_fontsize=compact_fs,
                fontsize=compact_fs,
                markerscale=1.2,
                labelspacing=0.35,
                handletextpad=0.4,
                columnspacing=0.9,
                borderpad=0.45,
                frameon=True,
                framealpha=0.96,
                edgecolor="#cccccc",
                fancybox=False,
                borderaxespad=0.45,
                ncol=back_ncol,
            )
            leg_back.get_frame().set_linewidth(0.6)
    else:
        leg = ax.legend(
            loc="lower right",
            fontsize=_legend_fs,
            markerscale=1.45,
            labelspacing=0.4,
            handletextpad=0.6,
            borderpad=0.6,
            frameon=True,
            framealpha=0.96,
            edgecolor="#cccccc",
            fancybox=False,
            borderaxespad=0.45,
        )
        leg.get_frame().set_linewidth(0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {output_path}")
    print(f"           and: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Accuracy vs Adaptation Training FLOPs")
    parser.add_argument("--top-k", type=int, default=2,
                        help="Number of best positive alphas to keep "
                             "(also adds their back-loaded counterparts). Default: 2.")
    parser.add_argument("--xcut", choices=["second-last", "last", "none"],
                        default="second-last",
                        help="X-axis truncation mode (default: second-last "
                             "Joint LoRA point)")
    parser.add_argument("--width", type=float, default=8.0,
                        help="Figure width in inches (default: 8.0)")
    parser.add_argument("--height", type=float, default=4.8,
                        help="Figure height in inches (default: 4.8)")
    args = parser.parse_args()

    results = load_ablation_results(args.results_dir)
    if not results:
        print("No results found")
        return

    all_groups = extract_accuracy_flops_points(results)
    print(f"Loaded {len(results)} files; {len(all_groups)} strategies")

    top_alphas = select_top_alphas(all_groups, args.top_k, verbose=True)
    if not top_alphas:
        print("No positive alphas with measurable points; nothing to plot.")
        return
    print(f"\n[final] Selected top-{args.top_k} alphas: {top_alphas}\n")

    selected = select_groups(all_groups, top_alphas)

    joint = selected.get("Joint LoRA", [])
    if joint and args.xcut != "none":
        sorted_flops = sorted(f for f, _ in joint)
        if args.xcut == "second-last" and len(sorted_flops) >= 2:
            cutoff = sorted_flops[-2]
        elif args.xcut == "last":
            cutoff = sorted_flops[-1]
        else:
            cutoff = float("inf")
        selected = truncate_x(selected, cutoff)
        kept_joint = selected.get("Joint LoRA", [])
        if kept_joint:
            print(f"[final] X-axis truncated at {cutoff:.2f} GF "
                  f"(Joint LoRA last kept = {max(f for f, _ in kept_joint):.2f} GF)")

    render(selected, top_alphas, args.title, args.output,
           figsize=(args.width, args.height))


if __name__ == "__main__":
    main()
