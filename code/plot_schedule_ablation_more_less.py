"""Schedule-only ablation plot: Equal vs all "More first" vs all "Less first".

This is a companion to ``plot_schedule_ablation_final.py``. The difference
is the comparison baseline:

* ``plot_schedule_ablation_final.py`` ranks alphas against **Joint LoRA**
  (chosen across budgets) and keeps the top-K winners + their negatives.
* ``plot_schedule_ablation_more_less.py`` (this file) ranks alphas against
  the **Equal Schedule** (the matched-budget α=0 sequential baseline) and
  keeps every alpha that strictly beats the Equal Schedule at *every*
  matched-FLOPs point. Joint LoRA is intentionally **not plotted**.

Color scheme (matches ``reproduce_fig_combined.py``):

* Equal Schedule: green.
* More first  (α > 0, front-loaded): all in shades of orange,
  larger ``α`` => darker shade.
* Less first  (α < 0, back-loaded):  all in shades of blue,
  larger ``|α|`` => darker shade.

Filtering rule:
We define a positive alpha ``α`` as a *winner* if its front-loaded variant
or its back-loaded variant beats the Equal Schedule (linearly interpolated
at the same FLOPs) on at least ``--threshold`` (default 0.8) of the
overlapping points. If a positive alpha is a winner, both its front-loaded
and back-loaded curves are kept (so the figure always shows the symmetric
pair).

Usage:
    python plot_schedule_ablation_more_less.py \\
        --results-dir efficient_ablation_cifar100_fixed \\
        --output figures_efficient_ablation/accuracy_vs_flops_cifar100_more_less.png \\
        --title "CIFAR-100"
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
from plot_schedule_ablation_final import (
    EQUAL_COLOR,
    LESS_FIRST_BASE,
    MORE_FIRST_BASE,
    DARK_L,
    LIGHT_L,
    _shade_palette,
    truncate_x,
)


def _curve(points: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.array([]), np.array([])
    flops, acc = zip(*sorted(points))
    return np.array(flops, dtype=float), np.array(acc, dtype=float)


def _winrate_vs_ref(
    points: List[Tuple[float, float]],
    ref_flops: np.ndarray,
    ref_acc: np.ndarray,
    margin: float = 0.0,
) -> Tuple[float, int, int]:
    """Return (win_rate, n_wins, n_compared) against the reference curve.

    A point is a *win* if the schedule's accuracy is strictly greater than
    the linearly-interpolated reference accuracy by at least ``margin``.
    Points outside the reference's FLOPs range are skipped from both numerator
    and denominator.
    """
    if len(ref_flops) < 2 or not points:
        return 0.0, 0, 0
    f_min, f_max = float(ref_flops[0]), float(ref_flops[-1])
    n_compared = 0
    n_wins = 0
    for f, a in points:
        if f < f_min or f > f_max:
            continue
        n_compared += 1
        ref = float(np.interp(f, ref_flops, ref_acc))
        if a > ref + margin:
            n_wins += 1
    if n_compared == 0:
        return 0.0, 0, 0
    return n_wins / n_compared, n_wins, n_compared


def select_winners_vs_equal(
    grouped_points: Dict[str, List[Tuple[float, float]]],
    threshold: float = 0.8,
    margin: float = 0.0,
    verbose: bool = True,
) -> List[float]:
    """Return positive alphas where the front-loaded *or* back-loaded variant
    beats the Equal Schedule on at least ``threshold`` fraction of overlapping
    points.
    """
    equal_points = grouped_points.get("Equal Schedule", [])
    ef, ea = _curve(equal_points)
    if ef.size < 2:
        if verbose:
            print("[more_less] Equal Schedule reference is empty; cannot filter.")
        return []

    # Collect unique positive alphas.
    alphas: List[float] = []
    for name in grouped_points:
        if not (name.startswith("Front-loaded") or name.startswith("Back-loaded")):
            continue
        a = _parse_alpha(name)
        if a is None or a <= 0:
            continue
        if a not in alphas:
            alphas.append(a)
    alphas.sort()

    if verbose:
        print(f"\n[more_less] Equal Schedule reference: {len(equal_points)} points "
              f"({ef[0]:.2f} – {ef[-1]:.2f} GF), threshold={threshold:.0%}")
        print(f"  {'alpha':<7} {'F-winrate':>10} {'F-w/n':>9}  "
              f"{'B-winrate':>10} {'B-w/n':>9}  kept?")
        print("  " + "-" * 60)

    winners: List[float] = []
    for a in alphas:
        front = grouped_points.get(f"Front-loaded (α={a:.1f})", [])
        back = grouped_points.get(f"Back-loaded (α={a:.1f})", [])
        f_rate, f_w, f_n = _winrate_vs_ref(front, ef, ea, margin=margin)
        b_rate, b_w, b_n = _winrate_vs_ref(back, ef, ea, margin=margin)
        f_pass = (f_n > 0) and (f_rate >= threshold)
        b_pass = (b_n > 0) and (b_rate >= threshold)
        kept = f_pass or b_pass
        if kept:
            winners.append(a)
        if verbose:
            mark = " *" if kept else ""
            print(f"  {a:<7.1f} {f_rate:>10.0%} {f_w:>3d}/{f_n:<5d}  "
                  f"{b_rate:>10.0%} {b_w:>3d}/{b_n:<5d}{mark}")
    return winners


def select_groups_more_less(
    grouped_points: Dict[str, List[Tuple[float, float]]],
    winners: List[float],
) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {}
    if grouped_points.get("Equal Schedule"):
        out["Equal Schedule"] = list(grouped_points["Equal Schedule"])
    for a in winners:
        front_name = f"Front-loaded (α={a:.1f})"
        back_name = f"Back-loaded (α={a:.1f})"
        if grouped_points.get(front_name):
            out[front_name] = list(grouped_points[front_name])
        if grouped_points.get(back_name):
            out[back_name] = list(grouped_points[back_name])
    return out


def render(
    grouped_points: Dict[str, List[Tuple[float, float]]],
    winners: List[float],
    title: str,
    output_path: str,
    figsize: Tuple[float, float] = (8.0, 4.8),
) -> None:
    sorted_alphas = sorted(winners)
    n_alphas = len(sorted_alphas)
    orange_shades = _shade_palette(MORE_FIRST_BASE, n_alphas)
    blue_shades = _shade_palette(LESS_FIRST_BASE, n_alphas)

    colors: Dict[str, Tuple[float, float, float] | str] = {
        "Equal Schedule": EQUAL_COLOR,
    }
    markers: Dict[str, str] = {
        "Equal Schedule": "s",
    }
    labels: Dict[str, str] = {
        "Equal Schedule": r"Equal schedule ($\alpha = 0$)",
    }

    for i, alpha in enumerate(sorted_alphas):
        front = f"Front-loaded (α={alpha:.1f})"
        back = f"Back-loaded (α={alpha:.1f})"
        colors[front] = orange_shades[i]
        colors[back] = blue_shades[i]
        markers[front] = "^"
        markers[back] = "v"
        labels[front] = rf"More first ($\alpha = {alpha:g}$)"
        labels[back] = rf"Less first ($\alpha = -{alpha:g}$)"

    # Plot order. Place Equal Schedule last so it sits on top of the schedule
    # spaghetti as the visual anchor.
    order: List[str] = []
    for a in sorted_alphas:
        order.append(f"Back-loaded (α={a:.1f})")
        order.append(f"Front-loaded (α={a:.1f})")
    order.append("Equal Schedule")

    # 2× the original defaults (12.5 → 25 for axis titles, 10.5 → 21 ticks).
    _label_fs = 15.0
    _tick_fs = 12.0
    _title_fs = 20.0

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": _label_fs,
        "axes.titlesize": _title_fs,
        "xtick.labelsize": _tick_fs,
        "ytick.labelsize": _tick_fs,
        "legend.fontsize": 20,
        "lines.linewidth": 2.0,
        "axes.linewidth": 0.8,
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # No constrained_layout so a bottom-anchored legend can sit on/near the
    # x-axis line without fighting the layout engine.
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
        is_equal = name == "Equal Schedule"

        ax.plot(
            flops, acc,
            color=color,
            linewidth=2.6 if is_equal else 1.7,
            alpha=0.95 if is_equal else 0.85,
            linestyle="-",
            zorder=4 if is_equal else 2,
        )
        ax.scatter(
            flops, acc,
            color=color,
            marker=marker,
            s=85 if is_equal else 50,
            edgecolor="white",
            linewidth=0.9,
            label=label,
            zorder=5 if is_equal else 3,
        )

    ax.set_xlabel(
        "Adaptation Training FLOPs (GFLOPs)",
        fontsize=_label_fs,
        fontweight="bold",
    )
    ax.set_ylabel("Test Accuracy", fontsize=_label_fs, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=_title_fs, fontweight="bold")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(colors="#444444", axis="both", labelsize=_tick_fs)

    # ------------------------------------------------------------------
    # Three-row legend (compact, no repeated text):
    #
    #   Less first:   ▽ -α₁   ▽ -α₂   ▽ -α₃   ...   ▽ -αₙ
    #   More first:   △  α₁   △  α₂   △  α₃   ...   △  αₙ
    #   Equal:        ■   0
    #
    # The row title is a single text entry per row (invisible handle).
    # Each subsequent column shows just the colored marker + numeric α
    # (no "Less first" or "α=" repeated), making the box ~2× more compact
    # while staying readable at a larger font size.
    #
    # Matplotlib fills the legend column-major with ``ncol=K``: entries
    # 0..ceil(N/K)-1 go to column 1, etc.  We have 3 entries per column
    # (one per row), so we order entries column-by-column:
    #   col 1: [Less title, More title, Equal title]
    #   col 2: [-α₁,        +α₁,        0]
    #   col 3: [-α₂,        +α₂,        padding]
    #   ...
    # ------------------------------------------------------------------
    from matplotlib.lines import Line2D

    def _legend_marker_proxy(marker: str, facecolor, *, ms: float = 13.5) -> Line2D:
        """Large legend glyphs (triangles / square) with white edge like the plot."""
        return Line2D(
            [], [],
            linestyle="None",
            marker=marker,
            color="none",
            markerfacecolor=facecolor,
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=ms,
        )

    all_handles, all_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(all_labels, all_handles))

    equal_lab = r"Equal schedule ($\alpha = 0$)"
    _ = by_label.get(equal_lab)  # scatter was registered; proxies replace legend

    alphas_sorted = sorted(sorted_alphas)
    _legend_scale = 1.2
    _ms_leg = 10.0 * _legend_scale

    less_proxies: List[Line2D] = []
    more_proxies: List[Line2D] = []
    less_alpha_strs: List[str] = []
    more_alpha_strs: List[str] = []
    for j, a in enumerate(alphas_sorted):
        less_lab_full = rf"Less first ($\alpha = -{a:g}$)"
        more_lab_full = rf"More first ($\alpha = {a:g}$)"
        if less_lab_full in by_label:
            less_proxies.append(_legend_marker_proxy("v", blue_shades[j], ms=_ms_leg))
            less_alpha_strs.append(rf"$\alpha = -{a:g}$")
        if more_lab_full in by_label:
            more_proxies.append(_legend_marker_proxy("^", orange_shades[j], ms=_ms_leg))
            more_alpha_strs.append(rf"$\alpha = {a:g}$")

    n_alphas_legend = max(len(less_proxies), len(more_proxies))
    equal_proxy = _legend_marker_proxy("s", EQUAL_COLOR, ms=_ms_leg)

    spacer = Line2D([], [], linestyle="None", marker="None", color="none")

    ncol_legend = n_alphas_legend + 1  # 1 title col + n alpha cols
    handles_legend: List = []
    labels_legend: List[str] = []

    # Column 1 — row titles (text only, no markers).
    handles_legend.extend([spacer, spacer, spacer])
    labels_legend.extend(["Less first:", "More first:", "Equal:"])

    # Columns 2..n+1 — alpha values (one less + one more + maybe equal).
    for i in range(n_alphas_legend):
        if i < len(less_proxies):
            handles_legend.append(less_proxies[i])
            labels_legend.append(less_alpha_strs[i])
        else:
            handles_legend.append(spacer)
            labels_legend.append("")
        if i < len(more_proxies):
            handles_legend.append(more_proxies[i])
            labels_legend.append(more_alpha_strs[i])
        else:
            handles_legend.append(spacer)
            labels_legend.append("")
        if i == 0 and (less_proxies or more_proxies):
            handles_legend.append(equal_proxy)
            labels_legend.append(r"$\alpha = 0$")
        else:
            handles_legend.append(spacer)
            labels_legend.append("")

    # Previous legend body sizes were ~12.5–16 pt; user asked ~1.7× smaller.
    if n_alphas_legend <= 4:
        leg_font = 16.0 / 1.7
    elif n_alphas_legend <= 6:
        leg_font = 14.5 / 1.7
    else:
        leg_font = 12.5 / 1.7
    leg_font *= _legend_scale

    plt.tight_layout()
    # Lower-right: flush to the right edge, near the x-axis.
    leg = ax.legend(
        handles_legend, labels_legend,
        ncol=ncol_legend,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.01),
        bbox_transform=ax.transAxes,
        columnspacing=0.52,
        handletextpad=0.24,
        labelspacing=0.26,
        fontsize=leg_font,
        markerscale=1.0,
        frameon=True,
        framealpha=0.93,
        edgecolor="#cccccc",
        fancybox=False,
        borderpad=0.28,
    )
    leg.get_frame().set_linewidth(0.6)
    for text in leg.get_texts()[:3]:
        text.set_horizontalalignment("left")
        text.set_fontweight("semibold")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        bbox_extra_artists=[leg],
        pad_inches=0.08,
    )
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        bbox_extra_artists=[leg],
        pad_inches=0.08,
    )
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
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Minimum win-rate against Equal Schedule required "
                             "to keep an alpha (and its negative). Default: 0.80.")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="Required strict accuracy margin over the Equal "
                             "Schedule at each overlapping FLOPs point. "
                             "Default: 0 (any strict win counts).")
    parser.add_argument("--xcut", choices=["second-last", "last", "none"],
                        default="second-last",
                        help="X-axis truncation: cut at second-last Equal Schedule "
                             "point (default), the last point, or do not cut.")
    parser.add_argument("--width", type=float, default=8.0,
                        help="Figure width in inches (default: 8.0)")
    parser.add_argument("--height", type=float, default=6,
                        help="Figure height in inches (default: 4.8)")
    args = parser.parse_args()

    results = load_ablation_results(args.results_dir)
    if not results:
        print("No results found")
        return

    all_groups = extract_accuracy_flops_points(results)
    print(f"Loaded {len(results)} files; {len(all_groups)} strategies")

    winners = select_winners_vs_equal(
        all_groups, threshold=args.threshold, margin=args.margin, verbose=True
    )
    if not winners:
        print(f"\nNo schedules clear the {args.threshold:.0%} win-rate threshold "
              f"vs Equal Schedule. Nothing to plot.")
        return
    print(f"\n[more_less] Kept {len(winners)} alpha pairs: {winners}\n")

    selected = select_groups_more_less(all_groups, winners)

    equal = selected.get("Equal Schedule", [])
    if equal and args.xcut != "none":
        sorted_flops = sorted(f for f, _ in equal)
        if args.xcut == "second-last" and len(sorted_flops) >= 2:
            cutoff = sorted_flops[-2]
        elif args.xcut == "last":
            cutoff = sorted_flops[-1]
        else:
            cutoff = float("inf")
        selected = truncate_x(selected, cutoff)
        kept_equal = selected.get("Equal Schedule", [])
        if kept_equal:
            print(f"[more_less] X-axis truncated at {cutoff:.2f} GF "
                  f"(Equal last kept = {max(f for f, _ in kept_equal):.2f} GF)")

    render(selected, winners, args.title, args.output,
           figsize=(args.width, args.height))


if __name__ == "__main__":
    main()
