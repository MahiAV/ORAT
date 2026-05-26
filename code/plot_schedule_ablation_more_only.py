"""Final-style accuracy-vs-FLOPs plot, but only ``Joint LoRA`` + ``Equal`` +
the top-K **More first** schedules (no ``Less first`` counterparts).

This is a thin wrapper around ``plot_schedule_ablation_final.py``: same scoring,
same x-axis truncation, same overall styling -- just drops the back-loaded
shades from ``select_groups``/``render``.

Usage:
    python plot_schedule_ablation_more_only.py \\
        --results-dir efficient_ablation_sst2_fixed \\
        --output figures_efficient_ablation/accuracy_vs_flops_sst2_more_first_top4.png \\
        --title "SST-2" \\
        --top-k 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_schedule_ablation import (  # noqa: E402
    extract_accuracy_flops_points,
    load_ablation_results,
)
from plot_schedule_ablation_final import (  # noqa: E402
    render,
    select_top_alphas,
    truncate_x,
)


def select_groups_more_only(
    grouped_points: Dict[str, List[Tuple[float, float]]],
    top_alphas: List[float],
) -> Dict[str, List[Tuple[float, float]]]:
    """Like ``plot_schedule_ablation_final.select_groups`` but skips back-loaded."""
    out: Dict[str, List[Tuple[float, float]]] = {}
    if grouped_points.get("Joint LoRA"):
        out["Joint LoRA"] = list(grouped_points["Joint LoRA"])
    if grouped_points.get("Equal Schedule"):
        out["Equal Schedule"] = list(grouped_points["Equal Schedule"])
    for a in top_alphas:
        front_name = f"Front-loaded (α={a:.1f})"
        if grouped_points.get(front_name):
            out[front_name] = list(grouped_points[front_name])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="More first schedules")
    parser.add_argument(
        "--top-k", type=int, default=4,
        help="Number of best positive alphas to keep. Ignored if --alphas is set. Default: 4.",
    )
    parser.add_argument(
        "--alphas", nargs="+", default=None,
        help=(
            "Explicit list of positive alphas to plot (overrides --top-k). "
            "Pass 'all' to take every positive alpha present in the results."
        ),
    )
    parser.add_argument(
        "--xcut", choices=["second-last", "last", "none"], default="second-last",
        help="X-axis truncation mode (default: second-last Joint LoRA point).",
    )
    parser.add_argument(
        "--drop-joint-tail", type=int, default=None,
        help="Drop this many highest-FLOPs Joint LoRA points (overrides --xcut).",
    )
    parser.add_argument(
        "--compact-legend", action="store_true",
        help="Two-row legend with a single 'More first (alpha)' header.",
    )
    parser.add_argument("--width", type=float, default=8.0)
    parser.add_argument("--height", type=float, default=4.8)
    args = parser.parse_args()

    results = load_ablation_results(args.results_dir)
    if not results:
        print("No results found")
        return

    all_groups = extract_accuracy_flops_points(results)
    print(f"Loaded {len(results)} files; {len(all_groups)} strategies")

    if args.alphas is not None:
        if any(str(a).strip().lower() == "all" for a in args.alphas):
            top_alphas = sorted({
                float(a) for name in all_groups
                if name.startswith("Front-loaded")
                for a in [float(name.split("=")[-1].rstrip(")"))]
                if a > 0
            })
            print(f"\n[more-only] Using all positive alphas present: {top_alphas}\n")
        else:
            top_alphas = sorted(set(float(a) for a in args.alphas if float(a) > 0))
            present = [
                a for a in top_alphas
                if all_groups.get(f"Front-loaded (α={a:.1f})")
            ]
            missing = [a for a in top_alphas if a not in present]
            if missing:
                print(f"[more-only] No data for alpha(s): {missing}; skipping them.")
            top_alphas = present
            print(f"\n[more-only] Using explicit alphas: {top_alphas}\n")
        select_top_alphas(all_groups, len(top_alphas), verbose=True)
    else:
        top_alphas = select_top_alphas(all_groups, args.top_k, verbose=True)
        if not top_alphas:
            print("No positive alphas with measurable points; nothing to plot.")
            return
        print(f"\n[more-only] Selected top-{args.top_k} alphas: {top_alphas}\n")

    selected = select_groups_more_only(all_groups, top_alphas)

    joint = selected.get("Joint LoRA", [])
    if joint:
        sorted_flops = sorted(f for f, _ in joint)
        cutoff: float = float("inf")
        if args.drop_joint_tail is not None:
            n_drop = max(0, int(args.drop_joint_tail))
            if n_drop >= len(sorted_flops):
                print(
                    f"[more-only] --drop-joint-tail={n_drop} is >= total Joint LoRA points "
                    f"({len(sorted_flops)}); nothing left, ignoring."
                )
            elif n_drop > 0:
                cutoff = sorted_flops[-(n_drop + 1)]
        elif args.xcut == "second-last" and len(sorted_flops) >= 2:
            cutoff = sorted_flops[-2]
        elif args.xcut == "last":
            cutoff = sorted_flops[-1]

        if cutoff != float("inf"):
            selected = truncate_x(selected, cutoff)
            kept_joint = selected.get("Joint LoRA", [])
            if kept_joint:
                print(
                    f"[more-only] X-axis truncated at {cutoff:.2f} GF "
                    f"(Joint LoRA last kept = {max(f for f, _ in kept_joint):.2f} GF)"
                )

    render(
        selected,
        top_alphas,
        args.title,
        args.output,
        figsize=(args.width, args.height),
        compact_legend=args.compact_legend,
    )


if __name__ == "__main__":
    main()
