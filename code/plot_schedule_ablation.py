"""Plot accuracy vs LoRA-only FLOPs for schedule ablation results.

This script reads the results from run_schedule_ablation.py and creates
a scatter plot showing the compute-efficiency tradeoff for different
allocation strategies.

Usage:
    python plot_schedule_ablation.py --results-dir ablation_cifar10 \\
        --output accuracy_vs_flops_cifar10.png --title "CIFAR-10 Schedule Ablation"
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lora_lib.flops import per_sample_flops_breakdown
from lora_lib.plot_style import xkcd_style
from schedule_ablation import estimate_lora_only_flops


_ALPHA_RE = re.compile(r"\(α=([0-9.]+)\)")


def _set_lightness(color, lightness: float) -> Tuple[float, float, float]:
    """Return ``color`` with HLS lightness replaced by ``lightness`` in [0, 1].

    Smaller lightness => darker shade; larger => lighter shade.
    Hue and saturation are preserved.
    """
    r, g, b = mcolors.to_rgb(color)
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, min(1.0, lightness)), s)


def _alpha_palette(alphas: List[float]) -> Dict[float, Tuple[float, float, float]]:
    """Map each unique alpha to a distinct base hue, evenly spaced on the
    color wheel. Excludes the hue around black/white so dark/light shades stay
    distinguishable.
    """
    palette: Dict[float, Tuple[float, float, float]] = {}
    if not alphas:
        return palette
    n = len(alphas)
    # Spread hues across [0, 1); use mid lightness so dark/light variants
    # are clearly separable later.
    for i, alpha in enumerate(sorted(alphas)):
        hue = (i / n) % 1.0
        palette[alpha] = colorsys.hls_to_rgb(hue, 0.50, 0.85)
    return palette


def _parse_alpha(group_name: str) -> float | None:
    m = _ALPHA_RE.search(group_name)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def load_ablation_results(results_dir: str) -> List[Dict]:
    """Load all result JSON files from the ablation directory."""
    results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath) as f:
                    data = json.load(f)
                
                config_name = filename[:-5]  # Remove .json
                data["config_name"] = config_name
                results.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
    
    return results


def extract_accuracy_flops_points(results: List[Dict]) -> Dict[str, List[Tuple[float, float]]]:
    """
    Extract (LoRA_FLOPs, accuracy) points grouped by strategy type.
    
    Returns:
        Dict mapping strategy names to lists of (flops_gf, accuracy) tuples
    """
    groups = {
        "Joint LoRA": [],
        "Equal Schedule": [],
    }
    
    for result in results:
        config = result.get("ablation_config", {})
        config_name = result["config_name"]
        
        # Extract accuracy
        if config.get("type") == "joint":
            # Joint LoRA: get accuracy from standard_lora
            accuracy = result.get("standard_lora", {}).get("final_accuracy")
            if accuracy is None:
                continue
        else:
            # Sequential: get final accuracy from the sequential path
            seq_paths = result.get("sequential_paths", {})
            if not seq_paths:
                continue
            # Take the first (and likely only) sequential path
            path_result = next(iter(seq_paths.values()))
            accuracy = path_result.get("final_accuracy")
            if accuracy is None:
                continue
        
        # Extract FLOPs info
        flops_meta = result.get("flops", {})
        if not flops_meta:
            continue
            
        layers = [tuple(pair) for pair in flops_meta["layers"]]
        samples_per_epoch = flops_meta["samples_per_epoch"]
        
        # Calculate LoRA-only FLOPs
        lora_flops = estimate_lora_only_flops(config, layers, samples_per_epoch)
        lora_flops_gf = lora_flops / 1e9  # Convert to GFLOPs
        
        # Categorize by strategy
        if config.get("type") == "joint":
            groups["Joint LoRA"].append((lora_flops_gf, accuracy))
        else:
            alpha = config.get("alpha", 0.0)
            if alpha == 0.0:
                groups["Equal Schedule"].append((lora_flops_gf, accuracy))
            elif alpha > 0:
                # Front-loaded - create group name based on actual alpha value
                group_name = f"Front-loaded (α={alpha:.1f})"
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append((lora_flops_gf, accuracy))
            else:
                # Back-loaded - create group name based on actual alpha value  
                group_name = f"Back-loaded (α={abs(alpha):.1f})"
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append((lora_flops_gf, accuracy))
    
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


def plot_accuracy_vs_flops(
    grouped_points: Dict[str, List[Tuple[float, float]]],
    title: str,
    output_path: str,
) -> None:
    """Create the accuracy vs FLOPs scatter plot.

    Color/marker policy:
      - "Joint LoRA"          -> black solid circle
      - "Equal Schedule"      -> blue square
      - "Front-loaded (α=X)"  -> dark shade of the hue assigned to alpha=X (triangle up)
      - "Back-loaded (α=X)"   -> light shade of the same hue                (triangle down)

    Front and back of the same |alpha| share a hue so they read as a pair;
    the dark/light split makes the trend direction visually obvious.
    """
    # Fixed assignments for the baselines.
    colors: Dict[str, Tuple[float, float, float] | str] = {
        "Joint LoRA": "#000000",       # black
        "Equal Schedule": "#3498DB",   # blue
    }
    markers: Dict[str, str] = {
        "Joint LoRA": "o",
        "Equal Schedule": "s",
    }

    # Collect the union of alpha values used by Front-/Back-loaded groups so
    # both sides of a pair end up sharing a hue.
    schedule_alphas: List[float] = []
    for name in grouped_points.keys():
        if name.startswith("Front-loaded") or name.startswith("Back-loaded"):
            a = _parse_alpha(name)
            if a is not None and a not in schedule_alphas:
                schedule_alphas.append(a)
    palette = _alpha_palette(schedule_alphas)

    # Lightness levels for dark (front-loaded) and light (back-loaded) shades.
    DARK_L = 0.30
    LIGHT_L = 0.72

    for name in grouped_points.keys():
        if name.startswith("Front-loaded"):
            a = _parse_alpha(name)
            base = palette.get(a, (0.5, 0.5, 0.5))
            colors[name] = _set_lightness(base, DARK_L)
            markers[name] = "^"
        elif name.startswith("Back-loaded"):
            a = _parse_alpha(name)
            base = palette.get(a, (0.5, 0.5, 0.5))
            colors[name] = _set_lightness(base, LIGHT_L)
            markers[name] = "v"
    
    with xkcd_style(scale=0.8, length=120, randomness=3):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each group
        for group_name, points in grouped_points.items():
            if not points:
                continue
                
            flops_vals, acc_vals = zip(*points)
            
            # Sort by FLOPs for connecting lines
            sorted_pairs = sorted(zip(flops_vals, acc_vals))
            sorted_flops, sorted_acc = zip(*sorted_pairs)
            
            color = colors.get(group_name, "#7F8C8D")
            marker = markers.get(group_name, "o")

            # Joint LoRA gets a slightly larger marker + thicker line so it
            # reads as the reference; back-loaded uses a thin dark outline so
            # the light shades stay visible against the figure background.
            is_joint = (group_name == "Joint LoRA")
            is_back = group_name.startswith("Back-loaded")
            scatter_kwargs = dict(
                c=[color], marker=marker,
                s=110 if is_joint else 80,
                label=group_name,
                alpha=0.9 if is_joint else 0.85,
                edgecolor="black" if is_back else ("black" if is_joint else "white"),
                linewidth=0.8 if is_back else (1.2 if is_joint else 1.0),
            )
            ax.scatter(sorted_flops, sorted_acc, **scatter_kwargs)

            # Connect points with lines for the same strategy
            if len(sorted_flops) > 1:
                ax.plot(sorted_flops, sorted_acc,
                       color=color,
                       alpha=0.6 if is_joint else 0.4,
                       linewidth=3 if is_joint else 2,
                       linestyle="-")
        
        ax.set_xlabel("LoRA-only Training FLOPs (GFLOPs)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Test Accuracy", fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")
        
        # Formatting
        ax.grid(True, alpha=0.3, linestyle="-")
        ax.set_axisbelow(True)
        
        # Legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", 
                 fontsize=10, frameon=True, framealpha=0.95)
        
        # Tight layout
        plt.tight_layout()
        
        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Saved plot to: {output_path}")


def print_summary_stats(grouped_points: Dict[str, List[Tuple[float, float]]]) -> None:
    """Print summary statistics for each strategy."""
    print("\nSummary Statistics:")
    print("=" * 60)
    
    for strategy, points in grouped_points.items():
        if not points:
            continue
            
        flops_vals, acc_vals = zip(*points)
        min_flops, max_flops = min(flops_vals), max(flops_vals)
        min_acc, max_acc = min(acc_vals), max(acc_vals)
        
        print(f"{strategy}:")
        print(f"  Points: {len(points)}")
        print(f"  FLOPs range: {min_flops:.1f} - {max_flops:.1f} GF")
        print(f"  Accuracy range: {min_acc:.3f} - {max_acc:.3f}")
        
        # Find most efficient point (highest accuracy per FLOP)
        if points:
            best_efficiency_idx = max(range(len(points)), 
                                    key=lambda i: acc_vals[i] / flops_vals[i])
            best_flops, best_acc = points[best_efficiency_idx]
            efficiency = best_acc / best_flops
            print(f"  Best efficiency: {best_acc:.3f} acc @ {best_flops:.1f} GF ({efficiency:.4f} acc/GF)")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing ablation result JSONs")
    parser.add_argument("--output", required=True,
                        help="Output PNG path")
    parser.add_argument("--title", default="Schedule Ablation: Accuracy vs LoRA-only FLOPs",
                        help="Plot title")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary statistics")
    
    args = parser.parse_args()
    
    # Load results
    results = load_ablation_results(args.results_dir)
    print(f"Loaded {len(results)} result files from {args.results_dir}")
    
    if not results:
        print("No results found!")
        return
    
    # Extract points
    grouped_points = extract_accuracy_flops_points(results)
    
    total_points = sum(len(points) for points in grouped_points.values())
    print(f"Extracted {total_points} data points across {len(grouped_points)} strategies")
    
    if total_points == 0:
        print("No valid data points found!")
        return
    
    # Generate plot
    plot_accuracy_vs_flops(grouped_points, args.title, args.output)
    
    # Print summary if requested
    if args.summary:
        print_summary_stats(grouped_points)


if __name__ == "__main__":
    main()