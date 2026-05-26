"""Schedule ablation study: accuracy vs LoRA-only FLOPs for different allocation strategies.

This module implements the ablation study comparing:
- Joint LoRA (various epoch budgets)
- Equal schedules (balanced α-β-γ with various budgets) 
- Front-loaded schedules (alpha > 0, more epochs on early components)
- Back-loaded schedules (alpha < 0, more epochs on later components)

The goal is to plot accuracy vs LoRA-only FLOPs to see which allocation
strategy is most compute-efficient at different budget levels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lora_lib.flops import per_sample_flops_breakdown


def allocate_iterations_alpha(r: int, total_iters: int, alpha: float, min_per_rank: int = 1) -> List[int]:
    """
    Allocation schedule with linear trend control (adapted from reproduce_fig_combined.py).

    Let k = 1,...,r and define a centered position x_k in [1, -1] from early to late.
    We assign scores linearly as:
        s_k = 1 + alpha * x_k,
    then normalize scores to allocate the extra iteration budget.

    - alpha = 0   -> uniform allocation (equal schedule)
    - alpha > 0   -> more iterations on earlier ranks (front-loaded)
    - alpha < 0   -> more iterations on later ranks (back-loaded)

    Args:
        r: Number of sequential components (rank)
        total_iters: Total epoch budget to allocate
        alpha: Trend parameter (-1 to 1 recommended)
        min_per_rank: Minimum epochs per component

    Returns:
        List of epoch allocations [α, β, γ] for r=3
    """
    if r <= 0:
        raise ValueError("r must be positive")
    if min_per_rank < 0:
        raise ValueError("min_per_rank must be non-negative")
    
    base = r * min_per_rank
    if total_iters < base:
        raise ValueError(
            f"total_iters={total_iters} is too small for r={r} with min_per_rank={min_per_rank}"
        )

    extra_budget = total_iters - base
    if extra_budget == 0:
        return [min_per_rank] * r

    # Linear trend from early (x=1) to late (x=-1)
    x = np.linspace(1.0, -1.0, r, dtype=float)
    scores = 1.0 + alpha * x
    # Keep scores strictly positive
    scores = np.maximum(scores, 1e-9)
    
    # Normalize to allocate extra budget
    weights = scores / scores.sum()
    extra_real = extra_budget * weights
    extra_int = np.floor(extra_real).astype(int)
    
    # Distribute remainder by largest fractional parts
    remainder = int(extra_budget - extra_int.sum())
    if remainder > 0:
        frac = extra_real - extra_int
        top_indices = np.argsort(-frac)[:remainder]
        extra_int[top_indices] += 1

    schedule = min_per_rank + extra_int
    return schedule.astype(int).tolist()


def generate_schedule_grid(
    epoch_budgets: List[int],
    alphas: List[float],
    rank: int = 3,
    min_per_rank: int = 1,
) -> Dict[str, Dict[str, any]]:
    """
    Generate a grid of all schedule configurations for the ablation.
    
    Returns a dict with keys like "joint_10", "equal_15", "front_0.5_18", "back_0.3_12"
    mapping to configuration dictionaries.
    """
    configs = {}
    
    # Joint LoRA baselines (various epoch budgets)
    for epochs in epoch_budgets:
        configs[f"joint_{epochs}"] = {
            "type": "joint",
            "rank": rank,
            "epochs": epochs,
            "alpha": None,
            "schedule": None,
        }
    
    # Sequential schedules
    for budget in epoch_budgets:
        for alpha in alphas:
            # Create both positive and negative alpha (unless alpha=0)
            alpha_values = [alpha]
            if alpha > 0:
                alpha_values.append(-alpha)  # Add back-loaded version
            
            for actual_alpha in alpha_values:
                try:
                    schedule = allocate_iterations_alpha(rank, budget, actual_alpha, min_per_rank)
                    schedule_str = "-".join(map(str, schedule))
                    
                    if actual_alpha == 0.0:
                        # Equal/balanced schedule
                        name = f"equal_{budget}"
                    elif actual_alpha > 0:
                        # Front-loaded
                        name = f"front_{actual_alpha:.1f}_{budget}"
                    else:
                        # Back-loaded  
                        name = f"back_{abs(actual_alpha):.1f}_{budget}"
                    
                    configs[name] = {
                        "type": "sequential", 
                        "rank": rank,
                        "total_epochs": budget,
                        "alpha": actual_alpha,
                        "schedule": schedule,
                        "schedule_str": schedule_str,
                    }
                except ValueError:
                    # Skip invalid configurations (budget too small)
                    continue
    
    return configs


def estimate_lora_only_flops(
    config: Dict[str, any],
    layers: List[Tuple[int, int]], 
    samples_per_epoch: int,
) -> float:
    """
    Calculate LoRA-only FLOPs (base cost subtracted) for a configuration.
    
    Args:
        config: Configuration dict from generate_schedule_grid
        layers: List of (input_dim, output_dim) for LoRA-adapted layers
        samples_per_epoch: Training set size
        
    Returns:
        Total LoRA-only FLOPs for this configuration
    """
    base_per_sample = per_sample_flops_breakdown(layers, r_active=0, r_train=0).base
    
    if config["type"] == "joint":
        # Joint LoRA: train all r components together for `epochs` epochs
        rank = config["rank"]
        epochs = config["epochs"]
        total_per_sample = per_sample_flops_breakdown(layers, r_active=rank, r_train=rank).total
        lora_per_sample = total_per_sample - base_per_sample
        return lora_per_sample * samples_per_epoch * epochs
    
    elif config["type"] == "sequential":
        # Sequential: sum over components
        schedule = config["schedule"]
        total_lora_flops = 0
        for k, epochs in enumerate(schedule, start=1):
            total_per_sample = per_sample_flops_breakdown(layers, r_active=k, r_train=1).total
            lora_per_sample = total_per_sample - base_per_sample
            total_lora_flops += lora_per_sample * samples_per_epoch * epochs
        return total_lora_flops
    
    else:
        raise ValueError(f"Unknown config type: {config['type']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-budgets", nargs="+", type=int, 
                        default=[6, 9, 12, 15, 18, 21, 24, 27, 30],
                        help="List of total epoch budgets to test")
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.0, 0.3, 0.5, 0.7],
                        help="List of alpha values (trend parameters)")
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--min-per-rank", type=int, default=1)
    parser.add_argument("--list-configs", action="store_true",
                        help="List all generated configurations and exit")
    parser.add_argument("--estimate-flops", action="store_true",
                        help="Estimate FLOPs for CIFAR-10 example")
    
    args = parser.parse_args()
    
    configs = generate_schedule_grid(
        epoch_budgets=args.epoch_budgets,
        alphas=args.alphas,
        rank=args.rank,
        min_per_rank=args.min_per_rank,
    )
    
    if args.list_configs:
        print(f"Generated {len(configs)} configurations:")
        for name, config in sorted(configs.items()):
            if config["type"] == "joint":
                print(f"  {name:<20} Joint LoRA r={config['rank']} × {config['epochs']} epochs")
            else:
                sched = config['schedule_str']
                alpha = config['alpha']
                trend = 'equal' if alpha == 0 else ('front' if alpha > 0 else 'back')
                print(f"  {name:<20} Sequential {sched} (α={alpha:.1f}, {trend}-loaded)")
        return
    
    if args.estimate_flops:
        from lora_lib.flops import vision_mlp_layers
        layers = vision_mlp_layers(3072, 10)  # CIFAR-10
        samples_per_epoch = 25000
        
        print("FLOPs estimates (GFLOPs) for CIFAR-10:")
        for name, config in sorted(configs.items(), key=lambda x: estimate_lora_only_flops(x[1], layers, samples_per_epoch)):
            flops_g = estimate_lora_only_flops(config, layers, samples_per_epoch) / 1e9
            print(f"  {name:<20} {flops_g:8.1f} GF")


if __name__ == "__main__":
    main()