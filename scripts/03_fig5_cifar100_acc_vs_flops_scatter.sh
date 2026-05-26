#!/usr/bin/env bash
# Reproduce rebuttal_fig5_updated.png  (CIFAR-100 accuracy vs adaptation FLOPs
# scatter: sequential r=1/2/3 + joint LoRA r=3, smoothed fits).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 code/plot_cifar100_flops_scatter_xkcd_matched.py \
    --results-dir cached_results/efficient_ablation_cifar100_fixed \
    -o figures/rebuttal_fig5_updated.png

echo "Wrote: figures/rebuttal_fig5_updated.png"
