#!/usr/bin/env bash
# Reproduce rebuttal_accuracy_vs_flops_cifar100_more_less.{png,pdf}
# More-first vs Less-first vs Equal schedules, FLOPs-matched against equal baseline.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 code/plot_schedule_ablation_more_less.py \
    --results-dir cached_results/efficient_ablation_cifar100_fixed \
    --output figures/rebuttal_accuracy_vs_flops_cifar100_more_less.png \
    --title ""

echo "Wrote: figures/rebuttal_accuracy_vs_flops_cifar100_more_less.{png,pdf}"
