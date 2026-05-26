#!/usr/bin/env bash
# Reproduce rebuttal_accuracy_vs_flops_cifar100_final.{png,pdf}
# Joint LoRA + Equal + top-2 More-first/Less-first schedules (per-alpha winners).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 code/plot_schedule_ablation_final.py \
    --results-dir cached_results/efficient_ablation_cifar100_fixed \
    --output figures/rebuttal_accuracy_vs_flops_cifar100_final.png \
    --title ""

echo "Wrote: figures/rebuttal_accuracy_vs_flops_cifar100_final.{png,pdf}"
