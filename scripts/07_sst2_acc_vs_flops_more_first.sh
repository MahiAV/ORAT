#!/usr/bin/env bash
# Reproduce rebuttal_accuracy_vs_flops_sst2_more_first_alphas.{png,pdf}
# SST-2: Joint LoRA + Equal + every positive-alpha "more-first" schedule.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 code/plot_schedule_ablation_more_only.py \
    --results-dir cached_results/efficient_ablation_sst2_fixed \
    --output figures/rebuttal_accuracy_vs_flops_sst2_more_first_alphas.png \
    --title "" \
    --alphas all \
    --compact-legend \
    --drop-joint-tail 1

echo "Wrote: figures/rebuttal_accuracy_vs_flops_sst2_more_first_alphas.{png,pdf}"
