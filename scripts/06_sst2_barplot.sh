#!/usr/bin/env bash
# Reproduce rebuttal_figure2_sst2_from_ablation.{png,pdf}
# SST-2 barplot picking the FLOPs-matched sequential schedule that beats joint LoRA.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 code/sst2_barplot.py \
    --results-dir cached_results/efficient_ablation_sst2_fixed \
    --output figures/rebuttal_figure2_sst2_from_ablation.png \
    --title ""

echo "Wrote: figures/rebuttal_figure2_sst2_from_ablation.{png,pdf}"
