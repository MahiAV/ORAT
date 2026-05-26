#!/usr/bin/env bash
# Reproduce rebuttal_seq_compute_analysis.png
# Per-sample LoRA-only FLOPs barplot: joint vs sequential rank-1 (CIFAR-100 MLP).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 code/seq_compute_analysis.py \
    --results cached_results/vision_seeds/vision_cifar100_equiv10ep_J10_E6-6-6_seed42.json \
    --output figures/rebuttal_seq_compute_analysis.png \
    --max-rank 6

echo "Wrote: figures/rebuttal_seq_compute_analysis.png"
