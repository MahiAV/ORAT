#!/usr/bin/env bash
# Reproduce rebuttal_fig2_updated_final.{png,pdf}
# Vision barplots: r=1, r=2, r=3 sequential vs LoRA-r=3 joint, mean ± SD over 5 seeds.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED_DIR="cached_results/vision_seeds"

for ext in png pdf; do
    python3 code/vision_barplots.py \
        --mnist-multi   "$SEED_DIR"/vision_mnist_equiv10ep_J10_E6-6-6_seed4*.json \
        --cifar10-multi "$SEED_DIR"/vision_cifar10_equiv10ep_J10_E6-6-6_seed4*.json \
        --cifar100-multi "$SEED_DIR"/vision_cifar100_equiv10ep_J10_E6-6-6_seed4*.json \
        --output figures/rebuttal_fig2_updated_final.${ext}
done

echo "Wrote: figures/rebuttal_fig2_updated_final.{png,pdf}"
