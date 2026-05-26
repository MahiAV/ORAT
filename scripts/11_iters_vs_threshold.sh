#!/usr/bin/env bash
# Reproduce iters_vs_threshold_at{1,1.5,2,2.5}.png
# Power-law W*, kappa=0, schedule comparison: less-first / more-first / equal.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 code/synthetic_iters_threshold.py --output-dir figures

echo "Wrote: figures/iters_vs_threshold_at{1,1.5,2,2.5}.png"
