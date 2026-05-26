#!/usr/bin/env bash
# Reproduce singular-value-profile appendix figures:
#   SVofW.png, SVofY.png, sv_reconstruction_error_comparison.png, sv_training_error_comparison.png

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 code/synthetic_sv_profiles.py --output-dir figures

echo "Wrote: figures/SVofW.png, figures/SVofY.png,"
echo "       figures/sv_reconstruction_error_comparison.png, figures/sv_training_error_comparison.png"
