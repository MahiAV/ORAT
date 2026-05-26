#!/usr/bin/env bash
# Reproduce noise-sweep + per-kappa appendix figures:
#   reconstruction_error_noise_sweep.png, training_error_noise_sweep.png
#   reconstruction_error_sparse_noise.png, training_error_sparse_noise.png
#   reconstruction_vs_iters_kappa_{0.1,0.5,1,1.5}.png
#   recon_svp_kappa_{0.05,0.1,0.5,1.0}.png

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 code/synthetic_noise_sweep.py --output-dir figures

echo "Wrote: figures/reconstruction_error_noise_sweep.png"
echo "       figures/training_error_noise_sweep.png"
echo "       figures/reconstruction_error_sparse_noise.png"
echo "       figures/training_error_sparse_noise.png"
echo "       figures/reconstruction_vs_iters_kappa_*.png  (4 files)"
echo "       figures/recon_svp_kappa_*.png                (4 files)"
