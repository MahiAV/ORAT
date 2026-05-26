#!/usr/bin/env bash
# Reproduce rebuttal_fig1_final.pdf and the two delta-vs-alpha plots.
#
# Produces (under figures/):
#   rebuttal_fig1_final.{png,pdf}
#   rebuttal_reconstruction_smooth_final_delta_vs_alpha.{png,pdf}
#   rebuttal_cumulative_psi_final_delta_vs_alpha.{png,pdf}

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="figures/synthetic_schedule_sweep_out"
mkdir -p "$OUT_DIR" figures

python3 code/synthetic_schedule_sweep.py \
    --out-root "$OUT_DIR" \
    --alpha-num 121 \
    --three-panel-alpha 1.0

# Copy / rename canonical outputs to the deliverable figures directory.
cp "$OUT_DIR/repo_setting/repo_setting_combined_three_panel_all_real.pdf" \
   figures/rebuttal_fig1_final.pdf
cp "$OUT_DIR/repo_setting/repo_setting_combined_three_panel_all_real.png" \
   figures/rebuttal_fig1_final.png

cp "$OUT_DIR/repo_setting/repo_setting_reconstruction_smooth_final_delta_vs_alpha.pdf" \
   figures/rebuttal_reconstruction_smooth_final_delta_vs_alpha.pdf
cp "$OUT_DIR/repo_setting/repo_setting_reconstruction_smooth_final_delta_vs_alpha.png" \
   figures/rebuttal_reconstruction_smooth_final_delta_vs_alpha.png

cp "$OUT_DIR/repo_setting/repo_setting_cumulative_psi_final_delta_vs_alpha.pdf" \
   figures/rebuttal_cumulative_psi_final_delta_vs_alpha.pdf
cp "$OUT_DIR/repo_setting/repo_setting_cumulative_psi_final_delta_vs_alpha.png" \
   figures/rebuttal_cumulative_psi_final_delta_vs_alpha.png

echo "Wrote: figures/rebuttal_fig1_final.{png,pdf}"
echo "       figures/rebuttal_reconstruction_smooth_final_delta_vs_alpha.{png,pdf}"
echo "       figures/rebuttal_cumulative_psi_final_delta_vs_alpha.{png,pdf}"
