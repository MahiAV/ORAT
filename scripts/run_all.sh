#!/usr/bin/env bash
# Regenerate every rebuttal figure into ../figures/.
#
# Each step is independent. To rerun just one figure, invoke the matching
# script in this directory directly:
#
#   bash scripts/01_fig1_schedule_sweep.sh
#   bash scripts/02_fig2_vision_barplots.sh
#   ...

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p figures

echo "[1/11] rebuttal_fig1_final + delta-vs-alpha"
bash scripts/01_fig1_schedule_sweep.sh

echo "[2/11] rebuttal_fig2_updated_final (vision barplots)"
bash scripts/02_fig2_vision_barplots.sh

echo "[3/11] rebuttal_fig5_updated (CIFAR-100 acc vs FLOPs scatter)"
bash scripts/03_fig5_cifar100_acc_vs_flops_scatter.sh

echo "[4/11] rebuttal_accuracy_vs_flops_cifar100_more_less"
bash scripts/04_cifar100_acc_vs_flops_more_less.sh

echo "[5/11] rebuttal_accuracy_vs_flops_cifar100_final"
bash scripts/05_cifar100_acc_vs_flops_final.sh

echo "[6/11] rebuttal_figure2_sst2_from_ablation"
bash scripts/06_sst2_barplot.sh

echo "[7/11] rebuttal_accuracy_vs_flops_sst2_more_first_alphas"
bash scripts/07_sst2_acc_vs_flops_more_first.sh

echo "[8/11] rebuttal_seq_compute_analysis (per-sample FLOPs barplot)"
bash scripts/08_seq_compute_analysis.sh

echo "[9/11] singular-value-profile figures (SVofW, SVofY, sv_*_comparison)"
bash scripts/09_sv_profiles.sh

echo "[10/11] noise-sweep + per-kappa figures"
bash scripts/10_noise_sweep.sh

echo "[11/11] iters_vs_threshold figures"
bash scripts/11_iters_vs_threshold.sh

echo
echo "Done. All figures are in $ROOT/figures/."
