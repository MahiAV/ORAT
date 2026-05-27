# One Rank at a Time: Code and Figure Reproduction

This repository contains the code and cached results for reproducing the experiments and figures in the TMLR paper "One Rank at a Time: Cascading Error Dynamics in Sequential Learning."

```
tmlr_deliverable/
├── code/                # All Python implementations.
├── scripts/             # One shell script per figure + a run-all driver.
├── cached_results/      # Vision + NLP results JSONs (pretrained models'
│                        # accuracies and per-sample FLOPs metadata).
└── figures/             # Output figures. Names match the LaTeX includegraphics.
```

## Running everything

```bash
cd tmlr_deliverable
bash scripts/run_all.sh
```

Wall-clock on a CPU box: ~15 minutes total. The synthetic experiments
(schedule sweep, noise sweep, iters-vs-threshold) are the slow steps;
everything that plots cached JSON results runs in seconds.

## Running one figure at a time

Each shell script reproduces one figure and is independent of the others:

| Script                                       | Figure (file in `figures/`)                                                                          | Source data                              |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `scripts/01_fig1_schedule_sweep.sh`          | `rebuttal_fig1_final.pdf` + `rebuttal_{reconstruction_smooth,cumulative_psi}_final_delta_vs_alpha.pdf` | Synthetic linear (W*∈R^{100×200}, r=20)  |
| `scripts/02_fig2_vision_barplots.sh`         | `rebuttal_fig2_updated_final.pdf`                                                                    | `cached_results/vision_seeds/`           |
| `scripts/03_fig5_cifar100_acc_vs_flops_scatter.sh` | `rebuttal_fig5_updated.png`                                                                    | `cached_results/efficient_ablation_cifar100_fixed/` |
| `scripts/04_cifar100_acc_vs_flops_more_less.sh`    | `rebuttal_accuracy_vs_flops_cifar100_more_less.{png,pdf}`                                      | `cached_results/efficient_ablation_cifar100_fixed/` |
| `scripts/05_cifar100_acc_vs_flops_final.sh`        | `rebuttal_accuracy_vs_flops_cifar100_final.{png,pdf}`                                          | `cached_results/efficient_ablation_cifar100_fixed/` |
| `scripts/06_sst2_barplot.sh`                       | `rebuttal_figure2_sst2_from_ablation.{png,pdf}`                                                | `cached_results/efficient_ablation_sst2_fixed/` |
| `scripts/07_sst2_acc_vs_flops_more_first.sh`       | `rebuttal_accuracy_vs_flops_sst2_more_first_alphas.{png,pdf}`                                  | `cached_results/efficient_ablation_sst2_fixed/` |
| `scripts/08_seq_compute_analysis.sh`               | `rebuttal_seq_compute_analysis.png`                                                            | `cached_results/vision_seeds/`           |
| `scripts/09_sv_profiles.sh`                        | `SVofW.png`, `SVofY.png`, `sv_{reconstruction,training}_error_comparison.png`                  | Synthetic linear (4 profiles)            |
| `scripts/10_noise_sweep.sh`                        | `reconstruction_error_noise_sweep.png`, `training_error_noise_sweep.png`,<br>`reconstruction_error_sparse_noise.png`, `training_error_sparse_noise.png`,<br>`reconstruction_vs_iters_kappa_{0.1,0.5,1,1.5}.png`,<br>`recon_svp_kappa_{0.05,0.1,0.5,1.0}.png` | Synthetic linear + noise |
| `scripts/11_iters_vs_threshold.sh`                 | `iters_vs_threshold_at{1,1.5,2,2.5}.png`                                                       | Synthetic linear, power-law W*           |

## Code layout

```
code/
├── synthetic_schedule_sweep.py      Fig 1 + delta-vs-alpha (synthetic linear)
├── synthetic_sv_profiles.py         SV-of-W/Y + SV reconstruction/training
├── synthetic_noise_sweep.py         All Gaussian/sparse noise + per-kappa figures
├── synthetic_iters_threshold.py     iters_vs_threshold_at{1,1.5,2,2.5}
├── vision_barplots.py               Vision r=1/2/3 barplots
├── sst2_barplot.py                  SST-2 sequential-vs-joint barplot
├── plot_schedule_ablation*.py       Accuracy-vs-FLOPs schedule comparison
├── plot_cifar100_flops_scatter_xkcd_matched.py    CIFAR-100 acc-vs-FLOPs scatter
├── seq_compute_analysis.py          Per-sample LoRA FLOPs barplot
├── schedule_ablation.py             Schedule + accuracy data loaders
├── figure_flops_comparison.py       FLOPs metadata helpers (shared)
└── lora_lib/                        Shared helpers: model defs, FLOPs estimator, plot style
```

## Cached results

Every figure that involves vision (MNIST / CIFAR-10 / CIFAR-100) or NLP
(SST-2) reads pre-computed JSON files in `cached_results/`. Re-training is
not part of this bundle. The
training scripts that produced these JSONs live in `../clean_code/` (see
`run_vision_experiment.py`, `run_nlp_experiment.py`, and
`run_efficient_ablation.py`).

```
cached_results/
├── vision_seeds/                       5 seeds × {MNIST, CIFAR-10, CIFAR-100}
└── efficient_ablation_{cifar100,sst2}_fixed/   alpha-sweep + joint LoRA points
```

## Requirements

- Python 3.8+
- numpy, scipy, matplotlib (for synthetic + figure rendering)

The synthetic figures need only numpy + scipy + matplotlib. The vision
barplot reuses `lora_lib` (no heavy ML deps for plotting — just for
the FLOPs estimator).

## Reproducibility notes

* The synthetic figures use fixed `numpy.random.RandomState` seeds.
  Re-running yields bit-identical PNGs/PDFs.
* The vision and NLP figures depend on the cached JSONs; if you regenerate
  those JSONs the bar values may shift by <0.5% (cuDNN nondeterminism,
  tokenizer details).
* The schedule-sweep iteration allocation is `t_k = 1 + α·x_k` where
  `x_k ∈ [+1,−1]` is the per-component centered position; setting `α = 0`
  recovers the equal schedule, `α > 0` is "more first", `α < 0` is
  "less first".
