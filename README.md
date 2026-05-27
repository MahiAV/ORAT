<p align="center">
  <h1 align="center">One Rank at a Time (ORAT)</h1>
  <p align="center"><strong>Cascading Error Dynamics in Sequential Learning</strong></p>
  <p align="center"><em>Code and figure-reproduction bundle for the TMLR-accepted paper.</em></p>
  <p align="center">
    <a href="https://arxiv.org/abs/2505.22602"><img src="https://img.shields.io/badge/arXiv-2505.22602-b31b1b.svg"></a>
    <a href="https://openreview.net/forum?id=TBD"><img src="https://img.shields.io/badge/TMLR-2026-blue.svg"></a>
    <a href="https://akyrillidis.github.io/aiowls/one_rank_at_a_time.html"><img src="https://img.shields.io/badge/Blog-AI--OWLS-FFD400.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white"></a>
    <a href="#"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  </p>
</p>

---

> **TL;DR** — When low-rank models are fit one rank-1 component at a time, per-step
> numerical errors amplify *geometrically* through every later step. The amplification
> factor is governed by the **spectral gaps of the output matrix** $\mathbf{Y}$. We prove
> this for fixed-design linear regression and empirically validate the same pattern in
> LoRA fine-tuning of vision (MNIST / CIFAR-10 / CIFAR-100) and language (DistilBERT /
> SST-2) models.

The sequential procedure we analyse:

<p align="center">
  Given <strong>X</strong> ∈ ℝ<sup>d×n</sup>, <strong>Y</strong> ∈ ℝ<sup>m×n</sup>, find low-rank <strong>W</strong> = <strong>BA</strong>,
  rank <em>r</em> ≪ min(m, d), such that <strong>Y</strong> ≈ <strong>WX</strong>, <strong>solved sequentially</strong>:<br>
  (<strong>a</strong><sub>k</sub>, <strong>b</strong><sub>k</sub>) = argmin ‖<strong>Y</strong><sub>k</sub> − <strong>b</strong> <strong>a</strong><sup>⊤</sup><strong>X</strong>‖² ,&nbsp; then <strong>Y</strong><sub>k+1</sub> ← <strong>Y</strong><sub>k</sub> − <strong>b</strong><sub>k</sub><strong>a</strong><sub>k</sub><sup>⊤</sup><strong>X</strong>.
</p>

<p align="center">
  <img src="figures/rebuttal_fig1_final.png" width="100%">
</p>
<p align="center"><em>Figure 1. Three schedules under fixed budget T=500, rank r=20.</em>
<em>Left: reconstruction error. Middle: training objective. Right: cumulative numerical-error proxy.</em>
<em>The more-first schedule (α &gt; 0) wins on every panel.</em></p>

---

## ✨ Key Features

- **Closed-form cascade bound** — Theorem 1 gives an explicit upper bound on residual training error as $(\text{truncation tail}) + \sum_k (\prod_{j<k} \rho_j) \|\Psi_k\|_F$ where $\rho_j = 2 + 6\sigma_j^\star / \mathcal{T}_j^\star$ measures the spectral-gap amplification (singular values and gaps of $\mathbf{Y}$).
- **Parameter recovery, noiseless + noisy** — Theorems 2 and 3 extend the bound to true-parameter recovery, with a clean bias-variance trade-off in the truncation rank $r$ under Gaussian label noise.
- **Practical compute prescription** — A one-parameter α-family of schedules `t_k(α) = 1 + α·x_k` makes the "more-first" intuition quantitative; optimal α saturates near 1.5.
- **Cross-domain validation** — Synthetic linear-regression experiments match theory tightly; LoRA on MNIST/CIFAR10/CIFAR100 (vision) and DistilBERT/SST-2 (language) probes show the qualitative pattern transfers.
- **Honest scope** — explicitly *not* a benchmark-beating method; the contribution is **explanatory**. The deep-learning experiments are exploratory probes outside the linear theorems.

---

## 📐 Theoretical Guarantees

| # | Theorem | What it bounds | Assumptions |
|---|---------|----------------|-------------|
| 1 | **Training-error propagation** | `‖Y − Σ_k b_k a_k^⊤ X‖_F ≤ trunc-tail + Σ_k (Π_j<k ρ_j)‖Ψ_k‖_F` | Strict singular gaps of $\mathbf{Y}$; cumulative error in perturbation regime |
| 2 | **Parameter recovery (noiseless)** | `‖Ŵ − W★‖_F` in terms of per-step errors and κ(X) | Same as T1 + σ_min(X) > 0 |
| 3 | **Parameter recovery (Gaussian noise)** | Bias–variance: cascade term + ε √(n log(1/γ)) noise term | Same + small-noise ordering condition |

Full statements + proofs in Sections 3–4 of the paper. Plain-English summaries in [the blog](https://akyrillidis.github.io/aiowls/one_rank_at_a_time.html).

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy scipy matplotlib
```

CPU-only is fine. No GPU required.

### Reproduce everything in one shot

```bash
bash scripts/run_all.sh
```

Wall-clock on a 2024 laptop: ~15 minutes total. The synthetic experiments (schedule sweep, noise sweep, iters-vs-threshold) are the slow steps; everything that plots cached JSON results runs in seconds.

### Reproduce one figure at a time

Each shell script is independent:

```bash
# Figure 1 (synthetic) — the cascade in action
bash scripts/01_fig1_schedule_sweep.sh

# Figures 4-5 (vision LoRA on CIFAR-100)
bash scripts/03_fig5_cifar100_acc_vs_flops_scatter.sh

# Figures 6-7 (SST-2 sequential LoRA)
bash scripts/06_sst2_barplot.sh
bash scripts/07_sst2_acc_vs_flops_more_first.sh
```

---

## 📂 Repository Structure

```
.
├── code/                            # Python implementations
│   ├── synthetic_schedule_sweep.py    # Figure 1 + Δ-vs-α (synthetic)
│   ├── synthetic_sv_profiles.py       # Singular-value profiles + recovery
│   ├── synthetic_noise_sweep.py       # Gaussian + sparse noise sweeps
│   ├── synthetic_iters_threshold.py   # Iter-budget threshold figures
│   ├── vision_barplots.py             # MNIST / CIFAR-10 / CIFAR-100 bars
│   ├── sst2_barplot.py                # SST-2 sequential-vs-joint barplot
│   ├── plot_schedule_ablation_*.py    # Accuracy-vs-FLOPs comparisons
│   ├── seq_compute_analysis.py        # Per-sample LoRA FLOPs barplot
│   └── lora_lib/                      # Shared helpers: model defs, FLOPs, plot style
├── scripts/                         # One shell script per figure
├── cached_results/                  # Pre-computed JSON outputs
│   ├── vision_seeds/                  # 5 seeds × {MNIST, CIFAR-10, CIFAR-100}
│   └── efficient_ablation_{cifar100,sst2}_fixed/
└── figures/                         # Output PDFs / PNGs (paths match the LaTeX includegraphics)
```

---

## 💻 Hardware

CPU-only. No GPU required. The synthetic experiments are pure NumPy / SciPy / matplotlib; the vision/NLP figures replay cached JSONs.

| Experiment | Wall-clock |
|------------|------------|
| Figure 1 (synthetic schedule sweep, fine α-grid) | ~5 min |
| Figures 2–8 (cached JSON replays) | <30 s each |
| Figure 9 (singular-value profiles) | ~2 min |
| Figure 10 (noise sweep + per-κ ablations) | ~5 min |
| Figure 11 (iters-vs-threshold) | ~3 min |

---

## 📊 Results Highlights

### Synthetic linear regression (Figure 1, Section 5.1)

Three schedules under fixed budget T=500, rank r=20:

| Schedule | α | Reconstruction error ↓ | Training loss ↓ | Cumulative error proxy ↓ |
|----------|------|-----------------------:|---------------:|--------------------------:|
| less-first | −1 | catastrophic | catastrophic | catastrophic |
| equal | 0 | baseline | baseline | baseline |
| **more-first** | **+1.5** | **best** | **best** | **best** |

Gains saturate around α ≈ 1.5; pushing further yields diminishing returns.

### Vision LoRA — feedforward net (Section 5.2)

Sequential rank-3 LoRA lands in the same accuracy band as jointly trained rank-3 LoRA on MNIST / CIFAR-10 / CIFAR-100, within 5% of total FLOPs. Front-loaded compute schedules dominate the accuracy-vs-FLOPs Pareto frontier.

### Language LoRA — DistilBERT / SST-2 (Section 5.3)

| Method | SST-2 Accuracy |
|--------|---------------:|
| Joint LoRA (reference) | 0.872 |
| Sequential rank-1 (ours) | **0.872** |
| Sequential rank-2 (ours) | 0.875 |
| Sequential rank-3 (ours) | 0.876 |

The **first sequential rank-1 component alone** matches the jointly trained rank-3 LoRA reference.

---

## 🔬 Cached results format

Every figure that involves vision or NLP reads pre-computed JSON files in `cached_results/`. Re-training is not part of this bundle. The training scripts that produced these JSONs are referenced in the paper's appendix.

```
cached_results/
├── vision_seeds/                                # 5 seeds × {MNIST, CIFAR-10, CIFAR-100}
└── efficient_ablation_{cifar100,sst2}_fixed/    # α-sweep + joint LoRA points
```

---

## 🧪 Reproducibility notes

- The synthetic figures use fixed `numpy.random.RandomState` seeds. Re-running yields bit-identical PNGs/PDFs.
- The vision and NLP figures depend on the cached JSONs; if you regenerate those JSONs the bar values may shift by <0.5% (cuDNN non-determinism, tokenizer details).
- The schedule-sweep iteration allocation is $t_k = 1 + \alpha \cdot x_k$ where $x_k \in [+1, -1]$ is the per-component centred position; setting $\alpha = 0$ recovers the equal schedule, $\alpha > 0$ is "more first", $\alpha < 0$ is "less first".

---

## 📖 Citation

```bibtex
@article{vandchali2026onerank,
  title   = {One Rank at a Time: Cascading Error Dynamics in Sequential Learning},
  author  = {Vandchali, Mahtab Alizadeh and Liao, Fangshuo and Kyrillidis, Anastasios},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year    = {2026},
  note    = {Accepted, in press. arXiv:2505.22602}
}
```

---

## 📝 Blog and Companion Work

- 🔗 [**Blog post**](https://akyrillidis.github.io/aiowls/one_rank_at_a_time.html) — sharp, accessible walk-through with all the figures and ACSP non-claims.
- 🔗 [**Companion paper — AdaPaD**](https://akyrillidis.github.io/aiowls/adapad.html): the *parallel* deflation analog, where the same per-step errors *self-correct* across rounds rather than compounding. Read the two together for the sequential / parallel trade-off.

---

## 👥 Authors

- **Mahtab Alizadeh Vandchali** — Rice University (CS)
- **Fangshuo (Jasper) Liao** — Rice University (CS) — [website](https://jasperliao.github.io/)
- **Anastasios Kyrillidis** — Rice University (CS + ECE) — [website](https://akyrillidis.github.io/)

Funding: NSF CAREER 2145629 · Rice K2I.

---

## 📄 License

MIT.
