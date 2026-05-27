<p align="center">
  <h1 align="center">One Rank at a Time (ORAT)</h1>
  <p align="center"><strong>Cascading Error Dynamics in Sequential Learning</strong></p>
  <p align="center">
    <a href="https://arxiv.org/abs/2505.22602"><img src="https://img.shields.io/badge/arXiv-2505.22602-b31b1b.svg"></a>
    <a href="https://akyrillidis.github.io/aiowls/one_rank_at_a_time.html"><img src="https://img.shields.io/badge/Blog-AI--OWLS-FFD400.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/TMLR-2026-blue.svg"></a>
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

The sequential procedure we analyse. Given $\mathbf{X} \in \mathbb{R}^{d \times n}$ and $\mathbf{Y} \in \mathbb{R}^{m \times n}$, find low-rank $\mathbf{W} = \mathbf{B}\mathbf{A}$ of rank $r \ll \min(m, d)$ such that $\mathbf{Y} \approx \mathbf{W}\mathbf{X}$, solved **sequentially**:

```math
(\mathbf{a}_k, \mathbf{b}_k) = \arg\min_{\mathbf{a}, \mathbf{b}} \tfrac{1}{2}\, \Vert \mathbf{Y}_k - \mathbf{b}\, \mathbf{a}^\top \mathbf{X} \Vert_F^2 , \qquad \mathbf{Y}_{k+1} \leftarrow \mathbf{Y}_k - \mathbf{b}_k\, \mathbf{a}_k^\top \mathbf{X}.
```

<p align="center">
  <img src="https://akyrillidis.github.io/aiowls/assets/img/one_rank/rebuttal_fig1_final-1.png" width="100%">
</p>
<p align="center"><em>Figure 1. Three schedules under fixed budget T = 500, rank r = 20. Left: reconstruction error. Middle: training objective. Right: cumulative numerical-error proxy. The more-first schedule (α &gt; 0) wins on every panel.</em></p>

---

## ✨ Key Features

- **Closed-form cascade bound** — Theorem 1 gives an explicit upper bound on residual training error (see the equation below the bullet list). Each per-step numerical error $\boldsymbol{\Psi}_k$ is amplified by a product of factors $\rho_j = 2 + 6\sigma_j^\star / \mathcal{T}_j^\star$, where $\sigma_j^\star$ and $\mathcal{T}_j^\star = \sigma_j^\star - \sigma_{j+1}^\star$ are the singular values and gaps of the output matrix $\mathbf{Y}$.
- **Parameter recovery, noiseless + noisy** — Theorems 2 and 3 extend the bound to true-parameter recovery, with a clean bias-variance trade-off in the truncation rank $r$ under Gaussian label noise.
- **Practical compute prescription** — A one-parameter $\alpha$-family of schedules $t_k(\alpha) = 1 + \alpha \cdot x_k$ (where $x_k \in [+1, -1]$ is the centred position) makes the "more-first" intuition quantitative; optimal $\alpha$ saturates near $1.5$.
- **Cross-domain validation** — Synthetic linear-regression experiments match theory tightly; LoRA on MNIST/CIFAR10/CIFAR100 (vision) and DistilBERT/SST-2 (language) probes show the qualitative pattern transfers.
- **Honest scope** — explicitly *not* a benchmark-beating method; the contribution is **explanatory**. The deep-learning experiments are exploratory probes outside the linear theorems.

The cascade bound (Theorem 1):

```math
\Big\Vert \mathbf{Y} - \sum_{k=1}^{r} \mathbf{b}_k \mathbf{a}_k^\top \mathbf{X} \Big\Vert_F \;\le\; \underbrace{\Big(\sum_{k \gt r}(\sigma_k^\star)^2\Big)^{1/2}}_{\text{truncation tail}} \;+\; \underbrace{\sum_{k=1}^{r}\Big(\prod_{j \lt k} \rho_j\Big)\, \Vert \boldsymbol{\Psi}_k \Vert_F}_{\text{cascade amplification}}
```

---

## 📐 Theoretical Guarantees

| # | Theorem | What it bounds | Assumptions |
|---|---------|----------------|-------------|
| 1 | **Training-error propagation** | $\Vert \mathbf{Y} - \sum_k \mathbf{b}_k \mathbf{a}_k^\top \mathbf{X} \Vert_F \le$ truncation tail + cascade-amplified $\sum_k \Vert \boldsymbol{\Psi}_k \Vert_F$ | Strict singular gaps of $\mathbf{Y}$; cumulative error in perturbation regime |
| 2 | **Parameter recovery (noiseless)** | $\Vert \widehat{\mathbf{W}} - \mathbf{W}^\star \Vert_F$ in terms of per-step errors and $\kappa(\mathbf{X})$ | Same as T1 plus $\sigma_{\min}(\mathbf{X}) > 0$ |
| 3 | **Parameter recovery (Gaussian noise)** | Bias–variance: cascade term + $\varepsilon \sqrt{n \log(1/\gamma)}$ noise term | Same plus a small-noise ordering condition |

Full statements and proofs in Sections 3–4 of the paper. Plain-English summaries in [the blog](https://akyrillidis.github.io/aiowls/one_rank_at_a_time.html).

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

The synthetic experiments (schedule sweep, noise sweep, iters-vs-threshold) are the slow steps; everything that plots cached JSON results runs in seconds.

### Reproduce one figure at a time

Each shell script is independent:

```bash
# Figure 1 (synthetic) — the cascade in action
bash scripts/01_fig1_schedule_sweep.sh

# Figures 4–5 (vision LoRA on CIFAR-100)
bash scripts/03_fig5_cifar100_acc_vs_flops_scatter.sh

# Figures 6–7 (SST-2 sequential LoRA)
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
└── figures/                         # Output PDFs / PNGs
```

---

## 📊 Results Highlights

### Synthetic linear regression (Figure 1, Section 5.1)

Three schedules under fixed budget $T = 500$, rank $r = 20$:

| Schedule | $\alpha$ | Reconstruction error ↓ | Training loss ↓ | Cumulative error proxy ↓ |
|----------|---------:|-----------------------:|---------------:|-------------------------:|
| less-first | $-1$ | catastrophic | catastrophic | catastrophic |
| equal | $0$ | baseline | baseline | baseline |
| **more-first** | $\mathbf{+1.5}$ | **best** | **best** | **best** |

Gains saturate around $\alpha \approx 1.5$; pushing further yields diminishing returns.

### Vision LoRA — feedforward net (Section 5.2)

Sequential rank-3 LoRA lands in the same accuracy band as jointly trained rank-3 LoRA on MNIST / CIFAR-10 / CIFAR-100, within 5% of total FLOPs. Front-loaded compute schedules dominate the accuracy-vs-FLOPs Pareto frontier.

### Language LoRA — DistilBERT / SST-2 (Section 5.3)

<p align="center">
  <img src="https://akyrillidis.github.io/aiowls/assets/img/one_rank/rebuttal_figure2_sst2_from_ablation-1.png" width="540">
</p>

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
- The vision and NLP figures depend on the cached JSONs; if you regenerate those JSONs the bar values may shift by under 0.5% (cuDNN non-determinism, tokenizer details).
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

- 🔗 [**Blog post**](https://akyrillidis.github.io/aiowls/one_rank_at_a_time.html) — sharp, accessible walk-through with all the figures.
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
