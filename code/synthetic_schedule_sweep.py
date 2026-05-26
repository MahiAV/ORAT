#!/usr/bin/env python3
from __future__ import annotations

"""
Enhanced Figure 1 generator built on the official GitHub notebook logic
(`Iteration_allocation_strategy.ipynb`).

What this script does
---------------------
For each requested setting (repo / paper / both), it generates separate files for:

1) Reconstruction error with notebook logging (staircase-like)
   - linear y-scale
   - log y-scale

2) Reconstruction error with smooth logging (current component included)
   - linear y-scale
   - log y-scale

3) Training objective error
   - linear y-scale
   - log y-scale

4) Accumulated optimization error from Definition 1 / Theorems 1 and 5
   - linear y-scale
   - log y-scale

It also:
- highlights the first-component iteration region for each schedule,
- adds compact schedule formulas to the legend,
- saves every plot as separate PNG and PDF files,
- saves the averaged curves as CSV files,
- writes a zip archive containing the full bundle.

Important notes
---------------
Repo setting:
    This follows the official notebook exactly:
      m=100, d=200, n=500, r=20,
      trials=3,
      X columns normalized,
      singular values = geomspace(100, 1, 20),
      eta_A = eta_B = 0.003,
      total_iters = 500,
      epsilon = 1e-10.

Paper setting:
    The paper text changes the experimental setup to:
      W* in R^{500 x 1000},
      trials=5,
      X sampled entrywise from N(0,1),
      rank r=20.
    The paper text does not explicitly state n for Figure 1, so this script uses
    --paper-n (default 2500). The schedule formulas remain the official notebook
    formulas, because the paper describes them qualitatively (equal / more-first /
    less-first) but does not specify a different exact allocation formula.

Cumulative Psi plot:
    Definition 1 defines Psi_k as the difference between the stage-k subroutine-optimal
    rank-1 matrix and the finite-iteration matrix returned by gradient descent.
    To visualize its *progress* over cumulative iterations, this script plots:

        sum_{j < k} ||Psi_j||_F + ||Psi_k^(t)||_F

    while the inner loop for component k is running. This is the natural iterative
    counterpart of the cumulative sum appearing in Theorems 1 and 5.

Usage
-----
python figure1_official_notebook_bundle_exact_plus.py \
    --out-root figure1_plot_bundle \
    --zip-path figure1_plot_bundle.zip \
    --settings both
"""

import argparse
import json
import math
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import numpy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12.5,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 8.0,
    "legend.title_fontsize": 8.2,
    "lines.linewidth": 2.0,
    "axes.linewidth": 0.8,
    "figure.dpi": 220,
    "savefig.dpi": 220,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

METHODS = ["early_heavy", "uniform", "late_heavy"]
PLOT_ORDER = ["late_heavy", "uniform", "early_heavy"]
COLORS = {
    "early_heavy": "#ff7f0e",
    "uniform": "#2ca02c",
    "late_heavy": "#1f77b4",
}
DISPLAY = {
    "early_heavy": "More First",
    "uniform": "Uniform",
    "late_heavy": "Less First",
}


@dataclass(frozen=True)
class Setting:
    slug: str
    title: str
    m: int
    d: int
    n: int
    r: int
    eta_a: float
    eta_b: float
    total_iters: int
    epsilon: float
    num_trials: int
    x_mode: str          # repo_column_normalized or paper_gaussian
    sigma_profile: str   # exponential, uniform, power_law
    seed: int
    noise_std: float
    note: str


def build_settings(paper_n: int, paper_profile: str, which: str, total_iters: int = 500) -> List[Setting]:
    repo = Setting(
        slug="repo_setting",
        title="Repo setting (exact notebook)",
        m=100,
        d=200,
        n=500,
        r=20,
        eta_a=0.003,
        eta_b=0.003,
        total_iters=total_iters,
        epsilon=1e-10,
        num_trials=3,
        x_mode="repo_column_normalized",
        sigma_profile="exponential",
        seed=42,
        noise_std=0.0,
        note=(
            "Exact notebook setting: m=100, d=200, n=500, r=20, trials=3, "
            "X columns normalized, exponential singular values."
        ),
    )

    paper = Setting(
        slug="paper_setting",
        title="Paper setting",
        m=500,
        d=1000,
        n=paper_n,
        r=20,
        eta_a=0.003,
        eta_b=0.003,
        total_iters=total_iters,
        epsilon=1e-10,
        num_trials=5,
        x_mode="paper_gaussian",
        sigma_profile=paper_profile,
        seed=42,
        noise_std=0.0,
        note=(
            f"Paper-style setting: W* in R^(500x1000), trials=5, X~N(0,1), r=20, "
            f"profile={paper_profile}, n={paper_n}. The paper text does not explicitly state n for Figure 1."
        ),
    )

    if which == "repo":
        return [repo]
    if which == "paper":
        return [paper]
    if which == "both":
        return [repo, paper]
    raise ValueError(f"Unknown setting choice: {which}")


def allocate_iterations_alpha(r: int, total_iters: int, alpha: float, min_per_rank: int = 2) -> np.ndarray:
    """
    Simple score-and-normalize schedule family with linear trend control.

    Let k = 1,...,r and define a centered position x_k in [1, -1] from early to late.
    We assign scores linearly as
        s_k = 1 + alpha * x_k,
    then normalize scores to allocate the extra iteration budget.

    - alpha = 0  -> uniform allocation
    - alpha > 0  -> more iterations on earlier ranks (roughly linear decay)
    - alpha < 0  -> more iterations on later ranks (roughly linear increase)

    This keeps the interpretation simple and makes trends closer to the original
    notebook's linear schedule shape.
    """
    if r <= 0:
        raise ValueError("r must be positive")
    if min_per_rank < 0:
        raise ValueError("min_per_rank must be non-negative")
    base = r * min_per_rank
    if total_iters < base:
        raise ValueError(
            f"total_iters={total_iters} is too small for r={r} with min_per_rank={min_per_rank}"
        )

    extra_budget = total_iters - base
    if extra_budget == 0:
        return np.full(r, min_per_rank, dtype=int)

    x = np.linspace(1.0, -1.0, r, dtype=float)
    scores = 1.0 + alpha * x
    # Keep scores strictly positive so normalization is well-defined.
    scores = np.maximum(scores, 1e-9)
    weights = scores
    weights /= float(weights.sum())

    extra_real = extra_budget * weights
    extra_int = np.floor(extra_real).astype(int)
    remainder = int(extra_budget - int(extra_int.sum()))
    if remainder > 0:
        frac = extra_real - extra_int
        top = np.argsort(-frac)[:remainder]
        extra_int[top] += 1

    schedule = min_per_rank + extra_int
    return schedule.astype(int)


def schedule_formula_text(method: str, alpha: float) -> str:
    if method == "early_heavy":
        return rf"More First: $t_k>t_{{k+1}}$ ($\alpha={alpha:g}$)"
    if method == "uniform":
        return rf"Equal: $t_k=25$ ($\alpha={alpha:g}$)"
    if method == "late_heavy":
        return rf"Less First: $t_k<t_{{k+1}}$ ($\alpha={alpha:g}$)"
    raise ValueError(f"Unknown method: {method}")


def build_x(rs: np.random.RandomState, setting: Setting) -> np.ndarray:
    X = rs.randn(setting.d, setting.n)
    if setting.x_mode == "repo_column_normalized":
        X = X / la.norm(X, axis=0, keepdims=True)
    elif setting.x_mode == "paper_gaussian":
        pass
    else:
        raise ValueError(f"Unknown X mode: {setting.x_mode}")
    return X


def generate_w_star(rs: np.random.RandomState, m: int, d: int, r: int, profile: str) -> np.ndarray:
    U, _ = la.qr(rs.randn(m, r))
    V, _ = la.qr(rs.randn(d, r))

    if profile == "uniform":
        S = np.ones(r) * 10.0
    elif profile == "exponential":
        S = np.geomspace(100.0, 1.0, num=r)
    elif profile == "power_law":
        k = np.arange(1, r + 1, dtype=float)
        S = 100.0 * k ** -2.0
    else:
        raise ValueError(f"Unknown singular-value profile: {profile}")

    return U @ np.diag(S) @ V.T


def build_problem(setting: Setting, alpha_by_method: Dict[str, float]) -> Dict[str, object]:
    """
    Build the problem using the same RNG flow as the notebook's global np.random.seed(42)
    followed by X generation and W* generation.
    """
    rs = np.random.RandomState(setting.seed)
    X = build_x(rs, setting)
    W_star = generate_w_star(rs, setting.m, setting.d, setting.r, setting.sigma_profile)
    Y_clean = W_star @ X

    C = X @ X.T
    evals, evecs = la.eigh(C)
    tol = max(float(evals.max()), 1.0) * 1e-12
    evals = np.clip(evals, tol, None)
    C_inv_sqrt = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T

    schedules = {
        m: allocate_iterations_alpha(setting.r, setting.total_iters, float(alpha_by_method[m]))
        for m in METHODS
    }

    return {
        "setting": setting,
        "X": X,
        "W_star": W_star,
        "Y_clean": Y_clean,
        "C": C,
        "C_inv_sqrt": C_inv_sqrt,
        "w_norm": float(la.norm(W_star, ord="fro")),
        "y_norm": float(la.norm(Y_clean, ord="fro")),
        "schedules": schedules,
        "alpha_by_method": dict(alpha_by_method),
    }


def best_rank1_matrix_for_residual(P_res: np.ndarray, C_inv_sqrt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the stage-k subroutine-optimal rank-1 matrix.

    The notebook's stage-k objective is:
        min_rank1 M ||Y_res - M X||_F.

    Writing C = X X^T and Z = (Y_res X^T) C^{-1/2}, this is exactly equivalent to
        min_rank1 N ||Z - N||_F,
    where N = M C^{1/2}. Therefore the optimum is given by the top rank-1 SVD of Z,
    and M_opt is obtained by multiplying by C^{-1/2} on the right.

    Returns factors b_opt, a_opt such that
        M_opt = b_opt a_opt^T,
    with ||b_opt||_2 = 1 and a_opt absorbing the singular value / whitening.
    """
    Z = P_res @ C_inv_sqrt
    U, s, Vh = la.svd(Z, full_matrices=False)
    b_opt = U[:, 0]
    a_opt = s[0] * (Vh[0, :] @ C_inv_sqrt)
    opt_norm_sq = float(np.dot(a_opt, a_opt))  # because ||b_opt||_2 = 1
    return b_opt, a_opt, opt_norm_sq


def pad_stack(curves: List[np.ndarray]) -> np.ndarray:
    max_len = max(len(c) for c in curves)
    arr = np.empty((len(curves), max_len), dtype=float)
    for i, c in enumerate(curves):
        arr[i, : len(c)] = c
        arr[i, len(c) :] = c[-1]
    return arr


def summarize(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    arr = pad_stack(curves)
    return arr.mean(axis=0), arr.std(axis=0)


def save_csv(path: Path, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    data = np.column_stack([x, mean, std])
    header = "iter,mean,std"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def clip_for_log(y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    positive = y[y > 0]
    eps = 1e-12 if positive.size == 0 else max(1e-12, float(positive.min()) * 0.5)
    yy = np.maximum(y, eps)
    low = np.maximum(y - s, eps)
    high = np.maximum(y + s, eps)
    return yy, low, high


def plain_decimal_tick(v: float, _pos: float) -> str:
    """Format ticks as plain decimals (no scientific notation)."""
    if not np.isfinite(v):
        return ""
    if v == 0:
        return "0"
    return f"{v:g}"


def add_t_bars(
    ax: plt.Axes,
    schedules: Dict[str, np.ndarray],
    place_at_top: bool,
    components_to_show: int = 1,
    label_fontsize: float = 9.5,
) -> None:
    """Draw compact in-axis bars for early component allocations (T1, T2, ...)."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    spacing = 0.055
    if place_at_top:
        start = 0.955
        bars_y = {m: start - i * spacing for i, m in enumerate(PLOT_ORDER)}
    else:
        start = 0.035
        bars_y = {m: start + i * spacing for i, m in enumerate(PLOT_ORDER)}
    bar_h = 0.032

    for method in PLOT_ORDER:
        sched = np.asarray(schedules[method], dtype=int)
        color = COLORS[method]

        x0 = 0.0
        n_show = min(int(components_to_show), int(sched.size))
        for j in range(n_show):
            tj = int(sched[j])
            x1 = x0 + float(tj)
            rect = Rectangle(
                (x0, bars_y[method]),
                float(tj),
                bar_h,
                transform=trans,
                facecolor=color,
                edgecolor=color,
                alpha=max(0.10, 0.26 - 0.05 * j),
                linewidth=0.8,
                clip_on=True,
            )
            ax.add_patch(rect)

            ls = "--" if j == 0 else ":"
            ax.axvline(x1, color=color, linestyle=ls, linewidth=0.9, alpha=0.82)
            ax.text(
                x1 + 2.0,
                bars_y[method] + bar_h / 2.0,
                rf"$t_{{{j + 1}}}={tj}$",
                transform=trans,
                ha="left",
                va="center",
                fontsize=label_fontsize,
                color=color,
                clip_on=True,
            )
            x0 = x1


def style_y_ticks_decimal(ax: plt.Axes) -> None:
    """Force plain decimal labels on y-ticks (avoids scientific notation text)."""
    ax.yaxis.set_major_formatter(FuncFormatter(plain_decimal_tick))
    ax.yaxis.set_minor_formatter(FuncFormatter(plain_decimal_tick))
    ax.yaxis.offsetText.set_visible(False)


def staircase_by_component(y: np.ndarray, s: np.ndarray, schedule: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert per-iteration curves to component-wise staircase using stage-start values."""
    yy = np.asarray(y, dtype=float).copy()
    ss = np.asarray(s, dtype=float).copy()
    start = 0
    for steps in np.asarray(schedule, dtype=int):
        if start >= len(yy):
            break
        end = min(start + int(steps), len(yy))
        yy[start:end] = yy[start]
        ss[start:end] = ss[start]
        start = end
    return yy, ss


def draw_metric_on_axis(
    ax: plt.Axes,
    setting: Setting,
    problem: Dict[str, object],
    metric_name: str,
    summary: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    log_scale: bool,
    normalize_cumulative_psi: bool,
    show_legend: bool,
    legend_inside: bool,
    show_title: bool,
    show_annotation: bool,
    bars_components_to_show: int,
    bars_label_fontsize: float,
    legend_font_scale: float = 1.0,
    ylabel_scale: float = 1.0,
    curve_mode: str = "native",
) -> None:
    schedules: Dict[str, np.ndarray] = problem["schedules"]  # type: ignore[assignment]
    alpha_by_method: Dict[str, float] = problem["alpha_by_method"]  # type: ignore[assignment]
    specs = metric_specs(normalize_cumulative_psi)[metric_name]

    for method in PLOT_ORDER:
        mean, std = summary[metric_name][method]
        denom_key = str(specs["normalize_by"])
        if denom_key == "w_norm":
            denom = float(cast(float, problem["w_norm"]))
        elif denom_key == "y_norm":
            denom = float(cast(float, problem["y_norm"]))
        else:
            denom = 1.0

        y = mean / denom
        s = std / denom
        if curve_mode == "staircase":
            y, s = staircase_by_component(y, s, schedules[method])
        x = np.arange(len(y))

        if log_scale:
            yy, low, high = clip_for_log(y, s)
        else:
            yy = y
            low = np.maximum(y - s, 0.0)
            high = y + s

        ax.plot(
            x,
            yy,
            color=COLORS[method],
            label=schedule_formula_text(method, float(alpha_by_method[method])),
        )
        ax.fill_between(x, low, high, color=COLORS[method], alpha=0.15)

        if bool(specs["show_transition_markers"]):
            bounds = np.cumsum(schedules[method]) - 1
            bounds = bounds[(bounds >= 0) & (bounds < len(yy))]
            ax.plot(
                bounds,
                yy[bounds],
                linestyle="None",
                marker="o",
                markersize=4.6,
                markerfacecolor="white",
                markeredgecolor=COLORS[method],
                markeredgewidth=1.0,
                zorder=5,
            )

    add_t_bars(
        ax,
        schedules,
        place_at_top=(metric_name == "cumulative_psi"),
        components_to_show=bars_components_to_show,
        label_fontsize=bars_label_fontsize,
    )

    ax.set_xlabel("Iteration (cumulative across components)")
    ylabel = str(specs["ylabel"])
    if log_scale:
        ylabel = f"{ylabel} (log scale)"
    base_ylabel_size = float(plt.rcParams.get("axes.labelsize", 12.5))
    ax.set_ylabel(ylabel, fontsize=base_ylabel_size * ylabel_scale)
    if show_title:
        ax.set_title(f"{setting.title}: {specs['title']} ({'log' if log_scale else 'linear'} y-scale)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    if log_scale:
        ax.set_yscale("log")

    max_total = max(int(np.sum(s)) for s in schedules.values())
    ax.set_xlim(-8, max_total + 6)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    style_y_ticks_decimal(ax)

    if show_legend:
        if legend_inside:
            legend = ax.legend(
                title="Schedules",
                loc="upper right",
                frameon=True,
                borderaxespad=0.25,
                fontsize=float(plt.rcParams.get("legend.fontsize", 8.0)) * legend_font_scale,
                title_fontsize=float(plt.rcParams.get("legend.title_fontsize", 8.2)) * legend_font_scale,
            )
        else:
            legend = ax.legend(
                title="Schedules",
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                frameon=True,
                borderaxespad=0.0,
                fontsize=float(plt.rcParams.get("legend.fontsize", 8.0)) * legend_font_scale,
                title_fontsize=float(plt.rcParams.get("legend.title_fontsize", 8.2)) * legend_font_scale,
            )
        legend.get_frame().set_alpha(0.96)
        legend.get_frame().set_linewidth(0.6)

    if show_annotation:
        ax.text(
            0.0,
            -0.18,
            "Bars indicate first-component iterations only; dashed lines mark T1 for each schedule.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
        )


def metric_specs(normalize_cumulative_psi: bool) -> Dict[str, Dict[str, object]]:
    psi_ylabel = r"$\sum_{j<k}\|\Psi_j\|_F + \|\Psi_k^{(t)}\|_F$"
    if normalize_cumulative_psi:
        psi_ylabel = r"$(\sum_{j<k}\|\Psi_j\|_F + \|\Psi_k^{(t)}\|_F)/\|W^*\|_F$"

    return {
        "reconstruction_staircase": {
            "title": "Reconstruction error (notebook staircase)",
            "ylabel": r"$\|W^* - \sum b_ia_i^\top\|_F$",
            "normalize_by": "w_norm",
            "show_transition_markers": True,
        },
        "reconstruction_smooth": {
            "title": "Reconstruction error (smooth logging)",
            "ylabel": r"$\|W^* - \sum b_ia_i^\top\|_F$",
            "normalize_by": "w_norm",
            "show_transition_markers": True,
        },
        "training": {
            "title": "Training objective error",
            "ylabel": r"$\|Y - \sum b_ia_i^\top X\|_F$",
            "normalize_by": "y_norm",
            "show_transition_markers": True,
        },
        "cumulative_psi": {
            "title": r"Accumulated optimization error from Definition 1 / Theorems 1 and 5",
            "ylabel": psi_ylabel,
            "normalize_by": "w_norm" if normalize_cumulative_psi else "none",
            "show_transition_markers": True,
        },
    }


def _run_one_trial_impl(
    setting: Setting,
    X: np.ndarray,
    C: np.ndarray,
    C_inv_sqrt: np.ndarray,
    W_star: np.ndarray,
    Y_clean: np.ndarray,
    schedule: np.ndarray,
    trial_idx: int,
    collect_stage_summary: bool,
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, object]]]:
    """
    Exact notebook dynamics, but with algebraically equivalent fast formulas.

    The notebook updates are:
        grad_A = B.T @ (B @ A @ X - Y_res) @ X.T
        grad_B = (B @ A @ X - Y_res) @ X.T @ A.T

    Since Y_res is fixed inside each component, define:
        P_res = Y_res X.T,
        C = X X.T.

    Then exactly:
        grad_A = (B.T B) A C - B.T P_res
        grad_B = B (A C A.T) - P_res A.T

    This is algebraically identical to the notebook, just much faster.
    """
    rs = np.random.RandomState(trial_idx)

    noise = rs.normal(0.0, setting.noise_std, size=(setting.m, setting.n))
    Y = Y_clean + noise

    P_res = Y @ X.T
    norm_y_res_sq = float(np.sum(Y * Y))

    R_w = W_star.copy()              # W* - W_res
    norm_r_w_sq = float(np.sum(R_w * R_w))

    schedule = np.asarray(schedule, dtype=int)

    training_errors: List[float] = []
    reconstruction_staircase: List[float] = []
    reconstruction_smooth: List[float] = []
    cumulative_psi: List[float] = []

    cumulative_psi_completed = 0.0
    psi_per_stage: List[float] = []

    stage_summary: Optional[Dict[str, object]] = None
    if collect_stage_summary:
        Z0 = P_res @ C_inv_sqrt
        sZ0 = la.svd(Z0, full_matrices=False, compute_uv=False)
        sY_full = la.svd(Y, full_matrices=False, compute_uv=False)
        sW_full = la.svd(W_star, full_matrices=False, compute_uv=False)
        r = int(setting.r)
        sY = np.asarray(sY_full[:r], dtype=float)
        sW = np.asarray(sW_full[:r], dtype=float)
        T_Y = np.empty(r, dtype=float)
        for idx in range(r):
            T_Y[idx] = sY[idx] - sY[idx + 1] if idx + 1 < r else sY[idx]
        T_min_Y = float(np.min(T_Y)) if T_Y.size else float("nan")
        inv_T_min_Y_star = (
            float(1.0 / T_min_Y)
            if np.isfinite(T_min_Y) and T_min_Y > max(1e-300, float(np.finfo(float).tiny))
            else float("inf")
        )
        amp_Y = 2.0 + 6.0 * sY / np.maximum(T_Y, max(1e-300, float(np.finfo(float).tiny)))
        log_prod_amp_Y = float(np.sum(np.log(np.maximum(amp_Y, 1e-300))))
        gaps_W = sW[:-1] - sW[1:]
        gap_floor = max(1e-300, float(np.finfo(float).tiny))
        T_min_W = float(np.min(gaps_W)) if gaps_W.size else float("nan")
        inv_T_min_W_star = float(1.0 / T_min_W) if np.isfinite(T_min_W) and T_min_W > gap_floor else float("inf")
        amp_W = 2.0 + 6.0 * sW[:-1] / np.maximum(gaps_W, gap_floor)
        log_prod_amp_W = float(np.sum(np.log(np.maximum(amp_W, 1e-300))))

        gaps_Z = sZ0[:-1] - sZ0[1:]
        T_min_Z = float(np.min(gaps_Z)) if gaps_Z.size else float("nan")

        stage_summary = {
            "singular_values_Y": sY,
            "T_star_Y": T_Y,
            "T_min_Y": T_min_Y,
            "inv_T_min_Y": inv_T_min_Y_star,
            "amplification_Y": amp_Y,
            "log_prod_amplification_Y": log_prod_amp_Y,
            "singular_values_Z_stage1": np.asarray(sZ0, dtype=float),
            "gap_Z_stage1": np.asarray(gaps_Z, dtype=float),
            "T_min_Z_stage1": T_min_Z,
            "singular_values_W_star": sW,
            "gap_W_star": np.asarray(gaps_W, dtype=float),
            "T_min_W_star": T_min_W,
            "inv_T_min_W_star": inv_T_min_W_star,
            "amplification_W_star": np.asarray(amp_W, dtype=float),
            "log_prod_amplification_W_star": log_prod_amp_W,
        }

    for k in range(setting.r):
        A = np.zeros((1, setting.d), dtype=float)
        B = rs.randn(setting.m, 1) / math.sqrt(setting.m)
        steps = int(schedule[k])

        # Stage-k subroutine-optimal rank-1 matrix.
        b_opt, a_opt, opt_norm_sq = best_rank1_matrix_for_residual(P_res, C_inv_sqrt)
        b_opt_col = b_opt.reshape(-1, 1)
        a_opt_col = a_opt.reshape(-1, 1)

        rec_const = math.sqrt(max(norm_r_w_sq, 0.0))
        prev_error = None
        current_error_sq = None

        for _ in range(steps):
            BB = float((B.T @ B).item())
            AC = A @ C
            grad_A = BB * AC - B.T @ P_res
            aCa = float((AC @ A.T).item())
            grad_B = B * aCa - P_res @ A.T

            A -= setting.eta_a * grad_A
            B -= setting.eta_b * grad_B

            BB = float((B.T @ B).item())
            AA = float((A @ A.T).item())
            aCa = float((A @ C @ A.T).item())
            inner_train = float((B.T @ P_res @ A.T).item())

            current_error_sq = norm_y_res_sq - 2.0 * inner_train + BB * aCa
            if current_error_sq < 0.0 and current_error_sq > -1e-10:
                current_error_sq = 0.0
            current_error_sq = max(current_error_sq, 0.0)
            current_error = math.sqrt(current_error_sq)

            inner_rec = float((B.T @ R_w @ A.T).item())
            rec_sq = norm_r_w_sq - 2.0 * inner_rec + BB * AA
            if rec_sq < 0.0 and rec_sq > -1e-10:
                rec_sq = 0.0
            rec_sq = max(rec_sq, 0.0)

            cross_psi = float((B.T @ b_opt_col).item()) * float((A @ a_opt_col).item())
            psi_sq = BB * AA + opt_norm_sq - 2.0 * cross_psi
            if psi_sq < 0.0 and psi_sq > -1e-10:
                psi_sq = 0.0
            psi_sq = max(psi_sq, 0.0)

            training_errors.append(current_error)
            reconstruction_staircase.append(rec_const)
            reconstruction_smooth.append(math.sqrt(rec_sq))
            cumulative_psi.append(cumulative_psi_completed + math.sqrt(psi_sq))

            if prev_error is not None and abs(current_error - prev_error) < setting.epsilon:
                break
            prev_error = current_error

        if current_error_sq is None:
            raise RuntimeError("No inner iterations were run; this should never happen.")

        BA = B @ A
        BB = float((B.T @ B).item())
        AA = float((A @ A.T).item())
        cross_psi = float((B.T @ b_opt_col).item()) * float((A @ a_opt_col).item())
        psi_final_sq = BB * AA + opt_norm_sq - 2.0 * cross_psi
        if psi_final_sq < 0.0 and psi_final_sq > -1e-10:
            psi_final_sq = 0.0
        psi_final_sq = max(psi_final_sq, 0.0)
        psi_k = math.sqrt(psi_final_sq)
        cumulative_psi_completed += psi_k
        psi_per_stage.append(psi_k)

        # Update residuals for the next component.
        R_w = R_w - BA
        norm_r_w_sq = float(np.sum(R_w * R_w))
        P_res = P_res - BA @ C
        norm_y_res_sq = current_error_sq

    curves = {
        "training": np.asarray(training_errors, dtype=float),
        "reconstruction_staircase": np.asarray(reconstruction_staircase, dtype=float),
        "reconstruction_smooth": np.asarray(reconstruction_smooth, dtype=float),
        "cumulative_psi": np.asarray(cumulative_psi, dtype=float),
    }
    if collect_stage_summary and stage_summary is not None:
        stage_summary["psi_per_stage"] = np.asarray(psi_per_stage, dtype=float)
        stage_summary["cumulative_psi_total"] = float(np.sum(psi_per_stage))
        stage_summary["final_reconstruction_fro"] = math.sqrt(max(norm_r_w_sq, 0.0))

    return curves, stage_summary


def run_one_trial(
    setting: Setting,
    X: np.ndarray,
    C: np.ndarray,
    C_inv_sqrt: np.ndarray,
    W_star: np.ndarray,
    Y_clean: np.ndarray,
    schedule: np.ndarray,
    trial_idx: int,
) -> Dict[str, np.ndarray]:
    curves, _ = _run_one_trial_impl(
        setting, X, C, C_inv_sqrt, W_star, Y_clean, schedule, trial_idx, collect_stage_summary=False
    )
    return curves


def run_one_trial_with_stage_summary(
    setting: Setting,
    X: np.ndarray,
    C: np.ndarray,
    C_inv_sqrt: np.ndarray,
    W_star: np.ndarray,
    Y_clean: np.ndarray,
    schedule: np.ndarray,
    trial_idx: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """
    Same dynamics as run_one_trial, plus per-stage subroutine error ||Psi_k||_F and
    spectral quantities for the stage-1 whitened residual Z = Y X^T C^{-1/2} and for W*.
    """
    curves, summary = _run_one_trial_impl(
        setting, X, C, C_inv_sqrt, W_star, Y_clean, schedule, trial_idx, collect_stage_summary=True
    )
    if summary is None:
        raise RuntimeError("Internal error: missing stage summary")
    return curves, summary


def plot_metric(
    out_dir: Path,
    setting: Setting,
    problem: Dict[str, object],
    metric_name: str,
    summary: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    log_scale: bool,
    normalize_cumulative_psi: bool,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=(7, 4))
    draw_metric_on_axis(
        ax=ax,
        setting=setting,
        problem=problem,
        metric_name=metric_name,
        summary=summary,
        log_scale=log_scale,
        normalize_cumulative_psi=normalize_cumulative_psi,
        show_legend=True,
        legend_inside=False,
        show_title=True,
        show_annotation=True,
        bars_components_to_show=1,
        bars_label_fontsize=7.8,
        legend_font_scale=1.0,
        ylabel_scale=1.0,
        curve_mode="native",
    )

    fig.tight_layout()

    suffix = "log" if log_scale else "linear"
    stem = f"{setting.slug}_{metric_name}_{suffix}"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_combined_three_panel(
    out_dir: Path,
    setting: Setting,
    problem: Dict[str, object],
    summary: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    normalize_cumulative_psi: bool,
) -> List[Path]:
    """Create one page-wide 3-panel figure: left/middle/right as requested."""
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), constrained_layout=True)

    panel_cfg = [
        ("reconstruction_staircase", True),   # left
        ("training", True),                   # middle
        ("cumulative_psi", False),            # right
    ]

    for i, (metric_name, log_scale) in enumerate(panel_cfg):
        draw_metric_on_axis(
            ax=axes[i],
            setting=setting,
            problem=problem,
            metric_name=metric_name,
            summary=summary,
            log_scale=log_scale,
            normalize_cumulative_psi=normalize_cumulative_psi,
            show_legend=(i == 1),
            legend_inside=True,
            show_title=False,
            show_annotation=False,
            bars_components_to_show=1,
            bars_label_fontsize=14.5,
            legend_font_scale=1.5,
            ylabel_scale=1.3,
            curve_mode="native",
        )

    stem = f"{setting.slug}_combined_three_panel"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_combined_three_panel_all_staircase(
    out_dir: Path,
    setting: Setting,
    problem: Dict[str, object],
    summary: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    normalize_cumulative_psi: bool,
) -> List[Path]:
    """Create a 3-panel figure where all panels are shown in staircase mode."""
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), constrained_layout=True)
    panel_cfg = [
        ("reconstruction_staircase", True),
        ("training", True),
        ("cumulative_psi", False),
    ]
    for i, (metric_name, log_scale) in enumerate(panel_cfg):
        draw_metric_on_axis(
            ax=axes[i],
            setting=setting,
            problem=problem,
            metric_name=metric_name,
            summary=summary,
            log_scale=log_scale,
            normalize_cumulative_psi=normalize_cumulative_psi,
            show_legend=(i == 1),
            legend_inside=True,
            show_title=False,
            show_annotation=False,
            bars_components_to_show=1,
            bars_label_fontsize=14.5,
            legend_font_scale=1.5,
            ylabel_scale=1.3,
            curve_mode="staircase",
        )

    stem = f"{setting.slug}_combined_three_panel_all_staircase"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_combined_three_panel_all_real(
    out_dir: Path,
    setting: Setting,
    problem: Dict[str, object],
    summary: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    normalize_cumulative_psi: bool,
    stem_suffix: str = "",
) -> List[Path]:
    """Create a 3-panel figure where all panels use within-component real errors."""
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), constrained_layout=True)
    panel_cfg = [
        ("reconstruction_smooth", True),
        ("training", True),
        ("cumulative_psi", False),
    ]
    for i, (metric_name, log_scale) in enumerate(panel_cfg):
        draw_metric_on_axis(
            ax=axes[i],
            setting=setting,
            problem=problem,
            metric_name=metric_name,
            summary=summary,
            log_scale=log_scale,
            normalize_cumulative_psi=normalize_cumulative_psi,
            show_legend=(i == 1),
            legend_inside=True,
            show_title=False,
            show_annotation=False,
            bars_components_to_show=1,
            bars_label_fontsize=14.5,
            legend_font_scale=1.5,
            ylabel_scale=1.3,
            curve_mode="native",
        )

    stem = f"{setting.slug}_combined_three_panel_all_real{stem_suffix}"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_alpha_delta_curve(
    out_dir: Path,
    setting: Setting,
    metric_name: str,
    alphas: np.ndarray,
    deltas: np.ndarray,
    t1_iters: np.ndarray,
) -> List[Path]:
    """Plot delta(final error) = final(alpha) - final(uniform alpha=0) versus alpha."""
    title_map = {
        "reconstruction_smooth": "Reconstruction Error",
        "training": "Training Objective Error",
        "cumulative_psi": "Accumulated Optimization Error",
    }
    ylabel_map = {
        "reconstruction_smooth": r"$\Delta$ final reconstruction error",
        "training": r"$\Delta$ final training error",
        "cumulative_psi": r"$\Delta$ final accumulated optimization error",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Subtle baseline curve plus per-point color encoding by first-component budget T1.
    ax.plot(alphas, deltas, color="#1f77b4", linewidth=1.5, alpha=0.6)
    sc = ax.scatter(
        alphas,
        deltas,
        c=t1_iters,
        cmap="viridis",
        s=32,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(ylabel_map.get(metric_name, r"$\Delta$ final error"))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_facecolor("#fbfcfe")
    ax.yaxis.set_major_formatter(FuncFormatter(plain_decimal_tick))
    ax.yaxis.set_minor_formatter(FuncFormatter(plain_decimal_tick))
    ax.yaxis.offsetText.set_visible(False)

    # Add a compact colorbar so each alpha point carries the corresponding T1 allocation.
    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label(r"First-component iterations $T_1(\alpha)$")

    fig.tight_layout()

    stem = f"{setting.slug}_{metric_name}_final_delta_vs_alpha"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def bundle_reports(
    root: Path,
    settings: List[Setting],
    schedules: Dict[str, np.ndarray],
    alpha_by_method: Dict[str, float],
    paper_n: int,
    paper_profile: str,
) -> None:
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    formulas = {
        "family": "w_k ∝ k^(alpha-1), t_k = min_per_rank + rounded(extra_budget * w_k)",
        "uniform_condition": "alpha = 1",
        "early_heavy_condition": "alpha < 1",
        "late_heavy_condition": "alpha > 1",
        "early_heavy_alpha": float(alpha_by_method["early_heavy"]),
        "uniform_alpha": float(alpha_by_method["uniform"]),
        "late_heavy_alpha": float(alpha_by_method["late_heavy"]),
        "early_heavy_schedule": schedules["early_heavy"].tolist(),
        "uniform_schedule": schedules["uniform"].tolist(),
        "late_heavy_schedule": schedules["late_heavy"].tolist(),
    }
    write_json(report_dir / "schedule_formulas_and_schedules.json", formulas)

    text = []
    text.append("Alpha-based schedule formulas used in all generated plots")
    text.append("========================================================")
    text.append("")
    text.append("Base family: w_k ∝ k^(alpha-1), t_k = min_per_rank + rounded(extra_budget * w_k)")
    text.append("Interpretation: alpha=1 uniform, alpha<1 early-heavy, alpha>1 late-heavy")
    text.append("")
    text.append(f"Early-heavy alpha={formulas['early_heavy_alpha']}")
    text.append(str(formulas["early_heavy_schedule"]))
    text.append("")
    text.append(f"Uniform alpha={formulas['uniform_alpha']}")
    text.append(str(formulas["uniform_schedule"]))
    text.append("")
    text.append(f"Late-heavy alpha={formulas['late_heavy_alpha']}")
    text.append(str(formulas["late_heavy_schedule"]))
    write_text(report_dir / "schedule_formulas_and_schedules.txt", "\n".join(text))

    notes = []
    notes.append("Repo setting vs paper setting used by this generator")
    notes.append("===============================================")
    notes.append("")
    notes.append("Repo setting follows the official notebook exactly:")
    notes.append("- m=100, d=200, n=500, r=20")
    notes.append("- X = N(0,1) with each column normalized to unit norm")
    notes.append("- trials = 3")
    notes.append("- singular values = geomspace(100, 1, 20)")
    notes.append("- schedules = alpha-based family with default alphas (early=0.5, uniform=1.0, late=2.0)")
    notes.append("")
    notes.append("Paper setting changes only the experimental setup:")
    notes.append("- W* in R^{500x1000}")
    notes.append("- X sampled entrywise from N(0,1)")
    notes.append("- trials = 5")
    notes.append("- r = 20")
    notes.append(f"- n = {paper_n} (paper does not state n explicitly for Figure 1)")
    notes.append(f"- singular profile = {paper_profile}")
    notes.append("- schedule formulas use the same alpha-based family for clearer explanation and direct ablation")
    write_text(report_dir / "repo_vs_paper_setting_notes.txt", "\n".join(notes))

    write_json(report_dir / "settings.json", [asdict(s) for s in settings])

    readme = []
    readme.append("Figure 1 bundle from the official notebook")
    readme.append("========================================")
    readme.append("")
    readme.append("Each setting folder contains these plot families:")
    readme.append("- reconstruction_staircase_{linear,log}")
    readme.append("- reconstruction_smooth_{linear,log}")
    readme.append("- training_{linear,log}")
    readme.append("- cumulative_psi_{linear,log}")
    readme.append("")
    readme.append("The staircase reconstruction plot matches the official notebook logging behavior.")
    readme.append("The smooth reconstruction plot is the requested non-staircase variant.")
    readme.append("The cumulative_psi plots answer the reviewer request about the progress of the accumulated optimization error.")
    readme.append("")
    readme.append("Each plot is saved separately as PNG and PDF. CSV files contain the mean/std curves used in the plots.")
    write_text(root / "README.txt", "\n".join(readme))


def zip_tree(root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep alpha in repo setting and plot final-error difference versus uniform allocation (alpha=0)."
    )
    parser.add_argument("--out-root", type=Path, default=Path("figure1_plot_bundle_alpha"))
    parser.add_argument("--alpha-min", type=float, default=-2.0)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-num", type=int, default=121)
    parser.add_argument("--total-iters", type=int, default=500)
    parser.add_argument("--three-panel-alpha", type=float, default=1.0)
    parser.add_argument("--normalize-cumulative-psi", action="store_true")
    args = parser.parse_args()

    if args.alpha_num < 2:
        raise ValueError("alpha-num must be at least 2")
    if args.alpha_min >= args.alpha_max:
        raise ValueError("alpha-min must be smaller than alpha-max")
    if args.total_iters <= 0:
        raise ValueError("total-iters must be positive")

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    setting = build_settings(2500, "exponential", "repo", total_iters=args.total_iters)[0]
    setting_dir = out_root / setting.slug
    setting_dir.mkdir(parents=True, exist_ok=True)

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_num)
    metrics = ["reconstruction_smooth", "training", "cumulative_psi"]

    # Build fixed repo problem tensors once; only schedule changes with alpha.
    problem_ref = build_problem(
        setting,
        {"early_heavy": 0.0, "uniform": 0.0, "late_heavy": 0.0},
    )
    X = cast(np.ndarray, problem_ref["X"])
    C = cast(np.ndarray, problem_ref["C"])
    C_inv_sqrt = cast(np.ndarray, problem_ref["C_inv_sqrt"])
    W_star = cast(np.ndarray, problem_ref["W_star"])
    Y_clean = cast(np.ndarray, problem_ref["Y_clean"])

    # Uniform baseline (alpha = 0).
    schedule_uniform = allocate_iterations_alpha(setting.r, setting.total_iters, 0.0)
    uniform_trials: Dict[str, List[float]] = {m: [] for m in metrics}
    for trial_idx in range(setting.num_trials):
        curves_u = run_one_trial(setting, X, C, C_inv_sqrt, W_star, Y_clean, schedule_uniform, trial_idx)
        for metric_name in metrics:
            uniform_trials[metric_name].append(float(curves_u[metric_name][-1]))

    denom_by_metric: Dict[str, float] = {}
    for metric_name in metrics:
        denom_key = metric_specs(args.normalize_cumulative_psi)[metric_name]["normalize_by"]
        if denom_key == "w_norm":
            denom_by_metric[metric_name] = float(cast(float, problem_ref["w_norm"]))
        elif denom_key == "y_norm":
            denom_by_metric[metric_name] = float(cast(float, problem_ref["y_norm"]))
        else:
            denom_by_metric[metric_name] = 1.0

    baseline_final = {
        metric_name: float(np.mean(uniform_trials[metric_name])) / denom_by_metric[metric_name]
        for metric_name in metrics
    }

    final_new: Dict[str, List[float]] = {m: [] for m in metrics}
    deltas: Dict[str, List[float]] = {m: [] for m in metrics}
    t1_by_alpha: List[int] = []

    for alpha in alphas:
        print(f"[{setting.slug}] alpha={alpha:.4f}", flush=True)
        schedule_alpha = allocate_iterations_alpha(setting.r, setting.total_iters, float(alpha))
        t1_by_alpha.append(int(schedule_alpha[0]))
        trials_alpha: Dict[str, List[float]] = {m: [] for m in metrics}

        for trial_idx in range(setting.num_trials):
            curves_a = run_one_trial(setting, X, C, C_inv_sqrt, W_star, Y_clean, schedule_alpha, trial_idx)
            for metric_name in metrics:
                trials_alpha[metric_name].append(float(curves_a[metric_name][-1]))

        for metric_name in metrics:
            new_val = float(np.mean(trials_alpha[metric_name])) / denom_by_metric[metric_name]
            final_new[metric_name].append(new_val)
            deltas[metric_name].append(new_val - baseline_final[metric_name])

    for metric_name in metrics:
        arr_alpha = np.asarray(alphas, dtype=float)
        arr_new = np.asarray(final_new[metric_name], dtype=float)
        arr_delta = np.asarray(deltas[metric_name], dtype=float)
        arr_t1 = np.asarray(t1_by_alpha, dtype=float)
        arr_uniform = np.full_like(arr_alpha, baseline_final[metric_name], dtype=float)

        csv_path = setting_dir / f"{setting.slug}_{metric_name}_final_delta_vs_uniform.csv"
        data = np.column_stack([arr_alpha, arr_new, arr_uniform, arr_delta, arr_t1])
        np.savetxt(
            csv_path,
            data,
            delimiter=",",
            header="alpha,final_new,final_uniform,delta_new_minus_uniform,t1_first_component_iters",
            comments="",
        )
        plot_alpha_delta_curve(setting_dir, setting, metric_name, arr_alpha, arr_delta, arr_t1)

    # Also generate the legacy three-panel figures at a user-chosen alpha magnitude (default 0.1).
    alpha_mag = abs(float(args.three_panel_alpha))
    alpha_three = {
        "early_heavy": alpha_mag,
        "uniform": 0.0,
        "late_heavy": -alpha_mag,
    }
    problem_three = build_problem(setting, alpha_three)
    all_curves_three: Dict[str, Dict[str, List[np.ndarray]]] = {
        metric: {method: [] for method in METHODS}
        for metric in ["training", "reconstruction_staircase", "reconstruction_smooth", "cumulative_psi"]
    }
    for method in METHODS:
        schedule = np.asarray(cast(Dict[str, np.ndarray], problem_three["schedules"])[method], dtype=int)
        for trial_idx in range(setting.num_trials):
            curves = run_one_trial(
                setting=setting,
                X=cast(np.ndarray, problem_three["X"]),
                C=cast(np.ndarray, problem_three["C"]),
                C_inv_sqrt=cast(np.ndarray, problem_three["C_inv_sqrt"]),
                W_star=cast(np.ndarray, problem_three["W_star"]),
                Y_clean=cast(np.ndarray, problem_three["Y_clean"]),
                schedule=schedule,
                trial_idx=trial_idx,
            )
            for metric_name in all_curves_three:
                all_curves_three[metric_name][method].append(curves[metric_name])

    summary_three: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {metric: {} for metric in all_curves_three}
    for metric_name in all_curves_three:
        for method in METHODS:
            summary_three[metric_name][method] = summarize(all_curves_three[metric_name][method])

    suffix = f"_alpha_{str(alpha_mag).replace('-', 'm').replace('.', 'p')}"
    outs = []
    outs += plot_combined_three_panel(
        out_dir=setting_dir,
        setting=setting,
        problem=problem_three,
        summary=summary_three,
        normalize_cumulative_psi=args.normalize_cumulative_psi,
    )
    outs += plot_combined_three_panel_all_staircase(
        out_dir=setting_dir,
        setting=setting,
        problem=problem_three,
        summary=summary_three,
        normalize_cumulative_psi=args.normalize_cumulative_psi,
    )
    outs += plot_combined_three_panel_all_real(
        out_dir=setting_dir,
        setting=setting,
        problem=problem_three,
        summary=summary_three,
        normalize_cumulative_psi=args.normalize_cumulative_psi,
        stem_suffix=suffix,
    )

    # Keep the canonical name as requested and overwrite it with alpha=0.1 result.
    plot_combined_three_panel_all_real(
        out_dir=setting_dir,
        setting=setting,
        problem=problem_three,
        summary=summary_three,
        normalize_cumulative_psi=args.normalize_cumulative_psi,
        stem_suffix="",
    )

    print(f"\nWrote alpha-difference sweep outputs under: {setting_dir}")


if __name__ == "__main__":
    main()
