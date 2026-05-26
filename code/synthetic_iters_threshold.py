#!/usr/bin/env python3
"""Iterations-vs-threshold figures (Appendix).

Reproduces iters_vs_threshold_at{1,1.5,2,2.5}.png from
messy_random_code/ComputationalEfficacy.ipynb.

For each schedule (more-first / equal / less-first) we run sequential rank-1 GD
on a power-law W^star (||S||_2^2 = r) and record, for each error threshold T,
the first cumulative iteration at which the reconstruction error
||W^star - sum_i B_i A_i||_F drops to T or below.

Different total-iteration budgets per schedule (matching the notebook):
  more_first: 8000   equal: 4000   less_first: 10000

The four output figures share the same data and only differ in the rightmost
threshold ("at" value) displayed on the x-axis.

Usage:
  python synthetic_iters_threshold.py --output-dir figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la


THRESHOLDS_FULL = [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1]
THRESHOLD_AT = {"1": 1.0, "1.5": 1.5, "2": 2.0, "2.5": 2.5}


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 28,
            "axes.labelsize": 34,
            "axes.titlesize": 36,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "legend.fontsize": 30,
            "figure.titlesize": 38,
            "lines.linewidth": 4.0,
            "lines.markersize": 12,
            "figure.dpi": 400,
            "savefig.dpi": 400,
            "font.family": "serif",
        }
    )


def generate_w_star_power_law(
    m: int, d: int, r: int, rng: np.random.RandomState, alpha: float = 2.0
) -> np.ndarray:
    U, _ = la.qr(rng.randn(m, r))
    V, _ = la.qr(rng.randn(d, r))
    k = np.arange(1, r + 1, dtype=float)
    S = 100.0 * (k ** -alpha)
    S *= np.sqrt(r) / np.linalg.norm(S)
    return U @ np.diag(S) @ V.T


def allocate_iterations(r: int, total_iters: int, method: str) -> np.ndarray:
    if method == "less_first":
        sched = np.linspace(2, total_iters // r, r)
    elif method == "more_first":
        sched = np.linspace(total_iters // r, 2, r)
    elif method == "equal":
        sched = np.full(r, total_iters // r)
    else:
        raise ValueError(method)
    sched = sched / sched.sum() * total_iters
    return np.maximum(sched.astype(int), 2)


def low_rank_gd_adaptive(
    X: np.ndarray,
    Y: np.ndarray,
    W_star: np.ndarray,
    r: int,
    eta_a: float,
    eta_b: float,
    iter_schedule: np.ndarray,
    epsilon: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sequential rank-1 GD with leftover-iteration pass-forward (matches the notebook).

    Returns the per-iteration cumulative reconstruction error
    ||W^star - sum_{j<k} B_j A_j||_F (constant within a single component, by design).
    """
    m, _ = Y.shape
    d = X.shape[0]
    Y_res = Y.copy()
    W_res = np.zeros((m, d))
    schedule = iter_schedule.copy()
    recon: List[float] = []
    for k in range(r):
        A = np.zeros((1, d))
        B = rng.randn(m, 1) / np.sqrt(m)
        iters_k = int(schedule[k])
        prev_train_err = None
        used_iters = 0
        for _ in range(iters_k):
            resid = B @ A @ X - Y_res
            grad_A = B.T @ resid @ X.T
            grad_B = resid @ X.T @ A.T
            A -= eta_a * grad_A
            B -= eta_b * grad_B
            train_err = la.norm(Y_res - B @ A @ X, ord="fro")
            recon.append(la.norm(W_star - W_res, ord="fro"))
            used_iters += 1
            if prev_train_err is not None and abs(train_err - prev_train_err) < epsilon:
                break
            prev_train_err = train_err
        leftover = int(schedule[k]) - used_iters
        if leftover > 0 and k < r - 1:
            schedule[k + 1] += leftover
        W_res += B @ A
        Y_res -= B @ A @ X
    return np.asarray(recon)


def iters_to_threshold(rec_curve: np.ndarray, threshold: float) -> float:
    """First iteration index at which rec_curve <= threshold, or NaN if never."""
    hits = np.where(rec_curve <= threshold)[0]
    return float(hits[0]) if len(hits) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--d", type=int, default=200)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--r", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.003)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--num-trials", type=int, default=3)
    args = parser.parse_args()

    _plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    iters_per_strategy = {"more_first": 8000, "equal": 4000, "less_first": 10000}
    methods = ["less_first", "more_first", "equal"]

    rng = np.random.RandomState(42)
    X = rng.randn(args.d, args.n)
    X /= np.linalg.norm(X, axis=0, keepdims=True)
    W_star = generate_w_star_power_law(args.m, args.d, args.r, rng)

    print(f"||W*||_F = {la.norm(W_star, 'fro'):.4f}")
    Y = W_star @ X  # kappa = 0 (noiseless)

    method_curves: Dict[str, np.ndarray] = {}
    for method in methods:
        total_iters = iters_per_strategy[method]
        sched = allocate_iterations(args.r, total_iters, method)
        per_trial = []
        for trial in range(args.num_trials):
            rng_t = np.random.RandomState(trial)
            rec = low_rank_gd_adaptive(
                X, Y, W_star, args.r, args.eta, args.eta, sched, args.epsilon, rng_t
            )
            per_trial.append(rec)
        # The number of iterations actually used varies across trials due to the early-stop
        # rule, so we pad to the minimum length and average.
        min_len = min(len(rec) for rec in per_trial)
        stacked = np.stack([rec[:min_len] for rec in per_trial])
        method_curves[method] = stacked.mean(axis=0)
        print(f"{method}: iters used (mean) = {min_len}")

    for at_key, at_value in THRESHOLD_AT.items():
        thresholds = [t for t in THRESHOLDS_FULL if t >= at_value]
        plt.figure(figsize=(15, 9))
        for method in methods:
            iters = [iters_to_threshold(method_curves[method], t) for t in thresholds]
            plt.plot(
                thresholds,
                iters,
                marker="o",
                linewidth=3,
                label=method.replace("_", " ").title(),
            )
        plt.gca().invert_xaxis()
        plt.xlabel(r"Error Threshold")
        plt.ylabel("Iterations")
        plt.legend(fontsize=30)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        out = args.output_dir / f"iters_vs_threshold_at{at_key}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
