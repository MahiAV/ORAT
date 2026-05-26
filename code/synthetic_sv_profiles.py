#!/usr/bin/env python3
"""Singular-value profile experiments (Appendix figures).

Reproduces:
  SVofW.png                            singular values of W^star
  SVofY.png                            singular values of Y = W^star X
  sv_reconstruction_error_comparison.png    sequential rank-1 GD recon error
  sv_training_error_comparison.png          sequential rank-1 GD train error

Setup follows messy_random_code/SV_profilesss.ipynb exactly:
  m=100, d=200, n=500, r=20, eta_A=eta_B=0.003, iters=2000 (equal schedule),
  X columns normalized, kappa=0 (noiseless), 3 trials.

Profiles (each rescaled so ||S||_2^2 = r):
  - uniform:    S_i = 10
  - exponential: S = geomspace(100, 1, r)
  - power-law:  S_i = 100 / i^2

Usage:
  python synthetic_sv_profiles.py --output-dir figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la


SV_PROFILES = ["uniform", "exponential", "power-law"]


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 28,
            "axes.labelsize": 34,
            "axes.titlesize": 36,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "legend.fontsize": 24,
            "figure.titlesize": 38,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "figure.dpi": 400,
            "savefig.dpi": 400,
            "font.family": "serif",
        }
    )


def generate_w_star(m: int, d: int, r: int, profile: str, rng: np.random.RandomState) -> np.ndarray:
    """Generate a random W^star with prescribed singular value profile.

    The singular values are normalized so ||S||_2^2 = r.
    """
    U, _ = la.qr(rng.randn(m, r))
    V, _ = la.qr(rng.randn(d, r))
    if profile == "uniform":
        S = np.ones(r) * 10.0
    elif profile == "exponential":
        S = np.geomspace(100.0, 1.0, num=r)
    elif profile == "power-law":
        k = np.arange(1, r + 1, dtype=float)
        S = 100.0 * (k ** -2.0)
    else:
        raise ValueError(f"Unknown profile: {profile}")
    S *= np.sqrt(r) / np.linalg.norm(S)
    return U @ np.diag(S) @ V.T


def allocate_iterations(r: int, total_iters: int, method: str = "equal") -> np.ndarray:
    """Per-component iteration budget (linear up/down/equal), exactly as in the notebook."""
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


def low_rank_gd(
    X: np.ndarray,
    Y: np.ndarray,
    W_star: np.ndarray,
    r: int,
    eta_a: float,
    eta_b: float,
    iter_schedule: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[List[float], List[float]]:
    """Sequential rank-1 gradient descent. Returns (train_errors, recon_errors) per iter."""
    m, _ = Y.shape
    d = X.shape[0]
    Y_res = Y.copy()
    W_res = np.zeros((m, d))
    train_errors: List[float] = []
    recon_errors: List[float] = []
    for k in range(r):
        A = np.zeros((1, d))
        B = rng.randn(m, 1) / np.sqrt(m)
        iters_k = int(iter_schedule[k])
        for _ in range(iters_k):
            resid = B @ A @ X - Y_res
            grad_A = B.T @ resid @ X.T
            grad_B = resid @ X.T @ A.T
            A -= eta_a * grad_A
            B -= eta_b * grad_B
            train_errors.append(la.norm(Y_res - B @ A @ X, ord="fro"))
            recon_errors.append(la.norm(W_star - W_res, ord="fro"))
        W_res += B @ A
        Y_res -= B @ A @ X
    return train_errors, recon_errors


def plot_singular_values(
    sv_dict: Dict[str, np.ndarray],
    *,
    matrix_symbol: str,
    marker: str,
    output_path: Path,
) -> None:
    r = next(iter(sv_dict.values())).shape[0]
    plt.figure(figsize=(10, 5))
    for profile, sv in sv_dict.items():
        plt.plot(
            range(1, r + 1),
            sv,
            marker=marker,
            label=f"{profile.title()} SVs of ${matrix_symbol}$",
        )
    plt.xlabel("Singular value index (i)")
    plt.ylabel("SV Magnitude")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(ticks=np.arange(1, r + 1, 2))
    plt.legend(fontsize=24)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_curves(
    error_dict: Dict[str, np.ndarray],
    *,
    ylabel: str,
    norm: float,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    for profile, runs in error_dict.items():
        runs = runs / norm
        mean = runs.mean(axis=0)
        std = runs.std(axis=0)
        plt.plot(mean, label=f"{profile.title()} SVs", linewidth=3)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
    plt.xlabel("Iteration (Cumulative Across Components)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--d", type=int, default=200)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--r", type=int, default=20)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--eta", type=float, default=0.003)
    parser.add_argument("--num-trials", type=int, default=3)
    args = parser.parse_args()

    _plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Match the notebook's global-seed pattern: one RandomState seeded with 42, used
    # for X and then for each profile's QR factors in sequence.
    rng = np.random.RandomState(42)
    X = rng.randn(args.d, args.n)
    X /= np.linalg.norm(X, axis=0, keepdims=True)

    sv_w: Dict[str, np.ndarray] = {}
    sv_y: Dict[str, np.ndarray] = {}
    w_star_per_profile: Dict[str, np.ndarray] = {}
    w_star_norm_per_profile: Dict[str, float] = {}
    y_norm_per_profile: Dict[str, float] = {}

    for profile in SV_PROFILES:
        W_star = generate_w_star(args.m, args.d, args.r, profile, rng)
        Y = W_star @ X
        sv_w[profile] = la.svd(W_star, compute_uv=False)[: args.r]
        sv_y[profile] = la.svd(Y, compute_uv=False)[: args.r]
        w_star_per_profile[profile] = W_star
        w_star_norm_per_profile[profile] = la.norm(W_star, ord="fro")
        y_norm_per_profile[profile] = la.norm(Y, ord="fro")

    plot_singular_values(
        sv_w,
        matrix_symbol="W^*",
        marker="o",
        output_path=args.output_dir / "SVofW.png",
    )
    plot_singular_values(
        sv_y,
        matrix_symbol="Y",
        marker="x",
        output_path=args.output_dir / "SVofY.png",
    )

    schedule = allocate_iterations(args.r, args.iters, method="equal")

    train_curves: Dict[str, List[np.ndarray]] = {}
    recon_curves: Dict[str, List[np.ndarray]] = {}
    for profile in SV_PROFILES:
        W_star = w_star_per_profile[profile]
        Y = W_star @ X
        train_runs, recon_runs = [], []
        for trial in range(args.num_trials):
            rng_t = np.random.RandomState(trial)
            te, re = low_rank_gd(X, Y, W_star, args.r, args.eta, args.eta, schedule, rng_t)
            train_runs.append(np.asarray(te))
            recon_runs.append(np.asarray(re))
        train_curves[profile] = np.stack(train_runs)
        recon_curves[profile] = np.stack(recon_runs)

    # Use the last-profile Y / W* norms exactly as in the notebook (matches the figure).
    last_profile = SV_PROFILES[-1]
    plot_error_curves(
        train_curves,
        ylabel=r"$\| Y - \sum B_i A_i X \|_F$",
        norm=y_norm_per_profile[last_profile],
        output_path=args.output_dir / "sv_training_error_comparison.png",
    )
    plot_error_curves(
        recon_curves,
        ylabel=r"$\| W^* - \sum B_i A_i \|_F$",
        norm=w_star_norm_per_profile[last_profile],
        output_path=args.output_dir / "sv_reconstruction_error_comparison.png",
    )

    print(f"Wrote SV figures under {args.output_dir}")


if __name__ == "__main__":
    main()
