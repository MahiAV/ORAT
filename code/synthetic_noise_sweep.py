#!/usr/bin/env python3
"""Noise-sweep experiments (Appendix figures).

Reproduces (all under more-first schedule):

  reconstruction_error_noise_sweep.png  Gaussian noise sweep, reconstruction error
  training_error_noise_sweep.png        Gaussian noise sweep, training error
  reconstruction_error_sparse_noise.png 5%-sparse noise sweep, reconstruction error
  training_error_sparse_noise.png       5%-sparse noise sweep, training error
  reconstruction_vs_iters_kappa_{0.1,0.5,1,1.5}.png    schedule comparison per kappa
  recon_svp_kappa_{0.05,0.1,0.5,1.0}.png               SV-profile comparison per kappa

Setup follows messy_random_code/ImapctOfNOise.ipynb exactly:
  m=100, d=200, n=500, r=20, eta_A=eta_B=0.003, iters=1000,
  X columns normalized, num_trials=3 (Gaussian) / 10 (sparse).

Usage:
  python synthetic_noise_sweep.py --output-dir figures
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


KAPPA_VALUES = [0, 0.1, 0.5, 1, 1.5, 2]


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


def generate_w_star(
    m: int, d: int, r: int, profile: str, rng: np.random.RandomState, *, normalize: bool = False
) -> np.ndarray:
    """Build W^star with prescribed singular value profile.

    `normalize=True` matches messy_random_code/SV_profilesss.ipynb (||S||_2^2 = r).
    `normalize=False` matches messy_random_code/ImapctOfNOise.ipynb (raw scale).
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
        raise ValueError(profile)
    if normalize:
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


def low_rank_gd(
    X: np.ndarray,
    Y: np.ndarray,
    W_star: np.ndarray,
    r: int,
    eta_a: float,
    eta_b: float,
    iter_schedule: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
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
    return np.asarray(train_errors), np.asarray(recon_errors)


def run_kappa_sweep(
    *,
    W_star: np.ndarray,
    X: np.ndarray,
    r: int,
    eta_a: float,
    eta_b: float,
    iter_schedule: np.ndarray,
    kappas: List[float],
    num_trials: int,
    noise_kind: str,
    sparsity: float = 0.05,
) -> Dict[float, Dict[str, np.ndarray]]:
    m, n = W_star.shape[0], X.shape[1]
    results: Dict[float, Dict[str, np.ndarray]] = {}
    for kappa in kappas:
        train_runs, recon_runs = [], []
        for trial in range(num_trials):
            rng = np.random.RandomState(trial)
            if noise_kind == "gaussian":
                noise = rng.normal(0, kappa, size=(m, n))
            elif noise_kind == "sparse":
                noise = np.zeros((m, n))
                mask = rng.rand(m, n) < sparsity
                noise[mask] = rng.normal(0, kappa, size=int(mask.sum()))
            else:
                raise ValueError(noise_kind)
            Y = W_star @ X + noise
            te, re = low_rank_gd(X, Y, W_star, r, eta_a, eta_b, iter_schedule, rng)
            train_runs.append(te)
            recon_runs.append(re)
        results[kappa] = {
            "gd_errors": np.stack(train_runs),
            "rec_errors": np.stack(recon_runs),
        }
    return results


def plot_noise_sweep(
    results: Dict[float, Dict[str, np.ndarray]],
    *,
    metric: str,
    ylabel: str,
    norm: float,
    output_path: Path,
    label_prefix: str,
    legend_fontsize: int = 20,
) -> None:
    plt.figure(figsize=(12, 6))
    for kappa, run in results.items():
        curves = run[metric] / norm
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        plt.plot(mean, label=rf"{label_prefix} $\kappa$ = {kappa}", linewidth=2)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
    plt.xlabel("Iteration (Cumulative Across Components)")
    plt.ylabel(ylabel)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_schedule_comparison_per_kappa(
    per_method: Dict[str, Dict[float, Dict[str, np.ndarray]]],
    *,
    kappas: List[float],
    output_dir: Path,
    norm: float = 1.0,
) -> None:
    """For each kappa, plot one figure comparing more_first/equal/less_first reconstruction."""
    methods = list(per_method.keys())
    for kappa in kappas:
        plt.figure(figsize=(10, 6))
        min_len = min(per_method[m][kappa]["rec_errors"].shape[1] for m in methods)
        for method in methods:
            curves = per_method[method][kappa]["rec_errors"][:, :min_len] / norm
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            plt.plot(mean, label=method.replace("_", " ").title(), linewidth=2)
            plt.fill_between(range(min_len), mean - std, mean + std, alpha=0.2)
        plt.xlabel("Iteration (Cumulative Across Components)")
        plt.ylabel(r"$\| W^* - \sum B_i A_i \|_F$")
        plt.legend(fontsize=30)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        # Kappa filename format matches the LaTeX includes (.1, .5, 1, 1.5 etc., w/o trailing zeros)
        k_str = f"{kappa:g}"
        plt.savefig(output_dir / f"reconstruction_vs_iters_kappa_{k_str}.png", dpi=300)
        plt.close()


def plot_sv_profile_per_kappa(
    per_profile: Dict[str, Dict[float, np.ndarray]],
    *,
    profile_norms: Dict[str, float],
    kappas: List[float],
    output_dir: Path,
) -> None:
    """For each kappa, overlay reconstruction-error curves under uniform/exponential/power-law W*.

    Curves are jointly normalized by the maximum initial reconstruction error
    across all three profiles (the exponential profile's ||W^star||_F at this scale),
    matching the rebuttal figure where the largest curve starts at 1.0.
    """
    profiles = list(per_profile.keys())
    joint_norm = max(profile_norms.values())
    for kappa in kappas:
        plt.figure(figsize=(10, 6))
        for profile in profiles:
            curve = per_profile[profile][kappa] / joint_norm
            plt.plot(curve, label=profile.title(), linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel(r"$\| W^* - \sum B_i A_i \|_F$")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        k_str = f"{kappa:g}"
        if kappa == 1:
            k_str = "1.0"
        plt.savefig(output_dir / f"recon_svp_kappa_{k_str}.png", dpi=300)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--d", type=int, default=200)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--r", type=int, default=20)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--eta", type=float, default=0.003)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--num-trials-sparse", type=int, default=10)
    parser.add_argument(
        "--skip-schedule-comparison",
        action="store_true",
        help="Skip the reconstruction_vs_iters_kappa_* figures (slowest part).",
    )
    parser.add_argument(
        "--skip-sv-profile",
        action="store_true",
        help="Skip the recon_svp_kappa_* figures.",
    )
    args = parser.parse_args()

    _plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Same global-seed pattern as the notebook (np.random.seed(42), then sequential calls).
    rng = np.random.RandomState(42)
    X = rng.randn(args.d, args.n)
    X /= np.linalg.norm(X, axis=0, keepdims=True)
    W_star_exp = generate_w_star(args.m, args.d, args.r, "exponential", rng)

    schedule_more = allocate_iterations(args.r, args.iters, "more_first")

    print("== Gaussian noise sweep (more_first schedule) ==")
    gauss_results = run_kappa_sweep(
        W_star=W_star_exp,
        X=X,
        r=args.r,
        eta_a=args.eta,
        eta_b=args.eta,
        iter_schedule=schedule_more,
        kappas=KAPPA_VALUES,
        num_trials=args.num_trials,
        noise_kind="gaussian",
    )

    Y_norm_ref = la.norm(W_star_exp @ X, ord="fro")
    W_star_norm = la.norm(W_star_exp, ord="fro")

    plot_noise_sweep(
        gauss_results,
        metric="rec_errors",
        ylabel=r"$\| W^* - \sum B_i A_i \|_F$",
        norm=W_star_norm,
        output_path=args.output_dir / "reconstruction_error_noise_sweep.png",
        label_prefix="Gaussian",
        legend_fontsize=20,
    )
    plot_noise_sweep(
        gauss_results,
        metric="gd_errors",
        ylabel=r"$\| Y - \sum B_i A_i X \|_F$",
        norm=Y_norm_ref,
        output_path=args.output_dir / "training_error_noise_sweep.png",
        label_prefix="Gaussian",
        legend_fontsize=20,
    )

    print("== Sparse noise sweep (5% nonzeros) ==")
    sparse_results = run_kappa_sweep(
        W_star=W_star_exp,
        X=X,
        r=args.r,
        eta_a=args.eta,
        eta_b=args.eta,
        iter_schedule=schedule_more,
        kappas=KAPPA_VALUES,
        num_trials=args.num_trials_sparse,
        noise_kind="sparse",
    )

    plot_noise_sweep(
        sparse_results,
        metric="rec_errors",
        ylabel=r"$\| W^* - \sum B_i A_i \|_F$",
        norm=W_star_norm,
        output_path=args.output_dir / "reconstruction_error_sparse_noise.png",
        label_prefix="Sparse",
        legend_fontsize=25,
    )
    plot_noise_sweep(
        sparse_results,
        metric="gd_errors",
        ylabel=r"$\| Y - \sum B_i A_i X \|_F$",
        norm=Y_norm_ref,
        output_path=args.output_dir / "training_error_sparse_noise.png",
        label_prefix="Sparse",
        legend_fontsize=30,
    )

    # ---- Schedule comparison per kappa (reconstruction_vs_iters_kappa_*.png) -----
    if not args.skip_schedule_comparison:
        print("== Schedule comparison per kappa (3 methods) ==")
        # The schedule-comparison appendix figure uses a shorter horizon
        # (iters=500) so the difference between schedules is visible early on.
        sched_iters = 500
        per_method: Dict[str, Dict[float, Dict[str, np.ndarray]]] = {}
        for method in ("more_first", "equal", "less_first"):
            sched = allocate_iterations(args.r, sched_iters, method)
            per_method[method] = run_kappa_sweep(
                W_star=W_star_exp,
                X=X,
                r=args.r,
                eta_a=args.eta,
                eta_b=args.eta,
                iter_schedule=sched,
                kappas=[0.1, 0.5, 1, 1.5],
                num_trials=args.num_trials,
                noise_kind="gaussian",
            )
        plot_schedule_comparison_per_kappa(
            per_method,
            kappas=[0.1, 0.5, 1, 1.5],
            output_dir=args.output_dir,
            norm=W_star_norm,
        )

    # ---- SV profile comparison per kappa (recon_svp_kappa_*.png) -----
    if not args.skip_sv_profile:
        print("== SV profile comparison per kappa ==")
        kappa_sv = [0.05, 0.1, 0.5, 1.0]
        per_profile: Dict[str, Dict[float, np.ndarray]] = {}
        profile_norms: Dict[str, float] = {}
        for profile in ("uniform", "exponential", "power-law"):
            # Match notebook's "np.random.seed(0)" pattern; one global stream per profile.
            rng_p = np.random.RandomState(42)
            W_star_p = generate_w_star(args.m, args.d, args.r, profile, rng_p)
            profile_norms[profile] = la.norm(W_star_p, ord="fro")
            per_profile[profile] = {}
            for kappa in kappa_sv:
                rng_k = np.random.RandomState(0)
                noise = rng_k.normal(0, kappa, size=(args.m, args.n))
                Y = W_star_p @ X + noise
                rng_init = np.random.RandomState(0)
                _, rec = low_rank_gd(
                    X, Y, W_star_p, args.r, args.eta, args.eta, schedule_more, rng_init
                )
                per_profile[profile][kappa] = rec
        plot_sv_profile_per_kappa(
            per_profile,
            profile_norms=profile_norms,
            kappas=kappa_sv,
            output_dir=args.output_dir,
        )

    print(f"Wrote noise-sweep figures under {args.output_dir}")


if __name__ == "__main__":
    main()
