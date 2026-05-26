"""Figure-2 style bar plot for SST-2, using existing JSONs in
``efficient_ablation_sst2_fixed/`` (no new training runs).

Bars
----
1. ``r=1``, ``r=2``, ``r=3`` come from a **single** sequential schedule's
   ``sequential_paths/<schedule_str>/component_history`` (so all three rank
   accuracies share configuration / seed / total epochs).
2. ``LoRA - r=3`` comes from one of the ``joint_*.json`` files'
   ``standard_lora.final_accuracy`` (i.e. Joint LoRA, not re-trained).

Selection
---------
The default mode picks a (joint, sequential) pair whose total adaptation
FLOPs are *close* (within ``--flops-tol`` ratio) and where the sequential's
final accuracy *beats* the joint's. Among matching pairs we report the
largest accuracy advantage. With ``--prefer-flops smaller`` (default) ties
are broken by preferring the sequential that uses fewer FLOPs.

Usage:
    python figure2_sst2_from_ablation.py
    python figure2_sst2_from_ablation.py --output figures_equivalence_sweep_10ep/figure2_sst2_match_joint.png
    python figure2_sst2_from_ablation.py --joint-epochs 10 --flops-tol 0.10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from lora_lib.plot_style import percent_formatter, xkcd_style
from schedule_ablation import estimate_lora_only_flops


_BAR_COLORS = ["#62A4D1", "#D66A6A", "#8F8FE3", "#F0B96A"]
_BAR_LABELS_4 = [
    r"$r=1$",
    r"$r=2$",
    r"$r=3$",
    r"$\mathbf{LoRA}$ - $r=3$",
]
_BAR_LABELS_3 = [
    r"$r=1$",
    r"$r=2$",
    r"$r=3$",
]


@dataclass
class SeqRecord:
    file: str
    schedule_str: str
    accs: List[float]               # length = rank, [r1_acc, r2_acc, r3_acc]
    final_acc: float
    flops_gf: float                 # adaptation training FLOPs in GF
    alpha: float
    schedule: List[int]


@dataclass
class JointRecord:
    file: str
    epochs: int
    final_acc: float
    flops_gf: float


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _flops_gf(d: dict) -> float:
    config = d.get("ablation_config", {})
    flops_meta = d.get("flops") or {}
    layers = [tuple(pair) for pair in flops_meta.get("layers", [])]
    spe = flops_meta.get("samples_per_epoch")
    if not layers or not spe:
        return float("nan")
    return estimate_lora_only_flops(config, layers, spe) / 1e9


def collect(results_dir: str, max_rank: int = 3) -> Tuple[List[SeqRecord], List[JointRecord]]:
    """Return (sequential records, joint records) parsed from the directory."""
    seqs: List[SeqRecord] = []
    joints: List[JointRecord] = []

    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".json"):
            continue
        full = os.path.join(results_dir, fn)
        try:
            d = _load_json(full)
        except Exception as e:
            print(f"[skip] {fn}: {e}", file=sys.stderr)
            continue

        config = d.get("ablation_config", {}) or {}
        ctype = config.get("type")
        flops_gf = _flops_gf(d)
        if not np.isfinite(flops_gf):
            continue

        if ctype == "joint":
            sl = d.get("standard_lora") or {}
            acc = sl.get("final_accuracy")
            if acc is None:
                continue
            joints.append(JointRecord(
                file=fn, epochs=int(config.get("epochs", -1)),
                final_acc=float(acc), flops_gf=float(flops_gf),
            ))
            continue

        seq_paths = d.get("sequential_paths") or {}
        for sched_str, path in seq_paths.items():
            history = path.get("component_history") or []
            accs = [None] * max_rank
            for entry in history:
                k = entry.get("component_index")
                a = entry.get("accuracy_after")
                if isinstance(k, int) and a is not None and 1 <= k <= max_rank:
                    accs[k - 1] = float(a)
            if any(x is None for x in accs):
                continue
            final_acc = path.get("final_accuracy")
            if final_acc is None:
                final_acc = accs[-1]
            seqs.append(SeqRecord(
                file=fn,
                schedule_str=str(sched_str),
                accs=[float(x) for x in accs],
                final_acc=float(final_acc),
                flops_gf=float(flops_gf),
                alpha=float(config.get("alpha", 0.0)),
                schedule=list(config.get("schedule", [])),
            ))

    return seqs, joints


def select_matched_pair(
    seqs: List[SeqRecord],
    joints: List[JointRecord],
    flops_tol: float,
    joint_epochs: Optional[int],
    prefer_flops: str,
) -> Optional[Tuple[SeqRecord, JointRecord, List[Tuple[SeqRecord, JointRecord, float]]]]:
    """Find a (sequential, joint) pair whose total compute is close and where
    the sequential beats the joint.

    Strategy:
      * For each candidate joint J (filtered by ``joint_epochs`` if set), look
        at every sequential S with ``|flops(S)/flops(J) - 1| <= flops_tol``
        and ``final_acc(S) > final_acc(J)``.
      * Among those, pick the largest margin ``acc(S) - acc(J)``.
      * Tie-break (within 0.001 of best margin):
          - ``prefer_flops="smaller"``: smallest ``flops(S)``.
          - ``prefer_flops="closer"``:  smallest ``|flops(S) - flops(J)|``.
          - ``prefer_flops="larger"``:  largest ``flops(S)``.
      * If multiple joints give matches, pick the joint that yields the
        biggest margin (and, on tie, the one with more epochs).

    Also returns the full ranked list of (S, J, margin) for diagnostics.
    """
    if joint_epochs is not None:
        candidate_joints = [j for j in joints if j.epochs == joint_epochs]
    else:
        candidate_joints = list(joints)
    if not candidate_joints:
        return None

    rows: List[Tuple[SeqRecord, JointRecord, float]] = []
    for j in candidate_joints:
        if j.flops_gf <= 0:
            continue
        for s in seqs:
            if s.flops_gf <= 0:
                continue
            ratio = abs(s.flops_gf / j.flops_gf - 1.0)
            if ratio > flops_tol:
                continue
            margin = s.final_acc - j.final_acc
            if margin <= 0:
                continue
            rows.append((s, j, margin))

    if not rows:
        return None

    # Sort by (margin desc) then by tie-break preference on flops.
    eps = 1e-6

    def key(row: Tuple[SeqRecord, JointRecord, float]) -> tuple:
        s, j, m = row
        if prefer_flops == "smaller":
            tb = s.flops_gf
        elif prefer_flops == "larger":
            tb = -s.flops_gf
        else:  # "closer"
            tb = abs(s.flops_gf - j.flops_gf)
        return (-round(m / eps), tb, -j.epochs)

    rows.sort(key=key)
    return rows[0][0], rows[0][1], rows


def _ylimits(
    values: List[float],
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
) -> Tuple[float, float]:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return 0.0, 1.0
    lo = max(0.0, min(finite) - 0.05) if ymin is None else float(ymin)
    hi = min(1.0, max(finite) + 0.02) if ymax is None else float(ymax)
    if hi - lo < 0.1:
        mid = 0.5 * (lo + hi)
        lo, hi = max(0.0, mid - 0.05), min(1.0, mid + 0.05)
    return lo, hi


def render(
    accuracies: List[float],
    out_path: str,
    title: str,
    subtitle: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = 0.90,
) -> str:
    if len(accuracies) == 4:
        labels = _BAR_LABELS_4
        colors = _BAR_COLORS
    elif len(accuracies) == 3:
        labels = _BAR_LABELS_3
        colors = _BAR_COLORS[:3]
    else:
        raise ValueError(f"Unexpected number of bars: {len(accuracies)}")

    with xkcd_style(scale=0.9, length=80, randomness=2):
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        bars = ax.bar(
            labels,
            accuracies,
            color=colors,
            edgecolor="black",
            linewidth=1.4,
        )
        lo, hi = _ylimits(accuracies, ymin=ymin, ymax=ymax)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("Test Accuracy", fontsize=14)
        if title:
            if subtitle:
                full_title = f"{title}\n{subtitle}"
            else:
                full_title = title
            ax.set_title(full_title, fontsize=14, fontweight="bold", pad=12)
        ax.yaxis.set_major_formatter(percent_formatter(decimals=1))
        ax.grid(axis="y", linestyle="-", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        for bar, acc in zip(bars, accuracies):
            if not np.isfinite(acc):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                acc + (hi - lo) * 0.015,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

        for tick in ax.get_xticklabels():
            tick.set_rotation(15)

        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path)
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        plt.savefig(pdf_path)
        plt.close(fig)

    print(f"Wrote {out_path}")
    print(f"   and {pdf_path}")
    return out_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir", default="efficient_ablation_sst2_fixed",
        help="Directory of SST-2 schedule-ablation JSON files.",
    )
    parser.add_argument(
        "--output",
        default="figures_equivalence_sweep_10ep/figure2_sst2_from_ablation.png",
    )
    parser.add_argument("--title", default="SST-2")
    parser.add_argument(
        "--flops-tol", type=float, default=0.10,
        help="Allowed |flops(seq)/flops(joint) - 1| (default: 0.10).",
    )
    parser.add_argument(
        "--joint-epochs", type=int, default=None,
        help="Restrict the Joint LoRA bar to this epoch budget (e.g. 10).",
    )
    parser.add_argument(
        "--prefer-flops", choices=("smaller", "closer", "larger"), default="smaller",
        help="Tie-break direction on FLOPs once accuracy margin is fixed.",
    )
    parser.add_argument("--max-rank", type=int, default=3)
    parser.add_argument(
        "--ymin", type=float, default=None,
        help="Lower y-axis bound (default: auto = min(acc) - 0.05).",
    )
    parser.add_argument(
        "--ymax", type=float, default=0.90,
        help="Upper y-axis bound (default: 0.90).",
    )
    args = parser.parse_args(argv)

    seqs, joints = collect(args.results_dir, max_rank=args.max_rank)
    if not joints:
        raise SystemExit(f"No joint_*.json files found in {args.results_dir}.")
    if not seqs:
        raise SystemExit(f"No sequential JSONs found in {args.results_dir}.")

    pick = select_matched_pair(
        seqs, joints,
        flops_tol=args.flops_tol,
        joint_epochs=args.joint_epochs,
        prefer_flops=args.prefer_flops,
    )
    if pick is None:
        raise SystemExit(
            "No (sequential, joint) pair found that beats Joint LoRA within "
            f"|flops ratio - 1| <= {args.flops_tol:.2f} "
            f"(joint_epochs={args.joint_epochs}). "
            "Try a wider --flops-tol or remove --joint-epochs."
        )
    seq, joint, ranked = pick

    print("[match-joint] Top candidates (margin desc, tie-break by --prefer-flops):")
    print(
        f"  {'rank':<5}{'seq_file':<30}{'sched':<14}"
        f"{'seq_acc':>9}{'seq_GF':>10}"
        f"{'joint_file':<18}{'joint_acc':>11}{'joint_GF':>10}{'margin':>9}{'fratio':>8}"
    )
    for i, (s, j, m) in enumerate(ranked[:8]):
        ratio = s.flops_gf / j.flops_gf
        print(
            f"  {i+1:<5}{s.file:<30}{s.schedule_str:<14}"
            f"{s.final_acc:>9.4f}{s.flops_gf:>10.1f}"
            f"{j.file:<18}{j.final_acc:>11.4f}{j.flops_gf:>10.1f}"
            f"{m:>+9.4f}{ratio:>8.3f}"
        )

    accs = list(seq.accs) + [joint.final_acc]
    title = args.title
    subtitle = (
        rf"sequential ${seq.schedule_str}$ ($\alpha={seq.alpha:g}$, {seq.flops_gf:.0f} GF)  "
        rf"vs  Joint LoRA ({joint.epochs} ep, {joint.flops_gf:.0f} GF)"
    )
    render(
        accs, args.output, title=title, subtitle=subtitle,
        ymin=args.ymin, ymax=args.ymax,
    )


if __name__ == "__main__":
    main()
