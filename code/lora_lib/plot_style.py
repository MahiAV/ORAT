"""Shared matplotlib styling helpers used by every figure script."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


@contextmanager
def xkcd_style(scale: float = 1.0, length: float = 100.0, randomness: float = 2.0) -> Iterator[None]:
    """Apply a hand-drawn / xkcd matplotlib style for the duration of the block.

    The defaults match the look used in the original paper figures
    (slightly thicker strokes, larger ticks, no top/right spines by default).
    """
    with plt.xkcd(scale=scale, length=length, randomness=randomness):
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 14,
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 13,
                "legend.frameon": True,
                "legend.framealpha": 0.88,
                "legend.facecolor": "white",
                "legend.edgecolor": "0.55",
                "lines.linewidth": 2.0,
                "lines.markersize": 8,
                "lines.markeredgewidth": 1.0,
                "axes.linewidth": 1.4,
                "axes.edgecolor": "#222222",
                "savefig.bbox": "tight",
                "savefig.dpi": 220,
                # Some matplotlib + xkcd combinations crash on dashed strokes
                # ("At least one value in the dash list must be positive").
                # Force solid strokes everywhere; xkcd jitter still gives a
                # hand-drawn look.
                "lines.dash_capstyle": "round",
                "lines.solid_capstyle": "round",
            }
        )
        yield


def percent_formatter(decimals: int = 0) -> FuncFormatter:
    """Return a matplotlib formatter that prints fractions as percentages."""
    fmt = f"{{:.{decimals}f}}%"
    return FuncFormatter(lambda y, _pos: fmt.format(100 * y))
