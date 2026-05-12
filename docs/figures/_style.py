"""Shared style for tausurv docs figures.

Each figure module imports from here, calls ``setup_style()`` once, and uses
the canonical palette. Keeping the style in one place is the entire reason
this module exists; resist adding plotting helpers here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

ACCENT = "#2563eb"
INK = "#111827"
MUTED = "#9ca3af"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
        }
    )
