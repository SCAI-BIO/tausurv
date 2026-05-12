r"""Internal rendering primitives shared by tausurv plots.

Centralises the visual constants from the API conventions section of
``docs/plots.md`` so that every plot inherits the same defaults from a single
source of truth:

- CI bands: ``alpha=0.18``, no edge, colour matches the line.
- Reference lines: ``#888888``, dashed, ``linewidth=0.6``, ``alpha=0.6``.
- At-risk tables: GridSpec layout, row labels colour-matched to the curve,
  counts aligned to the curve's major x-ticks.

These helpers are private (``_primitives``); they may be promoted to the
public surface once a real third-party caller pattern emerges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


def unique_order(arr: np.ndarray) -> list:
    """Unique values of ``arr`` in first-seen order."""
    _, idx = np.unique(arr, return_index=True)
    return list(arr[np.sort(idx)])


def ci_band(
    ax: "Axes",
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    color: str,
    *,
    step: str | None = None,
    alpha: float = 0.18,
) -> "PolyCollection":
    """Render a CI band matching the conventions.

    ``alpha=0.18``, no edge. ``step="post"`` for survival-style step bands;
    ``None`` for smooth bands.
    """
    return ax.fill_between(
        x, lo, hi, step=step, color=color, alpha=alpha, linewidth=0,
    )


def reference_line(
    ax: "Axes",
    value: float,
    *,
    orientation: Literal["h", "v"] = "h",
    color: str = "#888888",
    linestyle: str = "--",
    linewidth: float = 0.6,
    alpha: float = 0.6,
    zorder: float = 0.0,
) -> "Line2D":
    """Reference line per the conventions.

    Horizontal by default (``orientation="h"``); pass ``"v"`` for vertical.
    """
    if orientation == "h":
        return ax.axhline(
            value, color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha, zorder=zorder,
        )
    if orientation == "v":
        return ax.axvline(
            value, color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha, zorder=zorder,
        )
    raise ValueError(
        f"unknown orientation {orientation!r}; expected 'h' or 'v'"
    )


def make_curve_and_table_axes(
    ax: "Axes | None",
    at_risk: bool,
    n_groups: int,
) -> "tuple[Figure, Axes, Axes | None]":
    """Layout helper: curve axes on top, optional at-risk panel below.

    When the caller supplies ``ax`` the at-risk panel is suppressed -- the
    caller owns the figure's layout. Otherwise creates a new figure with a
    GridSpec where the table height scales with the number of groups.
    """
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax, None
    if not at_risk:
        fig, curve_ax = plt.subplots()
        return fig, curve_ax, None

    from matplotlib.gridspec import GridSpec

    fig = plt.figure()
    table_h = 0.55 + 0.32 * n_groups
    gs = GridSpec(2, 1, figure=fig, height_ratios=[4.5, table_h], hspace=0.18)
    curve_ax = fig.add_subplot(gs[0])
    table_ax = fig.add_subplot(gs[1], sharex=curve_ax)
    return fig, curve_ax, table_ax


def draw_at_risk_table(
    table_ax: "Axes",
    curve_ax: "Axes",
    rows: "dict[str, np.ndarray]",
    *,
    colors: "dict[str, str] | None" = None,
    xlabel: str = "Time",
) -> None:
    """Render at-risk count rows aligned to ``curve_ax``'s major x-ticks.

    Each entry in ``rows`` maps a group label to its raw observation times
    :math:`Y_g`. The count at each tick ``t`` is :math:`\\#\\{i : Y_{g,i} \\ge t\\}`.
    Row labels are colour-matched to the curves when ``colors`` is supplied.
    """
    from matplotlib.transforms import blended_transform_factory

    xlim = curve_ax.get_xlim()
    ticks = np.asarray(curve_ax.get_xticks(), dtype=np.float64)
    ticks = ticks[(ticks >= xlim[0]) & (ticks <= xlim[1])]

    table_ax.set_xlim(*xlim)
    table_ax.set_ylim(0.0, 1.0)
    for spine in table_ax.spines.values():
        spine.set_visible(False)
    table_ax.set_yticks([])
    table_ax.set_xticks(ticks)
    table_ax.tick_params(axis="x", length=0, pad=2.0, labelbottom=True)
    table_ax.grid(False)
    table_ax.set_xlabel(xlabel)

    labels = list(rows.keys())
    n_rows = len(labels)
    top, bottom = 0.78, 0.12
    trans = blended_transform_factory(table_ax.transData, table_ax.transAxes)
    colors = colors or {}
    for i, name in enumerate(labels):
        y = top - (i + 0.5) * (top - bottom) / n_rows
        if name:
            table_ax.text(
                -0.01, y, name, transform=table_ax.transAxes,
                ha="right", va="center", color=colors.get(name, "black"),
                fontsize=8,
            )
        raw_t = rows[name]
        for tick in ticks:
            n = int((raw_t >= float(tick)).sum())
            table_ax.text(
                tick, y, f"{n:d}", transform=trans,
                ha="center", va="center", fontsize=8,
            )

    table_ax.set_title(
        "Number at risk" if n_rows > 1 else "At risk",
        loc="left", fontsize=8, fontweight="medium", pad=2.0,
    )


def censor_marks(
    ax: "Axes",
    cens_times: np.ndarray,
    cens_y: np.ndarray,
    color: str,
) -> "Line2D":
    """Vertical tick markers at censoring times on a step curve."""
    (line,) = ax.plot(
        cens_times, cens_y, marker="|", linestyle="None",
        color=color, markersize=5, markeredgewidth=1.0,
    )
    return line
