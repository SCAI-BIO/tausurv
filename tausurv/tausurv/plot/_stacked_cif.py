r"""Stacked cumulative incidence area chart for competing risks.

Companion to :func:`tausurv.plot.cif` -- where :func:`cif` overlays line
curves, :func:`stacked_cif` stacks the cause-specific cumulative incidences
:math:`F_k(t)` and the residual survival :math:`S(t) = 1 - \sum_k F_k(t)`
so the cohort decomposition into "still surviving / has had cause k by t"
reads off the y-axis directly.

Stacking order from bottom up: cause 1 .. cause K (ascending integer
order, or sequence given to ``causes=``), then survival on top. At any t
the boundary between the topmost cause and the survival band is
:math:`\sum_k F_k(t) = 1 - S(t)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from tausurv.nonparametric import aalen_johansen, kaplan_meier

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike


@dataclass
class StackedCIFDisplay:
    """Result of :func:`stacked_cif`.

    Attributes
    ----------
    fig : Figure
    ax : Axes
    bands : dict[str, PolyCollection]
        Stacked area per cause label plus ``"Survival"`` (when shown).
    """

    fig: "Figure"
    ax: "Axes"
    bands: dict[str, "PolyCollection"] = field(default_factory=dict)


def stacked_cif(
    event_time: "ArrayLike",
    event_indicator: "ArrayLike",
    *,
    causes: "int | Sequence[int] | None" = None,
    cause_labels: "dict[int, str] | None" = None,
    show_survival: bool = True,
    survival_label: str = "Survival",
    survival_color: str = "#dddddd",
    ax: "Axes | None" = None,
    legend: bool = True,
    xlabel: str = "Time",
    ylabel: str = "Probability",
    title: str | None = None,
) -> StackedCIFDisplay:
    r"""Stacked cumulative incidence area chart.

    Parameters
    ----------
    event_time : (n,) array
    event_indicator : (n,) array
        Integer-coded: ``0`` censored, ``k >= 1`` for cause-:math:`k` event.
    causes : int or sequence of int, optional
        Causes to stack. Default: all observed in ascending order.
    cause_labels : dict[int, str], optional
        Custom legend labels per cause.
    show_survival : bool, default True
        Stack the residual survival band on top so the total height is 1.
    survival_label : str, default "Survival"
    survival_color : str
        Neutral gray by default; distinguishable from any palette colour.
    ax : Axes, optional
    legend : bool, default True
    xlabel, ylabel, title : str

    Returns
    -------
    StackedCIFDisplay

    References
    ----------
    Aalen & Johansen (1978), Scand. J. Stat. 5.
    """
    import matplotlib.pyplot as plt

    Y = np.asarray(event_time, dtype=np.float64)
    E = np.asarray(event_indicator, dtype=np.int64)
    if Y.shape != E.shape or Y.ndim != 1:
        raise ValueError(
            f"event_time and event_indicator must be 1-D arrays of equal "
            f"length; got {Y.shape} and {E.shape}"
        )

    causes_list = _resolve_causes(causes, E)

    # Evaluate every CIF (and KM survival, if needed) on a shared grid of
    # unique observation times so stacking is exact at every step.
    grid = np.unique(Y)
    grid = np.concatenate([[0.0], grid])

    cif_steps = {c: aalen_johansen(Y, E, c) for c in causes_list}
    cif_vals = {c: cif_steps[c](grid) for c in causes_list}

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    disp = StackedCIFDisplay(fig=fig, ax=ax)

    label_for = (cause_labels or {})
    lower = np.zeros_like(grid)
    for c in causes_list:
        upper = lower + cif_vals[c]
        lbl = label_for.get(c, f"Cause {c}")
        poly = ax.fill_between(
            grid, lower, upper, step="post",
            alpha=0.85, linewidth=0, label=lbl,
        )
        disp.bands[lbl] = poly
        lower = upper

    if show_survival:
        km = kaplan_meier(Y, E.clip(max=1))
        S = km(grid)
        upper = np.minimum(1.0, lower + S)
        poly = ax.fill_between(
            grid, lower, upper, step="post",
            color=survival_color, alpha=0.85, linewidth=0,
            label=survival_label,
        )
        disp.bands[survival_label] = poly

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    return disp


def _resolve_causes(
    causes: "int | Sequence[int] | None", E: np.ndarray,
) -> list[int]:
    if causes is None:
        observed = sorted({int(c) for c in np.unique(E) if c > 0})
        if not observed:
            raise ValueError(
                "no events found in event_indicator (all values <= 0); "
                "cannot stack CIF"
            )
        return observed
    if isinstance(causes, int):
        out = [causes]
    else:
        out = [int(c) for c in causes]
    if not out:
        raise ValueError("causes must be a non-empty sequence")
    for c in out:
        if c <= 0:
            raise ValueError(
                f"every cause must be a positive integer; got {c}"
            )
    return out
