r"""Forest plot.

Horizontal point-and-CI plot, one row per coefficient. Default x-axis is
log-scaled with a vertical reference at :math:`x = 1` -- the conventional
layout for hazard ratios, odds ratios, and risk ratios. Optional per-row
annotation prints ``estimate (lo, hi)`` to the right of each CI bar.

Accepts precomputed ``(names, estimates, ci=(lo, hi))``. For a fitted
:class:`~tausurv.linear.CoxPH`, ``np.exp(model.confidence_intervals())``
gives the hazard-ratio bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

from tausurv.plot._primitives import reference_line

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from numpy.typing import ArrayLike


SortBy = Literal["input", "estimate", "name"]


@dataclass
class ForestDisplay:
    """Result of :func:`forest`.

    Attributes
    ----------
    fig : Figure
    ax : Axes
    points : Line2D or None
        Marker artist holding all point estimates.
    ci_lines : LineCollection or None
        Horizontal CI bars; ``None`` when no CI was supplied.
    reference : Line2D or None
        Vertical reference line, if drawn.
    annotations : list[Text]
        Per-row "estimate (lo, hi)" text artists, in plot order.
    """

    fig: "Figure"
    ax: "Axes"
    points: "Line2D | None" = None
    ci_lines: "LineCollection | None" = None
    reference: "Line2D | None" = None
    annotations: list["Text"] = field(default_factory=list)


def forest(
    names: "Sequence[str]",
    estimates: "ArrayLike",
    *,
    ci: "tuple[ArrayLike, ArrayLike] | None" = None,
    ax: "Axes | None" = None,
    reference: float | None = 1.0,
    xlabel: str = "Hazard ratio",
    title: str | None = None,
    log_scale: bool = True,
    annotate: bool = True,
    annotation_format: str = "{:.2f} ({:.2f}, {:.2f})",
    color: str | None = None,
    sort: SortBy = "input",
) -> ForestDisplay:
    r"""Forest plot of point estimates and confidence intervals.

    Parameters
    ----------
    names : sequence of str
        Variable labels, one per row.
    estimates : (k,) array
        Point estimates, in the scale to be plotted (e.g. hazard ratios, not
        log-hazards).
    ci : (array, array), optional
        Lower / upper CI bounds, each of shape ``(k,)``.
    ax : Axes, optional
    reference : float, optional
        Vertical reference line. Default ``1.0`` (HR / OR / RR null).
        ``None`` suppresses it. Pass ``0.0`` for risk-difference plots
        with ``log_scale=False``.
    xlabel : str, default "Hazard ratio"
    title : str, optional
    log_scale : bool, default True
        Log-scaled x-axis. Conventional for multiplicative effects.
        With ``log_scale=True`` all estimates and CI bounds must be positive.
    annotate : bool, default True
        Print ``"estimate (lo, hi)"`` to the right of each CI bar.
    annotation_format : str
        Format string accepting three positional ``float``s.
    color : str, optional
        Override the marker / CI colour. Defaults to the first colour of the
        active palette.
    sort : {"input", "estimate", "name"}, default "input"
        Row order.

    Returns
    -------
    ForestDisplay
    """
    import matplotlib.pyplot as plt

    names_arr = np.asarray(list(names))
    est = np.asarray(estimates, dtype=np.float64)
    if names_arr.shape != est.shape or est.ndim != 1:
        raise ValueError(
            f"names and estimates must be 1d sequences of equal length; "
            f"got {names_arr.shape} and {est.shape}"
        )

    lo_arr: np.ndarray | None = None
    hi_arr: np.ndarray | None = None
    if ci is not None:
        lo, hi = ci
        lo_arr = np.asarray(lo, dtype=np.float64)
        hi_arr = np.asarray(hi, dtype=np.float64)
        if lo_arr.shape != est.shape or hi_arr.shape != est.shape:
            raise ValueError(
                f"ci lower / upper bounds must match estimates shape "
                f"{est.shape}; got {lo_arr.shape} and {hi_arr.shape}"
            )

    if log_scale:
        if np.any(est <= 0) or (
            lo_arr is not None and (np.any(lo_arr <= 0) or np.any(hi_arr <= 0))
        ):
            raise ValueError(
                "log_scale=True requires positive estimates and CI bounds; "
                "for non-positive values pass log_scale=False"
            )

    order = _sort_order(names_arr, est, sort)
    names_arr = names_arr[order]
    est = est[order]
    if lo_arr is not None:
        lo_arr = lo_arr[order]
        hi_arr = hi_arr[order]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    disp = ForestDisplay(fig=fig, ax=ax)

    k = est.size
    y = np.arange(k)

    if reference is not None:
        disp.reference = reference_line(ax, reference, orientation="v")

    if lo_arr is not None:
        segments = np.stack(
            [np.column_stack([lo_arr, y]), np.column_stack([hi_arr, y])],
            axis=1,
        )
        from matplotlib.collections import LineCollection

        ci_lc = LineCollection(
            segments, colors=color or "C0", linewidths=1.4, capstyle="round",
        )
        ax.add_collection(ci_lc)
        disp.ci_lines = ci_lc

    point_kw: dict = {
        "marker": "o", "markersize": 5,
        "linestyle": "None",
    }
    if color is not None:
        point_kw["color"] = color
    (points,) = ax.plot(est, y, **point_kw)
    disp.points = points

    if annotate:
        if lo_arr is None:
            for i, e in enumerate(est):
                disp.annotations.append(
                    _row_annotation(ax, e, None, None, i, annotation_format)
                )
        else:
            for i in range(k):
                disp.annotations.append(
                    _row_annotation(
                        ax, est[i], lo_arr[i], hi_arr[i], i, annotation_format,
                    )
                )

    ax.set_yticks(y)
    ax.set_yticklabels(names_arr)
    ax.invert_yaxis()
    if log_scale:
        from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 2.0, 5.0))
        )
        ax.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12)
        )
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(axis="x", linewidth=0.5, alpha=0.3)
    ax.grid(axis="y", visible=False)

    return disp


def _sort_order(
    names: np.ndarray, estimates: np.ndarray, sort: SortBy,
) -> np.ndarray:
    if sort == "input":
        return np.arange(estimates.size)
    if sort == "estimate":
        return np.argsort(estimates)
    if sort == "name":
        return np.argsort(names)
    raise ValueError(
        f"unknown sort {sort!r}; expected 'input', 'estimate', or 'name'"
    )


def _row_annotation(
    ax: "Axes",
    estimate: float,
    lo: float | None,
    hi: float | None,
    row: int,
    fmt: str,
) -> "Text":
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    if lo is None:
        text = f"{estimate:.2f}"
    else:
        text = fmt.format(estimate, lo, hi)
    return ax.text(
        1.02, row, text, transform=trans,
        ha="left", va="center", fontsize=8,
    )
