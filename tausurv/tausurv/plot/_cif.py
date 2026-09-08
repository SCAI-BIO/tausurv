r"""Cumulative incidence function (CIF) plot for competing risks.

Two layouts:

- **Multi-cause overlay** -- ``cif(Y, delta, causes=[1, 2, 3])`` draws one
  step curve per cause from the full sample. The at-risk row is shared
  across causes (one row, ``"Overall"``) because the at-risk set does not
  depend on which cause is plotted.
- **Single cause across groups** -- ``cif(Y, delta, causes=1, group=arm)``
  draws one curve per group of subjects, with a per-group at-risk row -- the
  same shape as :func:`tausurv.plot.km` for survival.

``causes`` and ``group`` cannot both be multi-valued at once: that would
produce a ``len(causes) x n_groups`` grid which is best handled by faceting
on the caller side rather than overlaying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from tausurv.nonparametric import aalen_johansen, aalen_johansen_variance
from tausurv.plot._km import CIMethod, _km_ci
from tausurv.plot._primitives import (
    censor_marks,
    ci_band,
    draw_at_risk_table,
    make_curve_and_table_axes,
    unique_order,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from numpy.typing import ArrayLike


@dataclass
class CIFDisplay:
    """Result of :func:`cif`.

    Attributes
    ----------
    fig : Figure
    ax : Axes
    at_risk_ax : Axes or None
    lines : dict[str, Line2D]
        One step curve per (cause or group), keyed by its label.
    ci_polys : dict[str, PolyCollection]
        CI band per curve; empty when ``ci=False``.
    """

    fig: "Figure"
    ax: "Axes"
    at_risk_ax: "Axes | None" = None
    lines: dict[str, "Line2D"] = field(default_factory=dict)
    ci_polys: dict[str, "PolyCollection"] = field(default_factory=dict)


def cif(
    event_time: "ArrayLike",
    event_indicator: "ArrayLike",
    *,
    causes: "int | Sequence[int] | None" = None,
    group: "ArrayLike | None" = None,
    ax: "Axes | None" = None,
    at_risk: bool = True,
    ci: bool = True,
    ci_method: CIMethod = "logit",
    ci_level: float = 0.95,
    legend: bool = True,
    censor_ticks: bool | None = None,
    xlabel: str = "Time",
    ylabel: str = "Cumulative incidence",
    title: str | None = None,
    cause_labels: "dict[int, str] | None" = None,
) -> CIFDisplay:
    r"""Aalen-Johansen cumulative incidence plot.

    Parameters
    ----------
    event_time : (n,) array
        Observed time :math:`Y = \min(T, C)`.
    event_indicator : (n,) array
        Integer coded: ``0`` censored, ``k >= 1`` for an event of cause ``k``.
    causes : int or sequence of int, optional
        Which cause(s) to plot. ``None`` (default) plots every observed cause
        in ascending order.
    group : (n,) array, optional
        Group labels for splitting subjects. Only valid with a single cause.
    ax : Axes, optional
    at_risk : bool, default True
        Render an at-risk table below the curves. When ``ax`` is supplied the
        table is suppressed (the GridSpec is owned by the caller).
    ci : bool, default True
        Render pointwise CI bands from the Aalen-Johansen variance.
    ci_method : {"logit", "log-log", "wald"}, default "logit"
        Transformation for the band; ``"logit"`` and ``"log-log"`` keep it
        inside :math:`[0, 1]`.
    ci_level : float, default 0.95
    legend : bool, default True
    censor_ticks : bool, optional
        ``None`` auto-enables for subsamples with :math:`n < 100`.
    xlabel, ylabel, title : str
    cause_labels : dict[int, str], optional
        Custom legend labels for each cause, e.g.
        ``{1: "Relapse", 2: "Death"}``. With a single cause and no group
        the curve is unlabelled unless its cause appears here.

    Returns
    -------
    CIFDisplay

    References
    ----------
    Aalen & Johansen (1978), Scand. J. Stat. 5.
    """
    Y, E = _as_arrays(event_time, event_indicator)
    causes_list = _resolve_causes(causes, E)

    if group is not None and len(causes_list) > 1:
        raise ValueError(
            "group= is only valid with a single cause; "
            "pass causes=<int> for a per-group comparison"
        )

    curves, at_risk_rows = _build_curves(
        Y, E, causes_list, group, cause_labels,
    )

    fig, curve_ax, table_ax = make_curve_and_table_axes(
        ax, at_risk, n_groups=len(at_risk_rows),
    )

    disp = CIFDisplay(fig=fig, ax=curve_ax, at_risk_ax=table_ax)

    for label, rec in curves.items():
        step_kw: dict = {"where": "post"}
        if label:
            step_kw["label"] = label
        (line,) = curve_ax.step(rec["times"], rec["values"], **step_kw)
        disp.lines[label] = line

        if ci:
            lo, hi = _km_ci(rec["values"], rec["variance"], ci_level, ci_method)
            disp.ci_polys[label] = ci_band(
                curve_ax, rec["times"], lo, hi, line.get_color(), step="post",
            )

        show_censor = (
            censor_ticks if censor_ticks is not None else rec["raw_t"].size < 100
        )
        if show_censor and (rec["raw_e"] == 0).any():
            cens_t = rec["raw_t"][rec["raw_e"] == 0]
            idx = np.searchsorted(rec["times"], cens_t, side="right") - 1
            censor_marks(curve_ax, cens_t, rec["values"][idx], line.get_color())

    curve_ax.set_ylim(0.0, 1.02)
    curve_ax.set_ylabel(ylabel)
    if title is not None:
        curve_ax.set_title(title)
    if table_ax is None:
        curve_ax.set_xlabel(xlabel)
    else:
        curve_ax.tick_params(axis="x", labelbottom=False)

    if legend and len([k for k in curves if k]) > 1:
        curve_ax.legend(loc="best")

    if table_ax is not None:
        row_colors = (
            {name: disp.lines[name].get_color() for name in at_risk_rows}
            if set(at_risk_rows) == set(disp.lines)
            else None
        )
        draw_at_risk_table(
            table_ax, curve_ax, rows=at_risk_rows, colors=row_colors,
            xlabel=xlabel,
        )

    return disp


def _as_arrays(
    event_time: "ArrayLike", event_indicator: "ArrayLike",
) -> tuple[np.ndarray, np.ndarray]:
    Y = np.asarray(event_time, dtype=np.float64)
    E = np.asarray(event_indicator, dtype=np.int64)
    if Y.shape != E.shape or Y.ndim != 1:
        raise ValueError(
            f"event_time and event_indicator must be 1d arrays of equal "
            f"length; got {Y.shape} and {E.shape}"
        )
    return Y, E


def _resolve_causes(
    causes: "int | Sequence[int] | None", E: np.ndarray,
) -> list[int]:
    if causes is None:
        observed = sorted({int(c) for c in np.unique(E) if c > 0})
        if not observed:
            raise ValueError(
                "no events found in event_indicator (all values <= 0); "
                "cannot plot CIF"
            )
        return observed
    if isinstance(causes, int):
        return [causes]
    out = [int(c) for c in causes]
    if not out:
        raise ValueError("causes must be a non-empty sequence")
    for c in out:
        if c <= 0:
            raise ValueError(
                f"every cause must be a positive integer; got {c}"
            )
    return out


def _build_curves(
    Y: np.ndarray,
    E: np.ndarray,
    causes_list: list[int],
    group: "ArrayLike | None",
    cause_labels: "dict[int, str] | None",
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    label_for = (cause_labels or {})

    if group is None:
        curves: dict[str, dict] = {}
        if len(causes_list) == 1:
            c = causes_list[0]
            lbl = label_for.get(c, "")
            curves[lbl] = _aj_record(Y, E, c)
            at_risk_rows = {lbl: Y}
        else:
            for c in causes_list:
                lbl = label_for.get(c, f"Cause {c}")
                curves[lbl] = _aj_record(Y, E, c)
            at_risk_rows = {"Overall": Y}
        return curves, at_risk_rows

    G = np.asarray(group)
    if G.shape != Y.shape:
        raise ValueError(
            f"group must match event_time shape; got {G.shape} vs {Y.shape}"
        )
    c = causes_list[0]
    curves = {}
    at_risk_rows = {}
    for g in unique_order(G):
        mask = G == g
        key = str(g)
        curves[key] = _aj_record(Y[mask], E[mask], c)
        at_risk_rows[key] = Y[mask]
    return curves, at_risk_rows


def _aj_record(Y: np.ndarray, E: np.ndarray, cause: int) -> dict:
    step = aalen_johansen(Y, E, cause)
    variance = aalen_johansen_variance(Y, E, cause)
    return {
        "times": np.concatenate([[0.0], step.time]),
        "values": np.concatenate([[0.0], step.value]),
        "variance": np.concatenate([[0.0], variance.value]),
        "raw_t": Y,
        "raw_e": E.astype(np.int8),
    }


