"""Kaplan-Meier plotting.

The user-facing entry point is :func:`km`. It is polymorphic on its first
argument: a fitted :class:`~tausurv.step.StepFunction` plots the curve
directly, while raw ``(event_time, event_indicator)`` arrays unlock the full
publication treatment -- Greenwood CI bands, an at-risk table aligned to the
curve's x-ticks, censor ticks, and group splits.

Defaults follow Pocock, Clayton & Altman (Lancet 2002) and Morris et al.
(BMJ Open 2019) on KM reporting: number-at-risk row(s) below the curve at
shared tick locations, logit-transformed pointwise CIs
(Meeker & Escobar 1998) rather than Wald, and censor ticks auto-enabled only
for small samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.stats import norm

from tausurv.plot._primitives import (
    censor_marks,
    ci_band,
    draw_at_risk_table,
    make_curve_and_table_axes,
    unique_order,
)
from tausurv.step import StepFunction

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from numpy.typing import ArrayLike


CIMethod = Literal["logit", "log-log", "wald"]


def km(
    arg1: StepFunction | ArrayLike,
    arg2: ArrayLike | None = None,
    *,
    group: ArrayLike | None = None,
    ax: Axes | None = None,
    at_risk: bool = True,
    ci: bool = True,
    ci_method: CIMethod = "logit",
    ci_level: float = 0.95,
    censor_ticks: bool | None = None,
    legend: bool = True,
    xlabel: str = "Time",
    ylabel: str = r"$\hat S(t)$",
    color: str | None = None,
    label: str | None = None,
) -> KMDisplay:
    r"""Plot a Kaplan-Meier survival curve.

    Polymorphic on the first argument: a fitted
    :class:`~tausurv.step.StepFunction` plots the curve only, while raw
    :math:`(Y, \delta)` arrays compute KM and Greenwood variance internally
    and enable CI bands, an at-risk table, censor ticks, and group splits.

    Parameters
    ----------
    arg1 : StepFunction or (n,) array
        Fitted survival function, or observed times :math:`Y = \min(T, C)`.
    arg2 : (n,) array, optional
        Event indicator :math:`\delta`, required when ``arg1`` is raw data.
    group : (n,) array, optional
        Group labels. One curve per unique label, in first-seen order. Raw-
        array input only.
    ax : Axes, optional
        Pre-existing axes to draw on. When given, ``at_risk`` is forced off
        (the at-risk table needs the function to own the figure's GridSpec).
    at_risk : bool, default True
        Render an at-risk table below the curve. Raw-array input only.
    ci : bool, default True
        Render pointwise CI bands. Raw-array input only.
    ci_method : {"logit", "log-log", "wald"}, default "logit"
        ``"logit"`` keeps the band inside :math:`[0, 1]` at the tails; Wald
        can exceed the bounds.
    ci_level : float, default 0.95
    censor_ticks : bool, optional
        ``None`` (default) auto-enables for groups with :math:`n < 100`.
    legend : bool, default True
        Legend only appears when more than one labelled group is plotted.
    xlabel, ylabel : str
    color, label : str, optional
        Override colour / legend label for a single curve.

    Returns
    -------
    KMDisplay
        ``fig``, ``ax``, optional ``at_risk_ax``, and per-group ``lines`` /
        ``ci_polys`` for further tweaking.

    Examples
    --------
    >>> ts.plot.km(Y, delta, group=arm)
    >>> ts.plot.km(Y, delta, ci_method="log-log", at_risk=False)
    >>> S = ts.kaplan_meier(Y, delta)
    >>> ts.plot.km(S)

    References
    ----------
    Kaplan & Meier (1958), JASA 53.
    Meeker, W. Q., Escobar, L. A. (1998). Statistical Methods for
    Reliability Data. Wiley.
    Pocock, Clayton & Altman (2002), Lancet 359.
    Morris et al. (2019), BMJ Open 9.
    """
    if isinstance(arg1, StepFunction):
        if arg2 is not None or group is not None:
            raise ValueError(
                "km(StepFunction) does not accept event_indicator or group; "
                "pass raw (event_time, event_indicator) for CI / at-risk / groups"
            )
        return _km_from_stepfunction(
            arg1,
            ax=ax,
            color=color,
            label=label,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=legend,
        )

    if arg2 is None:
        raise ValueError(
            "km() requires event_indicator when the first argument is not a "
            "fitted StepFunction"
        )

    Y = np.asarray(arg1, dtype=np.float64)
    D = np.asarray(arg2, dtype=np.int8)
    return _km_from_arrays(
        Y,
        D,
        group=group,
        ax=ax,
        at_risk=at_risk,
        ci=ci,
        ci_method=ci_method,
        ci_level=ci_level,
        censor_ticks=censor_ticks,
        legend=legend,
        xlabel=xlabel,
        ylabel=ylabel,
        color=color,
        label=label,
    )


@dataclass
class KMDisplay:
    """Result of :func:`km`.

    Attributes
    ----------
    fig : Figure
    ax : Axes
        The curve axes.
    at_risk_ax : Axes or None
        The at-risk table axes, if rendered.
    lines : dict[str, Line2D]
        One step curve per group, keyed by group label.
    ci_polys : dict[str, PolyCollection]
        CI band per group; empty when ``ci=False`` or for a StepFunction input.
    """

    fig: Figure
    ax: Axes
    at_risk_ax: Axes | None = None
    lines: dict[str, Line2D] = field(default_factory=dict)
    ci_polys: dict[str, PolyCollection] = field(default_factory=dict)


def _km_from_stepfunction(
    S: StepFunction,
    *,
    ax: Axes | None,
    color: str | None,
    label: str | None,
    xlabel: str,
    ylabel: str,
    legend: bool,
) -> KMDisplay:
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    t = np.concatenate([[0.0], S.time])
    s = np.concatenate([[S.baseline], S.value])

    kw: dict = {"where": "post"}
    if color is not None:
        kw["color"] = color
    if label is not None:
        kw["label"] = label

    (line,) = ax.step(t, s, **kw)
    ax.set_xlim(0.0, float(S.time.max()))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend and label is not None:
        ax.legend(loc="best")

    return KMDisplay(fig=fig, ax=ax, lines={label or "": line})


def _km_from_arrays(
    Y: np.ndarray,
    D: np.ndarray,
    *,
    group: ArrayLike | None,
    ax: Axes | None,
    at_risk: bool,
    ci: bool,
    ci_method: CIMethod,
    ci_level: float,
    censor_ticks: bool | None,
    legend: bool,
    xlabel: str,
    ylabel: str,
    color: str | None,
    label: str | None,
) -> KMDisplay:
    if Y.shape != D.shape or Y.ndim != 1:
        raise ValueError(
            f"event_time and event_indicator must be 1d arrays of equal "
            f"length; got shapes {Y.shape} and {D.shape}"
        )

    if group is None:
        labels = [label if label is not None else ""]
        curves = {labels[0]: _km_curve(Y, D)}
    else:
        if color is not None or label is not None:
            raise ValueError("color/label are single-curve only; not valid with group=")
        G = np.asarray(group)
        if G.shape != Y.shape:
            raise ValueError(
                f"group must match event_time shape; got {G.shape} vs {Y.shape}"
            )
        curves = {str(g): _km_curve(Y[G == g], D[G == g]) for g in unique_order(G)}

    fig, curve_ax, table_ax = make_curve_and_table_axes(
        ax,
        at_risk,
        n_groups=len(curves),
    )

    disp = KMDisplay(fig=fig, ax=curve_ax, at_risk_ax=table_ax)
    for name, c in curves.items():
        step_kw: dict = {"where": "post"}
        if name:
            step_kw["label"] = name
        if color is not None:
            step_kw["color"] = color
        (line,) = curve_ax.step(c.times, c.survival, **step_kw)
        disp.lines[name] = line

        if ci:
            lo, hi = _km_ci(c.survival, c.variance, ci_level, ci_method)
            disp.ci_polys[name] = ci_band(
                curve_ax,
                c.times,
                lo,
                hi,
                line.get_color(),
                step="post",
            )

        show_censor = censor_ticks if censor_ticks is not None else c.raw_n < 100
        if show_censor and (c.raw_e == 0).any():
            cens_t = c.raw_t[c.raw_e == 0]
            idx = np.searchsorted(c.times, cens_t, side="right") - 1
            censor_marks(curve_ax, cens_t, c.survival[idx], line.get_color())

    curve_ax.set_ylim(0.0, 1.02)
    curve_ax.set_ylabel(ylabel)
    if table_ax is None:
        curve_ax.set_xlabel(xlabel)
    else:
        curve_ax.tick_params(axis="x", labelbottom=False)

    if legend and len([k for k in curves if k]) > 1:
        curve_ax.legend(loc="best")

    if table_ax is not None:
        draw_at_risk_table(
            table_ax,
            curve_ax,
            rows={name: c.raw_t for name, c in curves.items()},
            colors={name: line.get_color() for name, line in disp.lines.items()},
            xlabel=xlabel,
        )

    return disp


@dataclass(frozen=True)
class _Curve:
    times: np.ndarray
    survival: np.ndarray
    variance: np.ndarray
    raw_t: np.ndarray
    raw_e: np.ndarray

    @property
    def raw_n(self) -> int:
        return int(self.raw_t.size)


def _km_curve(event_time: np.ndarray, event_indicator: np.ndarray) -> _Curve:
    Y = np.asarray(event_time, dtype=np.float64)
    D = np.asarray(event_indicator, dtype=np.float64)
    n = Y.size

    order = np.argsort(Y, kind="stable")
    Y, D = Y[order], D[order]

    unique_t, inv = np.unique(Y, return_inverse=True)
    counts_at = np.bincount(inv)
    events_at = np.bincount(inv, weights=D)

    n_at_risk = n - np.concatenate([[0], np.cumsum(counts_at[:-1])])

    survival = np.cumprod(1.0 - events_at / n_at_risk)

    denom = n_at_risk * (n_at_risk - events_at)
    with np.errstate(divide="ignore", invalid="ignore"):
        increments = np.where(denom > 0, events_at / denom, 0.0)
    variance = survival**2 * np.cumsum(increments)

    return _Curve(
        times=np.concatenate([[0.0], unique_t]),
        survival=np.concatenate([[1.0], survival]),
        variance=np.concatenate([[0.0], variance]),
        raw_t=Y,
        raw_e=D.astype(np.int8),
    )


def _km_ci(
    survival: np.ndarray,
    variance: np.ndarray,
    level: float,
    method: CIMethod,
) -> tuple[np.ndarray, np.ndarray]:
    z = float(norm.ppf(0.5 + level / 2.0))
    S = survival
    if method == "wald":
        sd = np.sqrt(variance)
        return np.clip(S - z * sd, 0.0, 1.0), np.clip(S + z * sd, 0.0, 1.0)

    Sc = np.clip(S, 1e-12, 1.0 - 1e-12)
    if method == "logit":
        with np.errstate(divide="ignore", invalid="ignore"):
            se_g = np.sqrt(variance) / (Sc * (1.0 - Sc))
        g = np.log(Sc / (1.0 - Sc))
        lo = 1.0 / (1.0 + np.exp(-(g - z * se_g)))
        hi = 1.0 / (1.0 + np.exp(-(g + z * se_g)))
    elif method == "log-log":
        with np.errstate(divide="ignore", invalid="ignore"):
            se_g = np.sqrt(variance) / (Sc * np.abs(np.log(Sc)))
        g = np.log(-np.log(Sc))
        hi = np.exp(-np.exp(g - z * se_g))
        lo = np.exp(-np.exp(g + z * se_g))
    else:
        raise ValueError(
            f"unknown ci_method {method!r}; expected 'logit', 'log-log', or 'wald'"
        )

    boundary = (S >= 1.0) | (S <= 0.0)
    lo = np.where(boundary, S, lo)
    hi = np.where(boundary, S, hi)
    return lo, hi
