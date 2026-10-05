r"""Predicted survival curve plot.

A generic primitive for visualising :math:`\hat S(t \mid x)` from any fitted
model. Three modes pick automatically from input shape and the ``group``
argument:

- **single curve.** ``S`` is 1-D ``(n_times,)``. Plots one step curve plus
  an optional CI band.
- **individual cohort.** ``S`` is 2-D ``(n_subjects, n_times)`` with no
  ``group``. Plots every subject as a faded line; alpha is auto-scaled to
  the cohort size.
- **per-group aggregate.** ``S`` is 2-D and ``group`` is supplied. Plots one
  curve per group at the chosen aggregate (median by default) with an IQR
  band (or SD via ``band="sd"``).

Setting ``aggregate="median"`` or ``"mean"`` together with 2-D ``S`` and no
``group`` produces a single overall-aggregate curve -- useful for
bootstrap-replicate visualisations of one subject.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from tausurv.plot._primitives import ci_band, unique_order

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from numpy.typing import ArrayLike


Aggregate = Literal["median", "mean"]
Band = Literal["iqr", "sd"]


@dataclass
class PredictedSurvivalDisplay:
    """Result of :func:`predicted_survival`.

    Attributes
    ----------
    fig : Figure
    ax : Axes
    lines : dict[str, Line2D]
        Aggregate / primary curves keyed by group label (or ``""`` for a
        single unnamed curve).
    ci_polys : dict[str, PolyCollection]
        Band per curve.
    individual_lines : list[Line2D]
        Faded per-subject lines, populated only in the *individual cohort*
        mode.
    """

    fig: "Figure"
    ax: "Axes"
    lines: dict[str, "Line2D"] = field(default_factory=dict)
    ci_polys: dict[str, "PolyCollection"] = field(default_factory=dict)
    individual_lines: list["Line2D"] = field(default_factory=list)


def predicted_survival(
    times: "ArrayLike",
    S: "ArrayLike",
    *,
    ci: "tuple[ArrayLike, ArrayLike] | None" = None,
    group: "ArrayLike | None" = None,
    aggregate: Aggregate | None = None,
    band: Band = "iqr",
    individual_alpha: float | None = None,
    ax: "Axes | None" = None,
    legend: bool = True,
    color: str | None = None,
    label: str | None = None,
    xlabel: str = "Time",
    ylabel: str = r"$\hat S(t \mid x)$",
    title: str | None = None,
    ylim: "tuple[float, float] | None" = (0.0, 1.02),
) -> PredictedSurvivalDisplay:
    r"""Plot one or many predicted survival curves.

    Parameters
    ----------
    times : (n_times,) array
    S : (n_times,) or (n_subjects, n_times) array
        Predicted survival probabilities.
    ci : (array, array), optional
        Lower / upper bounds for a single-curve band. Valid only when
        ``S`` is 1-D.
    group : (n_subjects,) array, optional
        Group labels for 2-D ``S``. Triggers per-group aggregation.
    aggregate : {"median", "mean"}, optional
        Aggregation across subjects. With ``group`` set, defaults to
        ``"median"``. With 2-D ``S`` and no ``group``, ``None`` plots
        individual subject lines and any non-None value collapses to a
        single overall-aggregate curve.
    band : {"iqr", "sd"}, default "iqr"
        Band style when aggregating. IQR uses the 25-75 percentile;
        SD uses the sample standard deviation (``ddof=1``).
    individual_alpha : float, optional
        Alpha for per-subject lines (individual mode). ``None`` (default)
        scales as ``max(0.05, min(0.5, 10 / n_subjects))``.
    ax : Axes, optional
    legend : bool, default True
        Shown when more than one named curve is plotted.
    color, label : str, optional
        Single-curve overrides.
    xlabel, ylabel, title : str
    ylim : (float, float), optional
        Default ``(0.0, 1.02)``. Pass ``None`` to keep autoscale.

    Returns
    -------
    PredictedSurvivalDisplay
    """
    import matplotlib.pyplot as plt

    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"times must be 1-D; got shape {t.shape}")

    S_arr = np.asarray(S, dtype=np.float64)
    if S_arr.ndim == 1:
        return _render_single(
            t,
            S_arr,
            ci=ci,
            ax=ax,
            color=color,
            label=label,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            ylim=ylim,
            legend=legend,
        )

    if S_arr.ndim != 2:
        raise ValueError(f"S must be 1-D or 2-D; got shape {S_arr.shape}")
    if S_arr.shape[1] != t.size:
        raise ValueError(
            f"S's time dimension ({S_arr.shape[1]}) must match times length ({t.size})"
        )
    if ci is not None:
        raise ValueError(
            "ci= is only valid for a single-curve (1-D) S; for cohorts use "
            "aggregate=/band= or pass a 1-D S with explicit bounds"
        )

    if group is None and aggregate is None:
        return _render_individual(
            t,
            S_arr,
            ax=ax,
            color=color,
            label=label,
            individual_alpha=individual_alpha,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            ylim=ylim,
            legend=legend,
        )

    return _render_aggregate(
        t,
        S_arr,
        group=group,
        aggregate=aggregate or "median",
        band=band,
        ax=ax,
        color=color,
        label=label,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        ylim=ylim,
        legend=legend,
    )


def _render_single(
    t,
    S,
    *,
    ci,
    ax,
    color,
    label,
    xlabel,
    ylabel,
    title,
    ylim,
    legend,
) -> PredictedSurvivalDisplay:
    import matplotlib.pyplot as plt

    if S.shape != t.shape:
        raise ValueError(
            f"S and times must be 1-D arrays of equal length; "
            f"got {S.shape} and {t.shape}"
        )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    disp = PredictedSurvivalDisplay(fig=fig, ax=ax)
    name = label or ""
    step_kw: dict = {"where": "post"}
    if name:
        step_kw["label"] = name
    if color is not None:
        step_kw["color"] = color
    (line,) = ax.step(t, S, **step_kw)
    disp.lines[name] = line

    if ci is not None:
        lo, hi = ci
        lo_a = np.asarray(lo, dtype=np.float64)
        hi_a = np.asarray(hi, dtype=np.float64)
        if lo_a.shape != t.shape or hi_a.shape != t.shape:
            raise ValueError(
                f"ci bounds must match times shape {t.shape}; "
                f"got {lo_a.shape} and {hi_a.shape}"
            )
        disp.ci_polys[name] = ci_band(
            ax,
            t,
            lo_a,
            hi_a,
            line.get_color(),
            step="post",
        )

    _apply_axes(ax, xlabel, ylabel, title, ylim)
    if legend and name:
        ax.legend(loc="best")
    return disp


def _render_individual(
    t,
    S,
    *,
    ax,
    color,
    label,
    individual_alpha,
    xlabel,
    ylabel,
    title,
    ylim,
    legend,
) -> PredictedSurvivalDisplay:
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    n = S.shape[0]
    alpha = individual_alpha if individual_alpha is not None else _auto_alpha(n)

    disp = PredictedSurvivalDisplay(fig=fig, ax=ax)
    line_color = color or "#0072B2"
    for i in range(n):
        (line,) = ax.step(
            t,
            S[i],
            where="post",
            color=line_color,
            alpha=alpha,
            linewidth=0.8,
        )
        disp.individual_lines.append(line)

    if label:
        # Add a single dummy line at full alpha for the legend, no data drawn.
        from matplotlib.lines import Line2D as _Line2D

        proxy = _Line2D([], [], color=line_color, linewidth=1.5, label=label)
        ax.add_line(proxy)
        disp.lines[label] = proxy
        if legend:
            ax.legend(loc="best")

    _apply_axes(ax, xlabel, ylabel, title, ylim)
    return disp


def _render_aggregate(
    t,
    S,
    *,
    group,
    aggregate,
    band,
    ax,
    color,
    label,
    xlabel,
    ylabel,
    title,
    ylim,
    legend,
) -> PredictedSurvivalDisplay:
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if group is None:
        groups = {label or "": np.arange(S.shape[0])}
    else:
        if color is not None or label is not None:
            raise ValueError("color/label are single-curve only; not valid with group=")
        G = np.asarray(group)
        if G.shape != (S.shape[0],):
            raise ValueError(f"group must have shape ({S.shape[0]},); got {G.shape}")
        groups = {str(g): np.where(G == g)[0] for g in unique_order(G)}

    disp = PredictedSurvivalDisplay(fig=fig, ax=ax)

    for name, idx in groups.items():
        S_g = S[idx]
        if aggregate == "median":
            center = np.median(S_g, axis=0)
        else:
            center = np.mean(S_g, axis=0)

        if band == "iqr":
            lo = np.percentile(S_g, 25, axis=0)
            hi = np.percentile(S_g, 75, axis=0)
        elif band == "sd":
            sd = (
                np.std(S_g, axis=0, ddof=1)
                if S_g.shape[0] >= 2
                else np.zeros_like(center)
            )
            lo = center - sd
            hi = center + sd
        else:
            raise ValueError(f"unknown band {band!r}; expected 'iqr' or 'sd'")

        step_kw: dict = {"where": "post"}
        if name:
            step_kw["label"] = name
        if color is not None and len(groups) == 1:
            step_kw["color"] = color
        (line,) = ax.step(t, center, **step_kw)
        disp.lines[name] = line

        disp.ci_polys[name] = ci_band(
            ax,
            t,
            lo,
            hi,
            line.get_color(),
            step="post",
        )

    _apply_axes(ax, xlabel, ylabel, title, ylim)
    if legend and len([k for k in groups if k]) > 1:
        ax.legend(loc="best")
    return disp


def _apply_axes(ax, xlabel, ylabel, title, ylim) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _auto_alpha(n: int) -> float:
    return max(0.05, min(0.5, 10.0 / max(n, 1)))
