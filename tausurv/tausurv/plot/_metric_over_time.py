r"""Metric-over-time evaluation plots: AUC(t), C(t), Brier(t).

All three share one backbone (:func:`_render`) and one ``Display`` shape so
that figures compare 1:1 in side-by-side panels. The differences are purely
cosmetic defaults (y-label, y-limits, reference line, default title).

Input shapes follow the seaborn convention: the shape of ``values`` says
what it is.

- ``(times, values)`` with ``values`` of shape ``(n_times,)`` is one curve;
  ``ci=(lo, hi)`` adds a precomputed band.
- ``values`` of shape ``(n_estimates, n_times)`` is several estimates of the
  same curve, from cross-validation folds, bootstrap resamples or repeated
  runs. The line is their mean and ``band`` is the spread, :math:`\pm 1` SD
  by default, computed here.
- ``models=`` overlays several curves, each entry choosing its own form.

A ``from_estimator`` classmethod is intentionally not provided at this layer
-- computing AUC/Brier/C from a fitted predictor requires a particular
predictor protocol and a censoring estimator. Once those concerns settle,
``Display.from_estimator`` slots in without changing the public function API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from tausurv.plot._primitives import ci_band, reference_line

Band = Literal["sd", "se"] | None

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from numpy.typing import ArrayLike


@dataclass
class MetricCurveDisplay:
    """Result of :func:`auc_over_time`, :func:`concordance_over_time`,
    :func:`brier_over_time`.

    Attributes
    ----------
    fig : Figure
    ax : Axes
    lines : dict[str, Line2D]
        One step curve per model, keyed by model name (or ``""`` for a single
        curve with no name).
    ci_polys : dict[str, PolyCollection]
        CI band per model. Empty when no CI was supplied.
    reference : Line2D or None
        The reference line (e.g. 0.5 for AUC / C-index), if drawn.
    """

    fig: "Figure"
    ax: "Axes"
    lines: dict[str, "Line2D"] = field(default_factory=dict)
    ci_polys: dict[str, "PolyCollection"] = field(default_factory=dict)
    reference: "Line2D | None" = None


def auc_over_time(
    times: "ArrayLike | None" = None,
    values: "ArrayLike | None" = None,
    *,
    ci: "tuple[ArrayLike, ArrayLike] | None" = None,
    band: Band = "sd",
    models: "dict[str, dict[str, Any]] | None" = None,
    ax: "Axes | None" = None,
    legend: bool = True,
    reference: float | None = 0.5,
    xlabel: str = "Time",
    ylabel: str = "AUC(t)",
    title: str | None = None,
    color: str | None = None,
    label: str | None = None,
    ylim: "tuple[float, float] | None" = (0.4, 1.0),
) -> MetricCurveDisplay:
    r"""Time-dependent cumulative/dynamic AUC vs time.

    See :mod:`tausurv.metrics.auc` for the underlying estimator
    (:func:`~tausurv.metrics.auc.uno`).

    Parameters
    ----------
    times : (n_times,) array-like
        The x-axis grid.
    values : (n_times,) or (n_estimates, n_times) array-like
        One curve, or several estimates of the same curve (cross-validation
        folds, bootstrap resamples, repeated runs). With several, the line
        is their mean and ``band`` draws the spread.
    ci : (array, array), optional
        Lower / upper bounds for a single curve's band.
    band : {"sd", "se"} or None, default "sd"
        Spread drawn around the mean of several estimates: ``mean +/- SD``,
        ``mean +/- SE``, or no band.
    models : dict[str, dict], optional
        Precomputed overlay. Each entry has ``"times"`` and ``"values"``,
        optionally ``"ci"`` for a single curve or ``"band"`` for several
        estimates.
    ax : Axes, optional
    legend : bool, default True
    reference : float, optional
        Horizontal reference line. Default ``0.5`` (random baseline).
        ``None`` suppresses it.
    xlabel, ylabel, title : str
    color, label : str, optional
        Single-curve overrides; not valid with ``models=``.
    ylim : (float, float), optional
        Default ``(0.4, 1.0)``, the conventional range for AUC plots.

    Returns
    -------
    MetricCurveDisplay
    """
    return _render(
        times=times,
        values=values,
        ci=ci,
        band=band,
        models=models,
        ax=ax,
        legend=legend,
        reference=reference,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        color=color,
        label=label,
        ylim=ylim,
    )


def concordance_over_time(
    times: "ArrayLike | None" = None,
    values: "ArrayLike | None" = None,
    *,
    ci: "tuple[ArrayLike, ArrayLike] | None" = None,
    band: Band = "sd",
    models: "dict[str, dict[str, Any]] | None" = None,
    ax: "Axes | None" = None,
    legend: bool = True,
    reference: float | None = 0.5,
    xlabel: str = "Time",
    ylabel: str = "C(t)",
    title: str | None = None,
    color: str | None = None,
    label: str | None = None,
    ylim: "tuple[float, float] | None" = (0.4, 1.0),
) -> MetricCurveDisplay:
    r"""Time-dependent concordance vs time.

    Same conventions as :func:`auc_over_time`: ``reference=0.5``,
    ``ylim=(0.4, 1.0)``, one curve, several estimates, or ``models=``.
    """
    return _render(
        times=times,
        values=values,
        ci=ci,
        band=band,
        models=models,
        ax=ax,
        legend=legend,
        reference=reference,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        color=color,
        label=label,
        ylim=ylim,
    )


def brier_over_time(
    times: "ArrayLike | None" = None,
    values: "ArrayLike | None" = None,
    *,
    ci: "tuple[ArrayLike, ArrayLike] | None" = None,
    band: Band = "sd",
    models: "dict[str, dict[str, Any]] | None" = None,
    ax: "Axes | None" = None,
    legend: bool = True,
    reference: float | None = None,
    xlabel: str = "Time",
    ylabel: str = "Brier(t)",
    title: str | None = None,
    color: str | None = None,
    label: str | None = None,
    ylim: "tuple[float, float] | None" = (0.0, 0.25),
) -> MetricCurveDisplay:
    r"""Time-dependent Brier score vs time.

    No default reference line -- there is no universal baseline for a Brier
    curve (a marginal-KM Brier serves the role and is best passed as another
    entry in ``models=``). Inputs as in :func:`auc_over_time`.
    """
    return _render(
        times=times,
        values=values,
        ci=ci,
        band=band,
        models=models,
        ax=ax,
        legend=legend,
        reference=reference,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        color=color,
        label=label,
        ylim=ylim,
    )


def _render(
    *,
    times: "ArrayLike | None",
    values: "ArrayLike | None",
    ci: "tuple[ArrayLike, ArrayLike] | None",
    band: Band,
    models: "dict[str, dict[str, Any]] | None",
    ax: "Axes | None",
    legend: bool,
    reference: float | None,
    xlabel: str,
    ylabel: str,
    title: str | None,
    color: str | None,
    label: str | None,
    ylim: "tuple[float, float] | None",
) -> MetricCurveDisplay:
    import matplotlib.pyplot as plt

    curves = _gather_curves(
        times=times,
        values=values,
        ci=ci,
        band=band,
        models=models,
        label=label,
        color=color,
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    disp = MetricCurveDisplay(fig=fig, ax=ax)

    if reference is not None:
        disp.reference = reference_line(ax, reference)

    for name, c in curves.items():
        kw: dict = {}
        if name:
            kw["label"] = name
        if c["color"] is not None:
            kw["color"] = c["color"]
        (line,) = ax.plot(c["times"], c["values"], **kw)
        disp.lines[name] = line

        if c["ci"] is not None:
            lo, hi = c["ci"]
            disp.ci_polys[name] = ci_band(
                ax,
                c["times"],
                lo,
                hi,
                line.get_color(),
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend and len([k for k in curves if k]) > 1:
        ax.legend()

    return disp


def _gather_curves(
    *,
    times: "ArrayLike | None",
    values: "ArrayLike | None",
    ci: "tuple[ArrayLike, ArrayLike] | None",
    band: Band,
    models: "dict[str, dict[str, Any]] | None",
    label: str | None,
    color: str | None,
) -> dict[str, dict[str, Any]]:
    if models is not None:
        if times is not None or values is not None or ci is not None:
            raise ValueError(
                "pass either (times, values [, ci]) or models= for an overlay, not both"
            )
        if color is not None or label is not None:
            raise ValueError(
                "color/label are single-curve only; not valid with models="
            )
        out: dict[str, dict[str, Any]] = {}
        for name, spec in models.items():
            t = np.asarray(spec["times"], dtype=np.float64)
            if t.ndim != 1:
                raise ValueError(
                    f"model {name!r}: times must be a 1d array, got shape {t.shape}"
                )
            v, ci_arr = _values_and_ci_from_spec(
                spec, t.shape, band, where=f"model {name!r}"
            )
            out[str(name)] = {"times": t, "values": v, "ci": ci_arr, "color": None}
        return out

    if times is None:
        raise ValueError("either single-curve inputs or models= must be supplied")
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"times must be a 1d array, got shape {t.shape}")

    v_arr, ci_arr = _values_and_ci(
        values=values,
        ci=ci,
        band=band,
        time_shape=t.shape,
        where="single curve",
    )
    return {label or "": {"times": t, "values": v_arr, "ci": ci_arr, "color": color}}


def _values_and_ci_from_spec(
    spec: "dict[str, Any]",
    time_shape: tuple[int, ...],
    default_band: Band,
    where: str,
) -> "tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]":
    return _values_and_ci(
        values=spec.get("values"),
        ci=spec.get("ci"),
        band=spec.get("band", default_band),
        time_shape=time_shape,
        where=where,
    )


def _values_and_ci(
    *,
    values: "ArrayLike | None",
    ci: "tuple[ArrayLike, ArrayLike] | None",
    band: Band,
    time_shape: tuple[int, ...],
    where: str,
) -> "tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]":
    if values is None:
        raise ValueError(f"{where}: values must be supplied")
    v_arr = np.asarray(values, dtype=np.float64)
    if v_arr.ndim == 2:
        if ci is not None:
            raise ValueError(
                f"{where}: ci is for a single curve; with several estimates the "
                f"band is computed from them"
            )
        return _mean_and_band(v_arr, time_shape, band, where)
    if v_arr.shape != time_shape:
        raise ValueError(
            f"{where}: values must have shape {time_shape} for one curve or "
            f"(n_estimates, {time_shape[0]}) for several; got {v_arr.shape}"
        )
    ci_arr = _check_ci(ci, time_shape, where)
    return v_arr, ci_arr


def _mean_and_band(
    estimates: np.ndarray,
    time_shape: tuple[int, ...],
    band: Band,
    where: str,
) -> "tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]":
    if estimates.shape[1:] != time_shape:
        raise ValueError(
            f"{where}: several estimates must have shape (n_estimates, n_times) "
            f"matching times {time_shape}; got {estimates.shape}"
        )
    if estimates.shape[0] < 2:
        raise ValueError(
            f"{where}: at least 2 estimates are needed for a band; "
            f"got {estimates.shape[0]}"
        )
    mean = np.nanmean(estimates, axis=0)
    if band is None:
        return mean, None
    std = np.nanstd(estimates, axis=0, ddof=1)
    if band == "sd":
        half = std
    elif band == "se":
        half = std / np.sqrt(estimates.shape[0])
    else:
        raise ValueError(f"{where}: unknown band {band!r}; expected 'sd', 'se' or None")
    return mean, (mean - half, mean + half)


def _check_ci(
    ci: "tuple[ArrayLike, ArrayLike] | None",
    expected_shape: tuple[int, ...],
    where: str,
) -> "tuple[np.ndarray, np.ndarray] | None":
    if ci is None:
        return None
    lo, hi = ci
    lo_a = np.asarray(lo, dtype=np.float64)
    hi_a = np.asarray(hi, dtype=np.float64)
    if lo_a.shape != expected_shape or hi_a.shape != expected_shape:
        raise ValueError(
            f"{where}: lower / upper bounds must match times shape "
            f"{expected_shape}; got {lo_a.shape} and {hi_a.shape}"
        )
    return lo_a, hi_a
