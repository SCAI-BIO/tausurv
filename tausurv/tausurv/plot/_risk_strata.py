r"""Risk-stratified Kaplan-Meier plot.

Splits subjects into ``n_bins`` quantile-based risk groups from a continuous
risk score, then plots KM survival per stratum with an at-risk table. The
bread-and-butter validation plot for any survival model: a model that
discriminates risk should produce visibly separated curves with the
highest-risk stratum dropping fastest.

A thin orchestrator on top of :func:`tausurv.plot.km`. Stratum labels are
``"Q1 (lowest)"`` .. ``"Q4 (highest)"`` for the conventional ``n_bins=4``
case, ``"D1"`` .. ``"D10"`` for deciles, otherwise ``"Q1"`` .. ``"Qk"``.
Override with ``labels=``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

from tausurv.plot._km import CIMethod, km

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from numpy.typing import ArrayLike


Binning = Literal["quantile", "equal_width", "manual"]


@dataclass
class RiskStrataDisplay:
    """Result of :func:`risk_strata`.

    Attributes
    ----------
    fig : Figure
    ax : Axes
    at_risk_ax : Axes or None
    lines : dict[str, Line2D]
        One step curve per stratum, keyed by stratum label.
    ci_polys : dict[str, PolyCollection]
    bin_edges : (n_bins + 1,) array
        Breakpoints of the risk-score bins.
    n_per_bin : (n_bins,) array
        Subject counts per stratum.
    bin_labels : list[str]
        Stratum labels, in ascending-risk order.
    """

    fig: "Figure"
    ax: "Axes"
    at_risk_ax: "Axes | None" = None
    lines: dict[str, "Line2D"] = field(default_factory=dict)
    ci_polys: dict[str, "PolyCollection"] = field(default_factory=dict)
    bin_edges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_per_bin: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    bin_labels: list[str] = field(default_factory=list)


def risk_strata(
    event_time: "ArrayLike",
    event_indicator: "ArrayLike",
    risk_score: "ArrayLike",
    *,
    n_bins: int = 4,
    binning: Binning = "quantile",
    breakpoints: "Sequence[float] | None" = None,
    labels: "Sequence[str] | None" = None,
    ax: "Axes | None" = None,
    at_risk: bool = True,
    ci: bool = True,
    ci_method: CIMethod = "logit",
    ci_level: float = 0.95,
    censor_ticks: bool | None = None,
    legend: bool = True,
    xlabel: str = "Time",
    ylabel: str = r"$\hat S(t)$",
    title: str | None = None,
) -> RiskStrataDisplay:
    r"""KM curves stratified by predicted-risk quantile.

    Parameters
    ----------
    event_time : (n,) array
    event_indicator : (n,) array
        :math:`\delta = 1` if event observed, else ``0``.
    risk_score : (n,) array
        Continuous risk score. Larger = higher risk (the
        :meth:`~tausurv.predictor.SurvivalPredictor.predict` convention).
    n_bins : int, default 4
        Number of strata.
    binning : {"quantile", "equal_width", "manual"}, default "quantile"
        ``"quantile"`` produces equal-size strata via percentile cuts.
        ``"equal_width"`` uses equal-width cuts on the score range.
        ``"manual"`` uses ``breakpoints``.
    breakpoints : sequence of float, optional
        Internal cut points for ``binning="manual"``. Length ``n_bins - 1``.
        The two endpoints are added from ``risk_score``'s range.
    labels : sequence of str, optional
        Custom stratum labels in ascending-risk order. Defaults to
        ``"Q1 (lowest)"`` .. ``"Q{n} (highest)"`` for small ``n``,
        ``"D1"`` .. ``"D10"`` for ``n=10``.
    ax, at_risk, ci, ci_method, ci_level, censor_ticks, legend :
        Forwarded to :func:`tausurv.plot.km`.
    xlabel, ylabel, title : str

    Returns
    -------
    RiskStrataDisplay
    """
    Y = np.asarray(event_time, dtype=np.float64)
    D = np.asarray(event_indicator, dtype=np.int8)
    R = np.asarray(risk_score, dtype=np.float64)
    if Y.shape != D.shape or Y.shape != R.shape or Y.ndim != 1:
        raise ValueError(
            f"event_time, event_indicator, and risk_score must be 1-D arrays "
            f"of equal length; got {Y.shape}, {D.shape}, {R.shape}"
        )
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2; got {n_bins}")

    edges = _edges(R, n_bins, binning, breakpoints)
    bin_labels = _labels(n_bins, labels)
    bin_idx = _assign(R, edges)
    n_per_bin = np.bincount(bin_idx, minlength=n_bins)
    if (n_per_bin == 0).any():
        empty = [bin_labels[i] for i in np.where(n_per_bin == 0)[0]]
        raise ValueError(
            f"empty strata: {empty}. Try fewer bins, a different binning, or "
            "manual breakpoints"
        )

    # Map bin index -> stratum label; pre-sort subjects by ascending bin so
    # km()'s first-seen group ordering produces Q1, Q2, ... in the legend.
    order = np.argsort(bin_idx, kind="stable")
    group = np.asarray([bin_labels[i] for i in bin_idx[order]])

    km_disp = km(
        Y[order], D[order], group=group,
        ax=ax, at_risk=at_risk, ci=ci, ci_method=ci_method, ci_level=ci_level,
        censor_ticks=censor_ticks, legend=legend,
        xlabel=xlabel, ylabel=ylabel,
    )
    if title is not None:
        km_disp.ax.set_title(title)

    return RiskStrataDisplay(
        fig=km_disp.fig,
        ax=km_disp.ax,
        at_risk_ax=km_disp.at_risk_ax,
        lines=km_disp.lines,
        ci_polys=km_disp.ci_polys,
        bin_edges=edges,
        n_per_bin=n_per_bin,
        bin_labels=bin_labels,
    )


def _edges(
    R: np.ndarray, n_bins: int, binning: Binning,
    breakpoints: "Sequence[float] | None",
) -> np.ndarray:
    if binning == "quantile":
        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(R, q)
    elif binning == "equal_width":
        edges = np.linspace(float(np.min(R)), float(np.max(R)), n_bins + 1)
    elif binning == "manual":
        if breakpoints is None:
            raise ValueError("binning='manual' requires breakpoints=")
        if len(breakpoints) != n_bins - 1:
            raise ValueError(
                f"manual breakpoints must have length n_bins - 1 = {n_bins - 1}; "
                f"got {len(breakpoints)}"
            )
        edges = np.concatenate([[float(np.min(R))], breakpoints, [float(np.max(R))]])
        if not np.all(np.diff(edges) > 0):
            raise ValueError(
                "breakpoints must lie strictly between the risk_score min and "
                "max, in ascending order"
            )
    else:
        raise ValueError(
            f"unknown binning {binning!r}; expected 'quantile', 'equal_width', "
            "or 'manual'"
        )
    # Disambiguate edges if duplicates appear (e.g. discrete risk scores).
    return np.unique(edges) if np.any(np.diff(edges) == 0) else edges


def _assign(R: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, R, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx


def _labels(n_bins: int, labels: "Sequence[str] | None") -> list[str]:
    if labels is not None:
        out = list(labels)
        if len(out) != n_bins:
            raise ValueError(
                f"labels must have length n_bins = {n_bins}; got {len(out)}"
            )
        return out
    if n_bins == 10:
        return [f"D{i}" for i in range(1, 11)]
    out = [f"Q{i}" for i in range(1, n_bins + 1)]
    out[0] = f"{out[0]} (lowest)"
    out[-1] = f"{out[-1]} (highest)"
    return out
