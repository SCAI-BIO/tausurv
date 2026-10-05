"""Calibration plot at a fixed horizon.

Predicted vs observed survival (or CIF) at a single time, binned by predicted
risk. The natural input pairs with :func:`tausurv.metrics.calibration.curve`,
which returns ``(predicted_per_bin, observed_per_bin, bin_sizes)``.

Two input shapes per the API conventions:

- single model: ``(predicted, observed)`` (plus optional ``label`` / ``color``)
- precomputed multi-model overlay: ``models={"Cox": {"predicted": ...,
  "observed": ...}, ...}``

The perfect-calibration diagonal :math:`y = x` is drawn as a reference line.
Axes auto-fit to the data with a small padding and a floor of ``0.05`` so
near-zero CIFs do not collapse into a corner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from numpy.typing import ArrayLike


@dataclass
class CalibrationDisplay:
    """Result of :func:`calibration`.

    Attributes
    ----------
    fig : Figure
    ax : Axes
    lines : dict[str, Line2D]
        One marker-and-line per model.
    reference : Line2D or None
        The :math:`y = x` diagonal, when drawn.
    """

    fig: "Figure"
    ax: "Axes"
    lines: dict[str, "Line2D"] = field(default_factory=dict)
    reference: "Line2D | None" = None


def calibration(
    predicted: "ArrayLike | None" = None,
    observed: "ArrayLike | None" = None,
    *,
    models: "dict[str, dict[str, Any]] | None" = None,
    ax: "Axes | None" = None,
    legend: bool = True,
    reference: bool = True,
    xlabel: str = "Predicted",
    ylabel: str = "Observed",
    title: str | None = None,
    color: str | None = None,
    label: str | None = None,
    aspect_equal: bool = True,
) -> CalibrationDisplay:
    r"""Survival calibration plot at a fixed horizon.

    Predicted vs observed survival (or CIF) per quantile bin, with the
    perfect-calibration diagonal as reference. Auto-scales the axes to the
    data range with a 15% pad and a 0.05 floor; for very different scales
    across causes this prevents tiny CIFs from collapsing into a corner.

    Parameters
    ----------
    predicted, observed : (n_bins,) array
        Single-model input. Typically the output of
        :func:`tausurv.metrics.calibration.curve`.
    models : dict[str, dict], optional
        Precomputed overlay. Each entry has keys ``"predicted"``,
        ``"observed"``.
    ax : Axes, optional
    legend : bool, default True
        Legend shown when more than one named curve is present.
    reference : bool, default True
        Draw the :math:`y = x` perfect-calibration diagonal.
    xlabel, ylabel : str
        Default ``"Predicted"`` / ``"Observed"`` -- override to e.g.
        ``"Predicted S(t=2y)"``.
    title : str, optional
    color, label : str, optional
        Single-curve overrides; not valid with ``models=``.
    aspect_equal : bool, default True
        Square aspect for honest predicted-vs-observed comparison.

    Returns
    -------
    CalibrationDisplay

    References
    ----------
    Royston & Altman (2013), Stat. Med. 32. Calibration of survival models.
    """
    import matplotlib.pyplot as plt

    curves = _gather_curves(
        predicted=predicted,
        observed=observed,
        models=models,
        color=color,
        label=label,
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    disp = CalibrationDisplay(fig=fig, ax=ax)

    upper = _axis_upper(curves)
    if reference:
        disp.reference = ax.plot(
            [0.0, upper],
            [0.0, upper],
            color="#888888",
            linestyle="--",
            linewidth=0.6,
            alpha=0.6,
            zorder=0,
        )[0]

    for name, c in curves.items():
        order = np.argsort(c["predicted"])
        kw: dict = {"marker": "o", "markersize": 4, "linewidth": 1.0}
        if name:
            kw["label"] = name
        if c["color"] is not None:
            kw["color"] = c["color"]
        (line,) = ax.plot(c["predicted"][order], c["observed"][order], **kw)
        disp.lines[name] = line

    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    if aspect_equal:
        ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if legend and len([k for k in curves if k]) > 1:
        ax.legend()

    return disp


def _gather_curves(
    *,
    predicted: "ArrayLike | None",
    observed: "ArrayLike | None",
    models: "dict[str, dict[str, Any]] | None",
    color: str | None,
    label: str | None,
) -> dict[str, dict[str, Any]]:
    if models is not None:
        if predicted is not None or observed is not None:
            raise ValueError(
                "pass either (predicted, observed) for a single curve or "
                "models= for an overlay, not both"
            )
        if color is not None or label is not None:
            raise ValueError(
                "color/label are single-curve only; not valid with models="
            )
        out: dict[str, dict[str, Any]] = {}
        for name, spec in models.items():
            p = np.asarray(spec["predicted"], dtype=np.float64)
            o = np.asarray(spec["observed"], dtype=np.float64)
            _check_shape(p, o, where=f"model {name!r}")
            out[str(name)] = {"predicted": p, "observed": o, "color": None}
        return out

    if predicted is None or observed is None:
        raise ValueError("either (predicted, observed) or models= must be supplied")
    p = np.asarray(predicted, dtype=np.float64)
    o = np.asarray(observed, dtype=np.float64)
    _check_shape(p, o, where="predicted/observed")
    return {label or "": {"predicted": p, "observed": o, "color": color}}


def _check_shape(p: np.ndarray, o: np.ndarray, where: str) -> None:
    if p.shape != o.shape or p.ndim != 1:
        raise ValueError(
            f"{where}: predicted and observed must be 1d arrays of equal "
            f"length; got {p.shape} and {o.shape}"
        )


def _axis_upper(curves: dict[str, dict[str, Any]]) -> float:
    data_max = 0.0
    for c in curves.values():
        data_max = max(data_max, float(np.nanmax(c["predicted"])))
        data_max = max(data_max, float(np.nanmax(c["observed"])))
    return max(data_max * 1.15, 0.05)
