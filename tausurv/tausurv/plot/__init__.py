"""Plotting for tausurv.

Matplotlib is an optional dependency (``tausurv[plot]``) and is loaded lazily
when a plotting function is called -- importing :mod:`tausurv` does not import
matplotlib.
"""

from tausurv.plot import colors, copula, shap
from tausurv.plot._calibration import calibration
from tausurv.plot._cif import cif
from tausurv.plot._forest import forest
from tausurv.plot._km import km
from tausurv.plot._metric_over_time import (
    auc_over_time,
    brier_over_time,
    concordance_over_time,
)
from tausurv.plot._predicted_survival import predicted_survival
from tausurv.plot._risk_strata import risk_strata
from tausurv.plot._stacked_cif import stacked_cif
from tausurv.plot.style import set_style, style_context

__all__ = [
    "auc_over_time",
    "brier_over_time",
    "calibration",
    "cif",
    "colors",
    "concordance_over_time",
    "copula",
    "forest",
    "km",
    "predicted_survival",
    "risk_strata",
    "set_style",
    "shap",
    "stacked_cif",
    "style_context",
]
