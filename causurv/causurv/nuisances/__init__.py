"""Nuisance functions and cross-fitting for causal-survival estimation.

See :mod:`causurv.nuisances.types` for the carrier dataclasses and
:mod:`causurv.nuisances.cross_fit` for fitting helpers.
"""

from causurv.nuisances import pseudo_outcomes
from causurv.nuisances.cross_fit import cross_fit, fit_nuisances
from causurv.nuisances.types import CrossFitNuisances, Nuisances

__all__ = [
    "CrossFitNuisances",
    "Nuisances",
    "cross_fit",
    "fit_nuisances",
    "pseudo_outcomes",
]
