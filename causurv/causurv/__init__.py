"""Causal survival analysis: per-subject treatment effects on time-to-event outcomes.

``causurv`` adds a heterogeneous-treatment-effect (HTE / CATE) layer on top
of :mod:`tausurv`. Estimands and contracts live in :mod:`causurv.predictor`
and :mod:`causurv.contrasts`. Learners are in :mod:`causurv.learners`; the
nuisance carriers and K-fold cross-fitting helpers in
:mod:`causurv.nuisances`; PEHE-family metrics in :mod:`causurv.metrics`;
paper-reproduction simulators in :mod:`causurv.simulations`. Neural-network
components — encoder, treatment heads, IPM, losses, models — live under
:mod:`causurv.nn`.

The headline names — the predictor contract, the result and nuisance
types, and the concrete learners — are re-exported at the package root:

>>> import causurv as cs
>>> sl = cs.SLearner(outcome_factory=...).fit(X, T, E, A)
>>> xf = cs.cross_fit(X, T, E, A, outcome_factory=..., propensity_factory=...)
"""

from causurv import contrasts, estimands, learners, metrics, nn, nuisances, simulations
from causurv.learners import (
    CausalSurvivalForest,
    DRLearner,
    OrthoLearner,
    RLearner,
    SLearner,
    SurvITE,
    TLearner,
)
from causurv.nuisances import (
    CrossFitNuisances,
    Nuisances,
    cross_fit,
    fit_nuisances,
)
from causurv.predictor import HTEEstimates, HTEPredictor

__version__ = "0.0.0"

__all__ = [
    "CausalSurvivalForest",
    "CrossFitNuisances",
    "DRLearner",
    "HTEEstimates",
    "HTEPredictor",
    "Nuisances",
    "OrthoLearner",
    "RLearner",
    "SLearner",
    "SurvITE",
    "TLearner",
    "contrasts",
    "cross_fit",
    "estimands",
    "fit_nuisances",
    "learners",
    "metrics",
    "nn",
    "nuisances",
    "simulations",
]
