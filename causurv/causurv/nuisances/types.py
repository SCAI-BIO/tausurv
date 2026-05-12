r"""Nuisance carrier types: :class:`Nuisances` and :class:`CrossFitNuisances`.

Three nuisance functions arise in causal-survival estimation:

- **Outcome** $\mu_a(t \mid x) = \mathbb{E}[Y(a) \mid X = x]$ — modelled
  per arm as a tausurv :class:`SurvivalPredictor`.
- **Propensity** $\pi(x) = P(A \mid x)$ — a multi-class classifier
  satisfying the sklearn convention ``predict_proba(X) \to (n, K)``.
- **Censoring** $G(t \mid x) = P(C > t \mid X = x)$ — a tausurv
  :class:`SurvivalPredictor` fit with the censoring indicator as the
  "event". Optional: only used by IPCW/DR-family learners.

These types are *carriers*: they hold fitted models. They do not fit
themselves. See :func:`causurv.nuisances.fit_nuisances` for fitting on
full data and :func:`causurv.nuisances.cross_fit` for K-fold
cross-fitting (required by orthogonal/DR learners).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tausurv.predictor import SurvivalPredictor


@dataclass(kw_only=True)
class Nuisances:
    r"""Bundle of fitted nuisance models.

    Attributes
    ----------
    outcome : ``dict[int, SurvivalPredictor]``
        Per-arm conditional survival models $\hat S_a(t \mid x)$, keyed
        by treatment-label value.
    propensity : object
        Treatment-assignment model $\pi(x) = P(A \mid x)$ — any sklearn
        classifier exposing ``predict_proba(X) -> (n, n_arms)``. No
        Protocol yet because there is only one concrete consumer
        (sklearn).
    censoring : :class:`SurvivalPredictor` or ``None``
        Censoring distribution model $\hat G(t \mid x)$. Optional.
    """

    outcome: dict[int, SurvivalPredictor]
    propensity: Any  # sklearn classifier with predict_proba
    censoring: SurvivalPredictor | None = None


@dataclass(kw_only=True)
class CrossFitNuisances(Nuisances):
    r"""Nuisances bundled with cross-fit out-of-fold predictions.

    Subclasses :class:`Nuisances`, so any learner that just needs
    fitted models (S/T/X-learner inference paths) accepts this without
    knowing about cross-fitting. Orthogonal/DR learners further consume
    the OOF arrays for the unbiased final-stage fit
    (Chernozhukov et al., 2018).

    Attributes
    ----------
    oof_outcome : ``dict[int, (n, T) array]``
        Out-of-fold survival predictions per arm — for subject $i$ at
        time $t_k$, the prediction is from the fold that did *not*
        train on subject $i$. Aligned to ``times``.
    oof_propensity : ``(n, n_arms)`` array
        Out-of-fold propensity scores summing to 1 along the arm axis.
    oof_censoring : ``(n, T)`` array or ``None``
        Out-of-fold censoring survival $\hat G(t \mid x_i)$ on
        ``times``. ``None`` when ``censoring_factory`` was not supplied.
    times : ``(T,)`` array
        Common time grid the OOF outcome/censoring predictions live on.
    fold_assignment : ``(n,)`` array
        Integer fold index per subject. Downstream code can use this
        for per-fold diagnostics or for variance estimation.
    n_folds : int
    """

    oof_outcome: dict[int, NDArray[np.float64]]
    oof_propensity: NDArray[np.float64]
    oof_censoring: NDArray[np.float64] | None = None
    times: NDArray[np.float64]
    fold_assignment: NDArray[np.int_]
    n_folds: int
