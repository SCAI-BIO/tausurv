r"""Causal survival forest (Cui et al., 2023) — GRF-style HTE learner.

Composes the orthogonal-survival nuisance pipeline (Frauen et al. 2025
DR pseudo-outcome) with our Rust gradient forest as the second stage,
predicting via forest weights — Athey-Tibshirani-Wager (2019).

The architecture distinguishes :class:`CausalSurvivalForest` from
:class:`causurv.learners.OrthoLearner`:

- Same first stage (cross-fitted per-arm outcome + propensity + censoring
  nuisances, DR pseudo-outcome).
- Different second stage: an honest partition-only forest fit on
  ``(X, pseudo)`` rather than an arbitrary sklearn regressor.
- Different prediction: forest weights over training pseudo-outcomes
  rather than calling ``regressor.predict``. This is the substrate
  underlying GRF's asymptotic-normality and CI results.

References
----------
Cui, Y., Kosorok, M. R., Sverdrup, E., Wager, S., Zhu, R. (2023).
*Estimating Heterogeneous Treatment Effects with Right-Censored Data
via Causal Survival Forests.* JASA 118(541).

Athey, S., Tibshirani, J., Wager, S. (2019). *Generalized Random
Forests.* Annals of Statistics 47(2).

Frauen, D., Schröder, M., Hess, K., & Feuerriegel, S. (2025).
*Orthogonal Survival Learners for Estimating Heterogeneous Treatment
Effects from Time-to-Event Data.* arXiv:2505.13072.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from causurv.nuisances.cross_fit import cross_fit
from causurv.nuisances.pseudo_outcomes import PSEUDO_OUTCOME_FOR
from causurv.nuisances.types import CrossFitNuisances
from causurv.predictor import HTEEstimates, HTEPredictor, _validate_fit_inputs
from tausurv.core import GradientForest, fit_gradient_forest

if TYPE_CHECKING:
    from causurv.estimands import SurvivalDiff


class CausalSurvivalForest(HTEPredictor):
    r"""GRF-style causal survival forest.

    Estimates $\tau_t(x) = S_t(x, 1) - S_t(x, 0)$ at each target time
    using a two-stage pipeline:

    1. **Cross-fitted nuisances**: per-arm event-free survival
       $\hat S_a(t \mid x)$, per-arm censoring $\hat G_a(t \mid x)$, and
       propensity $\hat \pi(x)$.
    2. **Honest gradient forest** fit on the per-sample DR pseudo-outcome
       (Frauen 2025 Eq 10). Prediction is the forest-weighted average of
       training pseudo-outcomes (Athey-Tibshirani-Wager 2019).

    Parameters
    ----------
    outcome_factory, propensity_factory, censoring_factory : callable
        Zero-argument factories for the nuisance models. Mirror
        :class:`causurv.learners.OrthoLearner`.
    estimand : :class:`SurvivalDiff`
        Survival-difference estimand with integer-valued ``times``.
    weighting : ``"DR"``, default ``"DR"``
        Pseudo-outcome variant. Only ``"DR"`` is supported in v0.1.
    n_trees : int, default ``500``
    min_samples_leaf : int, default ``5``
    max_features : int or ``"sqrt"`` or ``"all"`` or ``None``, default ``"sqrt"``
    bootstrap : bool, default ``False``
        ``False`` (sampling without replacement) is required for GRF's
        asymptotic theory. Bootstrap exposed only for ablation.
    subsample_fraction : float, default ``0.5``
        Fraction of training rows per tree (without-replacement
        subsample). grf default is ``0.5``.
    honesty_fraction : float, default ``0.5``
        Splitting/estimation split per tree.
    n_folds : int, default ``5``
        Cross-fit folds.
    propensity_clip : float, default ``1e-2``
    seed : int, default ``0``

    Notes
    -----
    Estimates the contrast directly — :meth:`predict_potential_outcomes`
    raises. Use :meth:`predict_hte` for per-subject survival differences.
    """

    def __init__(
        self,
        *,
        outcome_factory: Callable[[], Any],
        propensity_factory: Callable[[], Any],
        censoring_factory: Callable[[], Any],
        estimand: "SurvivalDiff",
        weighting: Literal["DR"] = "DR",
        n_trees: int = 500,
        min_samples_leaf: int = 5,
        max_features: int | Literal["sqrt", "all"] | None = "sqrt",
        bootstrap: bool = False,
        subsample_fraction: float = 0.5,
        honesty_fraction: float = 0.5,
        n_folds: int = 5,
        propensity_clip: float = 1e-2,
        seed: int = 0,
    ) -> None:
        from causurv.estimands import SurvivalDiff

        if weighting not in PSEUDO_OUTCOME_FOR:
            raise ValueError(
                f"weighting must be one of {sorted(PSEUDO_OUTCOME_FOR)}; "
                f"got {weighting!r}."
            )
        self._pseudo_fn = PSEUDO_OUTCOME_FOR[weighting]
        if not isinstance(estimand, SurvivalDiff):
            raise TypeError(
                "CausalSurvivalForest v0.1 only supports estimand=SurvivalDiff(...); "
                f"got {type(estimand).__name__}."
            )
        if (estimand.treatment, estimand.reference) != (1, 0):
            raise ValueError(
                "CausalSurvivalForest v0.1 is binary-treatment with the "
                "survival difference S_t(1|x) - S_t(0|x); estimand must "
                f"have (treatment, reference) = (1, 0). "
                f"Got ({estimand.treatment}, {estimand.reference})."
            )
        if (np.asarray(estimand.times) < 1).any():
            raise ValueError(
                "CausalSurvivalForest v0.1 uses an integer discrete-time "
                "grid; estimand.times must all be >= 1"
            )

        self._outcome_factory = outcome_factory
        self._propensity_factory = propensity_factory
        self._censoring_factory = censoring_factory
        self._estimand = estimand
        self.weighting = weighting
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.subsample_fraction = subsample_fraction
        self.honesty_fraction = honesty_fraction
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.seed = seed

        self._cf: CrossFitNuisances | None = None
        self._forests: dict[float, GradientForest] = {}
        self._pseudo: dict[float, NDArray[np.float64]] = {}

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
        treatment: ArrayLike,
    ) -> "CausalSurvivalForest":
        X, T, E, A = _validate_fit_inputs(X, event_time, event_indicator, treatment)
        n_arms = int(A.max()) + 1
        if n_arms != 2:
            raise ValueError(
                f"CausalSurvivalForest v0.1 is binary-treatment; got {n_arms} arms"
            )

        target_times = np.asarray(self._estimand.times, dtype=np.float64)
        t_max = int(np.ceil(target_times.max()))
        time_grid = np.arange(1, t_max + 1, dtype=np.float64)

        self._cf = cross_fit(
            X, T, E, A,
            outcome_factory=self._outcome_factory,
            propensity_factory=self._propensity_factory,
            censoring_factory=self._censoring_factory,
            times=time_grid,
            n_folds=self.n_folds,
            per_arm_censoring=True,
            stratify_by_arm=True,
            seed=self.seed,
        )
        self.times_ = target_times

        X_f = np.asfortranarray(X.astype(np.float64))
        self._forests = {}
        self._pseudo = {}
        for j, t in enumerate(target_times):
            pseudo, _weight = self._pseudo_fn(
                self._cf,
                event_time=T,
                event_indicator=E,
                treatment=A,
                target_time=float(t),
                propensity_clip=self.propensity_clip,
            )
            pseudo = np.ascontiguousarray(pseudo, dtype=np.float64)
            # Derive a stable per-time seed so different target times
            # produce independent (but reproducible) forests.
            forest = fit_gradient_forest(
                X_f,
                pseudo,
                n_trees=self.n_trees,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                subsample_fraction=self.subsample_fraction,
                honesty=True,
                honesty_fraction=self.honesty_fraction,
                seed=self.seed * 1_000 + j,
            )
            self._forests[float(t)] = forest
            self._pseudo[float(t)] = pseudo

        self._fit_X = X_f
        return self

    def predict_potential_outcomes(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
        *,
        cause: int | None = None,
    ):
        raise NotImplementedError(
            "CausalSurvivalForest estimates the survival-difference "
            "contrast tau_t(x) directly via a forest-weighted DR "
            "pseudo-outcome; per-arm potential outcomes are not "
            "separately recovered. Use predict_hte() for the "
            "survival-difference estimate."
        )

    def predict_hte(self, X: ArrayLike) -> HTEEstimates:
        r"""Forest-weighted survival-difference HTE.

        For each target time $t$, computes the GRF prediction
        $\hat\tau_t(x) = \sum_i w_i(x) \cdot \mathrm{pseudo}_i^{(t)}$
        where the forest weights are taken at the target time's forest.
        """
        if self._cf is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() first")
        X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        X_f = np.asfortranarray(X_arr)
        target_times = np.asarray(self._estimand.times, dtype=np.float64)
        values = np.empty((X_arr.shape[0], len(target_times)), dtype=np.float64)
        for j, t in enumerate(target_times):
            forest = self._forests[float(t)]
            w = forest.forest_weights(X_f)            # (n_query, n_train)
            values[:, j] = w @ self._pseudo[float(t)]
        return HTEEstimates(values=values, estimand=self._estimand)

    def predict_ate(self, X: ArrayLike | None = None) -> NDArray[np.float64]:
        """Mean of :meth:`predict_hte` over ``X`` (or training set)."""
        if X is None:
            if self._cf is None:
                raise RuntimeError(f"{type(self).__name__}: call fit() first")
            X = self._fit_X
        return self.predict_hte(X).values.mean(axis=0)
