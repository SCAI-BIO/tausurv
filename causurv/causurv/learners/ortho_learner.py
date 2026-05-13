r"""Orthogonal survival learner (Frauen et al., 2025).

A two-stage Neyman-orthogonal HTE learner for survival outcomes. The
first stage cross-fits per-arm event-free survival, per-arm censoring
survival, and propensity nuisances. The second stage regresses the DR
pseudo-outcome :func:`causurv.nuisances.pseudo_outcomes.dr_survival` on
$X$ at each target time, yielding an estimate of
$\tau_t(x) = S_t(x, 1) - S_t(x, 0)$.

The implementation is faithful to the legacy ``CauSurv.models.orthogonal_learner``
math (Frauen et al. 2025, Eq 9 with weighting $f \equiv 1$) but
swaps pytmle for our :func:`causurv.nuisances.cross_fit` and replaces
the Lightning second stage with a pluggable sklearn-style regressor.

v0.1 scope:
- Survival-difference target only (CIF / competing-risks: Appendix F, future).
- DR weighting only (R / C / S / TCS via pluggable $f$: future).
- Discrete integer time grid.

References
----------
Frauen, D., Schröder, M., Hess, K., & Feuerriegel, S. (2025).
*Orthogonal Survival Learners for Estimating Heterogeneous Treatment
Effects from Time-to-Event Data.* arXiv:2505.13072.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from causurv.nuisances.cross_fit import cross_fit
from causurv.nuisances.pseudo_outcomes import PSEUDO_OUTCOME_FOR
from causurv.nuisances.types import CrossFitNuisances
from causurv.predictor import HTEEstimates, HTEPredictor, _validate_fit_inputs


class OrthoLearner(HTEPredictor):
    r"""Two-stage Neyman-orthogonal survival HTE learner.

    Estimates $\tau_t(x) = S_t(x, 1) - S_t(x, 0)$ at each
    ``target_time``. First stage: K-fold cross-fitted nuisances
    (per-arm outcome, per-arm censoring, propensity). Second stage:
    a regressor of the DR pseudo-outcome on $X$, per target time.

    Parameters
    ----------
    outcome_factory : callable
        Zero-argument factory returning a tausurv :class:`SurvivalPredictor`
        for the per-arm event-free survival $\hat S_a(t \mid x)$.
    propensity_factory : callable
        Factory returning an sklearn-style classifier
        (``predict_proba(X) \to (n, 2)``) for $\hat\pi(x) = P(A=1\mid x)$.
    censoring_factory : callable
        Factory returning a :class:`SurvivalPredictor` for the per-arm
        censoring survival $\hat G_a(t \mid x)$.
    second_stage_factory : callable
        Factory returning an sklearn-style regressor (``fit(X, y,
        sample_weight=w)``, ``predict(X)``). One instance is fit per
        ``target_time``.
    target_times : ``(K,)`` array-like
        Time points at which to estimate $\tau_t$. Must be values on
        the cross-fit integer time grid (which spans
        $\{1, \dots, \lceil\max\tau\rceil\}$).
    weighting : ``"DR"``, default ``"DR"``
        Which orthogonal weighting function $f$ to use. Only ``"DR"``
        ($f \equiv 1$) is supported in v0.1; ``"R"`` / ``"C"`` / ``"S"`` /
        ``"TCS"`` are planned future presets.
    n_folds : int, default ``5``
        Cross-fitting folds.
    propensity_clip : float, default ``1e-2``
        Clip $\pi(X)$ to $[\text{clip}, 1 - \text{clip}]$ inside the
        DR pseudo-outcome to bound variance under low treatment overlap.
    seed : int, default ``0``

    Notes
    -----
    The learner estimates the *contrast directly* and does not
    factorise into separate $S_0(t|x)$ / $S_1(t|x)$ predictions —
    :meth:`predict_potential_outcomes` therefore raises. Use
    :meth:`predict_hte` to get the survival-difference estimates.

    Examples
    --------
    >>> import causurv as cs
    >>> from causurv.estimands import SurvivalDiff
    >>> from tausurv.linear import CoxPH
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> ol = cs.OrthoLearner(
    ...     outcome_factory=lambda: CoxPH(),
    ...     propensity_factory=lambda: LogisticRegression(),
    ...     censoring_factory=lambda: CoxPH(),
    ...     second_stage_factory=lambda: RandomForestRegressor(n_estimators=100),
    ...     estimand=SurvivalDiff(times=[5, 10, 20]),
    ...     n_folds=5,
    ... )
    >>> ol.fit(X, T, E, A)
    >>> hte = ol.predict_hte(X_test)
    """

    def __init__(
        self,
        *,
        outcome_factory: Callable[[], Any],
        propensity_factory: Callable[[], Any],
        censoring_factory: Callable[[], Any],
        second_stage_factory: Callable[[], Any],
        estimand: "SurvivalDiff",
        weighting: Literal["DR", "R"] = "DR",
        n_folds: int = 5,
        propensity_clip: float = 1e-2,
        seed: int = 0,
    ) -> None:
        from causurv.estimands import SurvivalDiff

        if weighting not in PSEUDO_OUTCOME_FOR:
            raise ValueError(
                f"weighting must be one of {sorted(PSEUDO_OUTCOME_FOR)}; "
                f"got {weighting!r}. (C/S/TCS variants from Frauen 2025 "
                f"are planned follow-ups.)"
            )
        self._pseudo_fn = PSEUDO_OUTCOME_FOR[weighting]
        if not isinstance(estimand, SurvivalDiff):
            raise TypeError(
                "OrthoLearner v0.1 only supports estimand=SurvivalDiff(...); "
                f"got {type(estimand).__name__}. CIF/RMST variants are "
                "planned follow-ups."
            )
        if (estimand.treatment, estimand.reference) != (1, 0):
            raise ValueError(
                "OrthoLearner v0.1 is binary-treatment with the survival "
                "difference S_t(1|x) - S_t(0|x); estimand must have "
                f"(treatment, reference) = (1, 0). Got "
                f"({estimand.treatment}, {estimand.reference})."
            )
        if (np.asarray(estimand.times) < 1).any():
            raise ValueError(
                "OrthoLearner v0.1 uses an integer discrete-time grid; "
                "estimand.times must all be >= 1"
            )

        self._outcome_factory = outcome_factory
        self._propensity_factory = propensity_factory
        self._censoring_factory = censoring_factory
        self._second_stage_factory = second_stage_factory
        self._estimand = estimand
        self.weighting = weighting
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.seed = seed

        self._cf: CrossFitNuisances | None = None
        self._second_stage: dict[float, Any] = {}

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
        treatment: ArrayLike,
    ) -> "OrthoLearner":
        X, T, E, A = _validate_fit_inputs(X, event_time, event_indicator, treatment)
        n_arms = int(A.max()) + 1
        if n_arms != 2:
            raise ValueError(
                f"OrthoLearner v0.1 is binary-treatment; got {n_arms} arms"
            )

        target_times = np.asarray(self._estimand.times, dtype=np.float64)
        # Cross-fit grid: integer 1..max(target_times). The discrete-time
        # DR formula sums over i = 1..t for each target_time t.
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

        # Second stage: one regressor per target time.
        self._second_stage = {}
        for t in target_times:
            pseudo, weight = self._pseudo_fn(
                self._cf,
                event_time=T,
                event_indicator=E,
                treatment=A,
                target_time=float(t),
                propensity_clip=self.propensity_clip,
            )
            model = self._second_stage_factory()
            self._fit_second_stage(model, X, pseudo, weight)
            self._second_stage[float(t)] = model

        self._fit_X = X
        return self

    @staticmethod
    def _fit_second_stage(
        model: Any,
        X: NDArray[np.float64],
        pseudo: NDArray[np.float64],
        weight: NDArray[np.float64],
    ) -> None:
        """Fit ``model`` on ``(X, pseudo)``, passing ``weight`` if supported.

        DR weighting yields a constant weight of 1, so the sample-weight
        path doesn't matter for v0.1 — but the second stage is written
        weight-aware so future R/C/S/TCS variants drop in cleanly.
        """
        # All sample weights equal 1 under DR; only pass sample_weight if
        # the regressor accepts it and weights are non-trivial.
        if np.allclose(weight, 1.0):
            model.fit(X, pseudo)
        else:
            model.fit(X, pseudo, sample_weight=weight)

    def predict_potential_outcomes(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
        *,
        cause: int | None = None,
    ):
        raise NotImplementedError(
            "OrthoLearner estimates the survival-difference contrast "
            "tau_t(x) directly via a Neyman-orthogonal second-stage "
            "regression; per-arm potential outcomes S_0(t|x), S_1(t|x) "
            "are not separately recovered. Use predict_hte() for the "
            "survival-difference estimate."
        )

    def predict_hte(self, X: ArrayLike) -> HTEEstimates:
        """Compute the survival-difference HTE bound at construction.

        OrthoLearner's estimand is fixed at construction (the
        pseudo-outcome formula is estimand-specific). For different
        estimands, construct another OrthoLearner.
        """
        if self._cf is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() first")
        X_arr = np.asarray(X, dtype=np.float64)
        target_times = np.asarray(self._estimand.times, dtype=np.float64)
        values = np.empty((X_arr.shape[0], len(target_times)), dtype=np.float64)
        for j, t in enumerate(target_times):
            values[:, j] = self._second_stage[float(t)].predict(X_arr)
        return HTEEstimates(values=values, estimand=self._estimand)

    def predict_ate(self, X: ArrayLike | None = None) -> NDArray[np.float64]:
        """Mean of :meth:`predict_hte` over ``X`` (or training set)."""
        if X is None:
            X = self._stored_X()
        return self.predict_hte(X).values.mean(axis=0)


class DRLearner(OrthoLearner):
    r"""DR-survival learner — :class:`OrthoLearner` with $f \equiv 1$ baked in.

    Convenience alias for the most-used special case. All parameters
    forward to :class:`OrthoLearner` except ``weighting`` (fixed to
    ``"DR"``).
    """

    def __init__(
        self,
        *,
        outcome_factory: Callable[[], Any],
        propensity_factory: Callable[[], Any],
        censoring_factory: Callable[[], Any],
        second_stage_factory: Callable[[], Any],
        estimand: "SurvivalDiff",
        n_folds: int = 5,
        propensity_clip: float = 1e-2,
        seed: int = 0,
    ) -> None:
        super().__init__(
            outcome_factory=outcome_factory,
            propensity_factory=propensity_factory,
            censoring_factory=censoring_factory,
            second_stage_factory=second_stage_factory,
            estimand=estimand,
            weighting="DR",
            n_folds=n_folds,
            propensity_clip=propensity_clip,
            seed=seed,
        )


class RLearner(OrthoLearner):
    r"""R-survival learner — :class:`OrthoLearner` with $f = \pi(1-\pi)$.

    The R-weighting downweights low-treatment-overlap samples (the
    paper's "T-learner" in their overlap-type naming). Mathematically
    equivalent to the residualised Robinson-style loss
    $\mathbb{E}[(\tilde Y - \tilde A\,g(X))^2]$ from paper Eq 14, cast
    as a weighted MSE for sklearn-compatible second-stage regressors.

    Less sensitive than DR to extreme propensities but still sensitive
    to low censoring or survival overlap — see the paper's
    :class:`Survival-C-` / :class:`-S-` / :class:`-TCS-learner` variants
    for further-robust weightings (planned follow-ups).
    """

    def __init__(
        self,
        *,
        outcome_factory: Callable[[], Any],
        propensity_factory: Callable[[], Any],
        censoring_factory: Callable[[], Any],
        second_stage_factory: Callable[[], Any],
        estimand: "SurvivalDiff",
        n_folds: int = 5,
        propensity_clip: float = 1e-2,
        seed: int = 0,
    ) -> None:
        super().__init__(
            outcome_factory=outcome_factory,
            propensity_factory=propensity_factory,
            censoring_factory=censoring_factory,
            second_stage_factory=second_stage_factory,
            estimand=estimand,
            weighting="R",
            n_folds=n_folds,
            propensity_clip=propensity_clip,
            seed=seed,
        )
