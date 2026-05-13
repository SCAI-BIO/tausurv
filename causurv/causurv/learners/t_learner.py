r"""T-learner: one outcome model per treatment arm.

Fits $K$ **independent** base models — one per unique value of $A$ — on
the arm's subset of the data. Predicts potential outcomes by querying
every model on the same $X$:

$$
S_a(t \mid x) = \hat S_a\big(t \mid x\big), \quad a \in \{0, 1, \dots, K-1\}.
$$

Structurally the opposite trade-off from S-learner:

- S-learner trains one model with treatment as a feature — treatment
  signal can be smeared out across feature interactions.
- T-learner trains separate models — each arm gets its own representation,
  but sees only its own data.

T-learner is the simplest "respect-the-treatment" meta-learner. Not
doubly robust; sensitive to arm-imbalance (rare arms get tiny training
sets). Reach for X-learner or DR-learner when you need to share strength
across arms or want robustness.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from causurv.predictor import HTEPredictor, _validate_fit_inputs


class TLearner(HTEPredictor):
    r"""T-learner — one outcome model per treatment arm.

    Parameters
    ----------
    outcome_factory : ``Callable[[], object]``
        Zero-argument factory returning a fresh **sklearn-style**
        survival model. Called once per unique arm in ``A`` at fit
        time; each returned model is trained on that arm's subset.

        Works directly with :class:`tausurv.linear.CoxPH`,
        :class:`WeibullAFT`/`LogNormalAFT`/`LogLogisticAFT`,
        :class:`FineGray`, :class:`SurvivalTree`,
        :class:`RandomSurvivalForest`. For neural tausurv models
        wrap them with an adapter (planned
        ``causurv.adapters.*``).

        Note that — unlike S-learner — the base model is *not* given a
        treatment column. The factory should produce a model whose
        ``in_features`` matches ``X.shape[1]`` directly.

    Attributes
    ----------
    times_ : ``(T,)`` array
        Union of every arm-specific model's ``times_``. Use as the
        default time grid for ``predict_*`` so both arms are evaluated
        on a shared axis.

    Examples
    --------
    Binary treatment, Cox per arm::

        from tausurv.linear import CoxPH
        tl = TLearner(outcome_factory=lambda: CoxPH())
        tl.fit(X, T, E, A)
        hte = tl.predict_hte(X_test)

    Multi-arm extension (no API change)::

        tl = TLearner(outcome_factory=lambda: CoxPH())
        tl.fit(X, T, E, A_3)         # A_3 ∈ {0, 1, 2}
        arms = tl.predict_potential_outcomes(X_test)  # tuple of length 3
        hte_1v0 = tl.predict_hte(X_test, treatment=1, reference=0)
        hte_2v1 = tl.predict_hte(X_test, treatment=2, reference=1)
    """

    def __init__(self, *, outcome_factory: Callable[[], Any]) -> None:
        self._make = outcome_factory
        self._models: dict[int, Any] = {}

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
        treatment: ArrayLike,
    ) -> "TLearner":
        X, T, E, A = _validate_fit_inputs(X, event_time, event_indicator, treatment)

        self._models = {}
        for a in np.unique(A):
            mask = A == a
            arm_X, arm_T, arm_E = X[mask], T[mask], E[mask]
            self._models[int(a)] = self._make().fit(arm_X, arm_T, arm_E)

        # Union of per-arm times for the default predict grid.
        per_arm_times: list[NDArray[np.float64]] = []
        for m in self._models.values():
            if hasattr(m, "times_"):
                try:
                    per_arm_times.append(np.asarray(m.times_, dtype=np.float64))
                except (RuntimeError, AttributeError):
                    pass
        if per_arm_times:
            self.times_ = np.unique(np.concatenate(per_arm_times))

        self._fit_X = X
        return self

    def predict_potential_outcomes(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
        *,
        cause: int | None = None,
    ) -> tuple[NDArray[np.float64], ...]:
        if not self._models:
            raise RuntimeError(f"{type(self).__name__}: call fit() first")
        X = np.asarray(X, dtype=np.float64)
        times_arr = self._resolve_times(times) if times is None else np.asarray(times, dtype=np.float64)

        outs = []
        for a in sorted(self._models):
            m = self._models[a]
            if cause is None:
                out = m.predict_survival_function(X, times_arr)
            else:
                out = m.predict_cif(X, times_arr, cause=cause)
            outs.append(np.asarray(out, dtype=np.float64))
        return tuple(outs)
