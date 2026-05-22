r"""S-learner: single outcome model with treatment as an extra feature.

Fits one base survival model on the augmented feature matrix
``[X | A]``, then predicts potential outcomes by querying the same
model with the treatment column set to each arm value:

$$
S_a(t \mid x) = \hat S\big(t \mid [x, a]\big).
$$

S-learner is the simplest meta-learner. It works with any base model
satisfying the tausurv :class:`SurvivalPredictor` contract (CoxPH, AFT,
RSF, DeepSurv, DeepHit, DSM, HACSurv, …). It is **not doubly robust**
and **does not handle confounding** beyond what the base model's
functional form permits — sensitive to the treatment effect being
absorbed into the encoder rather than the treatment coefficient.

Use S-learner as a baseline; reach for T-learner, X-learner, or DR-learner
for stronger estimates.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from causurv.predictor import HTEPredictor, _validate_fit_inputs
from tausurv.predictor import SurvivalPredictor


class SLearner(HTEPredictor):
    r"""S-learner — single outcome model with treatment as a feature.

    Parameters
    ----------
    outcome_factory : ``Callable[[], object]``
        Zero-argument factory returning a fresh **sklearn-style**
        survival model — an object exposing ``fit(X, T, E) -> self``
        and the standard :class:`SurvivalPredictor` predict API.
        Works directly with :class:`tausurv.linear.CoxPH`,
        :class:`WeibullAFT`/`LogNormalAFT`/`LogLogisticAFT`,
        :class:`FineGray`, :class:`SurvivalTree`,
        :class:`RandomSurvivalForest`.

        Neural tausurv models (DeepSurv, DeepHit, LogisticHazard,
        DSM, CopulaSurv, HACSurv) train via the :mod:`tausurv.nn`
        ``Trainer`` rather than a sklearn-style ``.fit``; wrap them
        in a fittable adapter to use as S-learner bases. A
        ``causurv.adapters.*`` module providing those wrappers is
        planned but not yet shipped.

        The model's ``in_features`` must accommodate ``X.shape[1] + 1``
        features — the extra column is the treatment indicator.

    Attributes
    ----------
    times_ : ``(T,)`` array
        Inherited from the base model after fit.

    Examples
    --------
    Binary treatment, Cox base::

        from tausurv.linear import CoxPH
        sl = SLearner(outcome_factory=lambda: CoxPH())
        sl.fit(X, T, E, A)
        hte = sl.predict_hte(X_test)

    Competing risks, Fine-Gray base::

        from tausurv.linear import FineGray
        sl = SLearner(outcome_factory=lambda: FineGray(cause=1))
        sl.fit(X, T, E, A)
        cif_diff = sl.predict_hte(
            X_test, contrast="cif_diff", cause=1
        )
    """

    def __init__(self, *, outcome_factory: Callable[[], Any]) -> None:
        self._make = outcome_factory
        self._model: SurvivalPredictor | None = None
        self._n_arms: int = 0

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
        treatment: ArrayLike,
    ) -> "SLearner":
        X, event_time, event_indicator, A = _validate_fit_inputs(
            X, event_time, event_indicator, treatment
        )
        self._n_arms = int(A.max()) + 1

        XA = np.hstack([X, A.reshape(-1, 1).astype(X.dtype)])
        self._model = self._make().fit(XA, event_time, event_indicator)

        # Inherit times_ if the base has it; some bases (DeepSurv pre-baseline)
        # raise on access — be permissive.
        if hasattr(self._model, "times_"):
            try:
                self.times_ = np.asarray(
                    self._model.times_, dtype=np.float64
                )
            except (RuntimeError, AttributeError):
                pass

        self._fit_X = X
        return self

    def predict_potential_outcomes(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
        *,
        cause: int | None = None,
    ) -> tuple[NDArray[np.float64], ...]:
        if self._model is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() first")
        X = np.asarray(X, dtype=np.float64)
        outs = []
        for a in range(self._n_arms):
            XA = np.hstack(
                [X, np.full((X.shape[0], 1), a, dtype=X.dtype)]
            )
            if cause is None:
                out = self._model.predict_survival_function(XA, times)
            else:
                out = self._model.predict_cif(XA, times, cause=cause)
            outs.append(np.asarray(out, dtype=np.float64))
        return tuple(outs)
