r"""Scorers: a metric plus the prediction it needs, for cross-validation.

Each factory here returns a :class:`~tausurv.model_selection.Scorer`, e.g.
``uno(tau=5.0)``: a function ``(model, X, event_time, event_indicator,
train) -> float`` plus the direction that improves it. ``train`` is the
training :class:`~tausurv.model_selection.Fold`; the IPCW scorers estimate
the censoring distribution on it, never on the test fold.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.metrics import auc, brier, calibration, concordance
from tausurv.model_selection._cross_validate import Scorer
from tausurv.model_selection._fold import Fold
from tausurv.nonparametric import censoring_distribution

CurveFn = Callable[
    [Any, NDArray[Any], NDArray[np.float64], NDArray[Any], Fold], NDArray[np.float64]
]


def harrell() -> Scorer:
    """Harrell's C-index of ``model.predict``."""

    def score(
        model: Any,
        X: NDArray[Any],
        Y: NDArray[np.float64],
        E: NDArray[Any],
        train: Fold,
    ) -> float:
        return concordance.harrell(Y, E, model.predict(X))

    return Scorer(score, name="harrell_c")


def uno(tau: float) -> Scorer:
    """Uno's IPCW C-index of ``model.predict`` up to ``tau``."""

    def score(
        model: Any,
        X: NDArray[Any],
        Y: NDArray[np.float64],
        E: NDArray[Any],
        train: Fold,
    ) -> float:
        G = censoring_distribution(train.event_time, train.event_indicator)
        return concordance.uno(Y, E, model.predict(X), tau=tau, censoring_survival=G)

    return Scorer(score, name="uno_c")


def antolini(times: ArrayLike) -> Scorer:
    """Antolini's time-dependent C-index of the predicted survival on ``times``."""
    grid = np.asarray(times, dtype=np.float64)

    def score(
        model: Any,
        X: NDArray[Any],
        Y: NDArray[np.float64],
        E: NDArray[Any],
        train: Fold,
    ) -> float:
        return concordance.antolini(
            Y, E, model.predict_survival_function(X, grid), grid
        )

    return Scorer(score, name="antolini_c")


def integrated_brier(times: ArrayLike) -> Scorer:
    """Integrated IPCW Brier score of the predicted survival over ``times``."""
    grid = np.asarray(times, dtype=np.float64)

    def score(
        model: Any,
        X: NDArray[Any],
        Y: NDArray[np.float64],
        E: NDArray[Any],
        train: Fold,
    ) -> float:
        G = censoring_distribution(train.event_time, train.event_indicator)
        S = model.predict_survival_function(X, grid)
        return brier.integrated(Y, E, S, grid, censoring_survival=G)

    return Scorer(score, greater_is_better=False, name="integrated_brier")


def d_calibration(times: ArrayLike, *, n_bins: int = 10) -> Scorer:
    """D-calibration p-value of the predicted survival on ``times``."""
    grid = np.asarray(times, dtype=np.float64)

    def score(
        model: Any,
        X: NDArray[Any],
        Y: NDArray[np.float64],
        E: NDArray[Any],
        train: Fold,
    ) -> float:
        S = model.predict_survival_function(X, grid)
        return calibration.distributional(Y, E, S, grid, n_bins=n_bins)

    return Scorer(score, name="d_calibration")


def auc_over_time(times: ArrayLike) -> CurveFn:
    """Uno's dynamic AUC at each of ``times``, a curve rather than a number.

    Returns a bare function: use it with
    :meth:`~tausurv.model_selection.CVResult.evaluate`, and plot the result
    with :func:`tausurv.plot.auc_over_time`.
    """
    grid = np.asarray(times, dtype=np.float64)

    def score(
        model: Any,
        X: NDArray[Any],
        Y: NDArray[np.float64],
        E: NDArray[Any],
        train: Fold,
    ) -> NDArray[np.float64]:
        G = censoring_distribution(train.event_time, train.event_indicator)
        return auc.uno(Y, E, model.predict(X), grid, censoring_survival=G)

    return score


def brier_over_time(times: ArrayLike) -> CurveFn:
    """IPCW Brier score at each of ``times``, a curve rather than a number.

    Returns a bare function for
    :meth:`~tausurv.model_selection.CVResult.evaluate`; plot the result with
    :func:`tausurv.plot.brier_over_time`.
    """
    grid = np.asarray(times, dtype=np.float64)

    def score(
        model: Any,
        X: NDArray[Any],
        Y: NDArray[np.float64],
        E: NDArray[Any],
        train: Fold,
    ) -> NDArray[np.float64]:
        G = censoring_distribution(train.event_time, train.event_indicator)
        S = model.predict_survival_function(X, grid)
        return brier.score(Y, E, S, grid, censoring_survival=G)

    return score
