r"""Scorers: a metric plus the prediction it needs, for cross-validation.

Each factory here returns a :class:`~tausurv.model_selection.Scorer`, e.g.
``uno(tau=5.0)``: a function ``(model, X, event_time, event_indicator,
train) -> float`` plus the direction that improves it. ``train`` is the
training :class:`~tausurv.model_selection.Fold`; the IPCW scorers estimate
the censoring distribution on it, never on the test fold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.metrics import brier, calibration, concordance
from tausurv.model_selection._cross_validate import Fold, Scorer
from tausurv.nonparametric import censoring_distribution


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

    return Scorer(score)


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

    return Scorer(score)


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

    return Scorer(score)


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

    return Scorer(score, greater_is_better=False)


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

    return Scorer(score)
