r"""Cross-validation on the model's own ``fit`` and a scorer contract."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from tausurv.model_selection._fold import Fold
from tausurv.model_selection._results import CVResult
from tausurv.model_selection._split import stratified_folds

ScoreFn = Callable[[Any, NDArray[Any], NDArray[np.float64], NDArray[Any], Fold], float]


@dataclass(frozen=True)
class Scorer:
    """A score function and the direction that improves it.

    ``fn(model, X, event_time, event_indicator, train) -> float`` is
    evaluated on a test fold with ``train`` the training :class:`Fold`. Wrap
    a custom function whose lower values are better in
    ``Scorer(fn, greater_is_better=False)`` so :func:`tune` minimises it;
    a bare function is taken to be maximised.
    """

    fn: ScoreFn
    greater_is_better: bool = True

    def __call__(
        self,
        model: Any,
        X: NDArray[Any],
        event_time: NDArray[np.float64],
        event_indicator: NDArray[Any],
        train: Fold,
    ) -> float:
        return float(self.fn(model, X, event_time, event_indicator, train))


Build = Callable[..., Any]
Train = Callable[..., None]
Scoring = Scorer | ScoreFn
Splits = int | Iterable[tuple[ArrayLike, ArrayLike]]


def cross_validate(
    build: Build,
    X: ArrayLike,
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    *,
    scoring: Scoring | dict[str, Scoring],
    train: Train | None = None,
    cv: Splits = 5,
    seed: int | None = 0,
) -> CVResult:
    r"""Fit a fresh model per fold and score it on the held-out fold.

    Parameters
    ----------
    build : callable
        ``build(trial=None) -> model``. Called once per fold with no
        arguments here; :func:`tune` passes the Optuna trial.
    X : (n, d) array_like
    event_time : (n,) array_like
    event_indicator : (n,) array_like
    scoring : scorer or dict of scorers
        A :class:`Scorer`, or any function ``(model, X, event_time,
        event_indicator, train) -> float`` evaluated on the test fold with
        ``train`` the training :class:`Fold`, which IPCW scorers use for the
        censoring distribution. See :mod:`tausurv.model_selection.scoring`
        for the built-in ones.
    train : callable, optional
        ``train(model, X, event_time, event_indicator, trial=None)`` fits the
        model in place. Defaults to ``model.fit(X, event_time,
        event_indicator)``.
    cv : int or iterable of (train_idx, test_idx), default 5
        Number of stratified folds from :func:`stratified_folds`, or explicit
        index pairs.
    seed : int, optional
        Seed for the folds.

    Returns
    -------
    CVResult
        Scores per fold (one column per scorer, ``score`` for a single one),
        the fold models and the splits. Predicts out of fold on the same
        ``X``; ``.ensemble`` predicts on new data.
    """
    return _cross_validate(
        build,
        X,
        event_time,
        event_indicator,
        scoring=scoring,
        train=train,
        cv=cv,
        seed=seed,
        trial=None,
    )


def _cross_validate(
    build: Build,
    X: ArrayLike,
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    *,
    scoring: Scoring | dict[str, Scoring],
    train: Train | None,
    cv: Splits,
    seed: int | None,
    trial: Any,
) -> CVResult:
    X_arr = np.asarray(X)
    Y = np.asarray(event_time, dtype=np.float64)
    E = np.asarray(event_indicator)
    scorers = scoring if isinstance(scoring, dict) else {"score": scoring}

    splits = _resolve_splits(cv, E, seed)
    rows, models = [], []
    for k, (train_idx, test_idx) in enumerate(splits):
        train_fold = Fold(X_arr[train_idx], Y[train_idx], E[train_idx])
        model = build() if trial is None else build(trial)
        _train(train, model, train_fold, trial)
        row: dict[str, Any] = {"fold": k}
        for name, scorer in scorers.items():
            row[name] = float(
                scorer(model, X_arr[test_idx], Y[test_idx], E[test_idx], train_fold)
            )
        rows.append(row)
        models.append(model)
    return CVResult(
        scores=pl.DataFrame(rows),
        models=models,
        splits=splits,
        params=[{} for _ in splits],
    )


def _train(train: Train | None, model: Any, fold: Fold, trial: Any) -> None:
    if train is None:
        model.fit(fold.X, fold.event_time, fold.event_indicator)
    elif trial is None:
        train(model, fold.X, fold.event_time, fold.event_indicator)
    else:
        train(model, fold.X, fold.event_time, fold.event_indicator, trial)


def _resolve_splits(
    cv: Splits, event_indicator: NDArray[Any], seed: int | None
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    if isinstance(cv, int):
        return stratified_folds(event_indicator, cv, seed=seed)
    return [
        (np.asarray(tr, dtype=np.intp), np.asarray(te, dtype=np.intp)) for tr, te in cv
    ]
