r"""Cross-validation on the model's own ``fit`` and a scorer contract."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

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
    a bare function is taken to be maximised. ``name`` labels the score's
    column in :attr:`CVResult.scores`.
    """

    fn: ScoreFn
    greater_is_better: bool = True
    name: str = "score"

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
    progress: bool = True,
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
    progress : bool, default True
        Show a progress bar with one step per model fit; a widget in a
        notebook, text in a terminal.

    Returns
    -------
    CVResult
        Scores per fold, one column per scorer named by the dict key or the
        scorer's ``name``, plus ``seconds`` of training time; the fold models
        and the splits. Predicts out of fold on the same ``X``; ``.ensemble``
        predicts on new data.
    """
    splits = _resolve_splits(cv, np.asarray(event_indicator), seed)
    bar = fit_bar(progress, total=len(splits), leave=False)
    result = _cross_validate(
        build,
        X,
        event_time,
        event_indicator,
        scoring=scoring,
        train=train,
        cv=splits,
        seed=seed,
        trial=None,
        bar=bar,
    )
    if bar is not None:
        bar.close()
    return result


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
    bar: Any = None,
) -> CVResult:
    """``bar`` is a fit counter shared with the caller, or ``None``."""
    X_arr = np.asarray(X)
    Y = np.asarray(event_time, dtype=np.float64)
    E = np.asarray(event_indicator)
    scorers = scoring if isinstance(scoring, dict) else {scorer_name(scoring): scoring}

    splits = _resolve_splits(cv, E, seed)
    rows, models = [], []
    for k, (train_idx, test_idx) in enumerate(splits):
        train_fold = Fold(X_arr[train_idx], Y[train_idx], E[train_idx])
        model = build() if trial is None else build(trial)
        started = time.perf_counter()
        _train(train, model, train_fold, trial)
        seconds = time.perf_counter() - started
        if bar is not None:
            bar.update(1)
        row: dict[str, Any] = {"fold": k}
        for name, scorer in scorers.items():
            row[name] = float(
                scorer(model, X_arr[test_idx], Y[test_idx], E[test_idx], train_fold)
            )
        row["seconds"] = seconds
        rows.append(row)
        models.append(model)
    return CVResult(
        scores=pl.DataFrame(rows),
        models=models,
        splits=splits,
        params=[{} for _ in splits],
    )


def fit_bar(progress: bool, *, total: int, leave: bool = True) -> Any:
    """A tqdm bar counting model fits, or ``None`` when ``progress`` is off.

    A widget in a notebook, text in a terminal. Private callers share one
    bar down the call chain so a whole study reports through it.
    """
    if not progress:
        return None
    return tqdm(total=total, desc="fits", leave=leave)


def scorer_name(scoring: Any) -> str:
    """Column name for a scorer: its ``name`` when it has one, else ``score``."""
    return str(getattr(scoring, "name", "score"))


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
