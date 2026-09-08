r"""Bayesian hyperparameter search and nested cross-validation on top of Optuna."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from tausurv.model_selection._cross_validate import (
    Build,
    Fold,
    Scoring,
    Splits,
    Train,
    _cross_validate,
    _resolve_splits,
    _train,
)
from tausurv.model_selection._results import CVResult, TuneResult, trials_table

Direction = Literal["maximize", "minimize"]


def tune(
    build: Build,
    X: ArrayLike,
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    *,
    scoring: Scoring,
    train: Train | None = None,
    cv: Splits = 3,
    n_trials: int = 30,
    direction: Direction | None = None,
    seed: int | None = 0,
    refit: bool = True,
    storage: str | None = None,
    study_name: str | None = None,
    verbose: bool = False,
) -> TuneResult:
    r"""Bayesian search over the parameters ``build`` and ``train`` sample.

    Each trial calls ``build(trial)``, trains with ``train(model, X,
    event_time, event_indicator, trial)`` and scores by cross-validation;
    the objective is the mean fold score. Hyperparameters are declared where
    they are used, with Optuna's ``trial.suggest_*``; nothing is declared
    here. Requires ``optuna`` (``pip install 'tausurv[tune]'``).

    Parameters
    ----------
    build, X, event_time, event_indicator, train, cv, seed
        As in :func:`cross_validate`. ``build`` and ``train`` receive the
        trial as their last argument.
    scoring : scorer
        A single scorer; see :mod:`tausurv.model_selection.scoring`.
    n_trials : int, default 30
    direction : {"maximize", "minimize"}, optional
        Defaults to the scorer's ``greater_is_better``; a bare function is
        maximised.
    refit : bool, default True
        Build and train a model with the best parameters on all the data.
    storage, study_name : str, optional
        Optuna storage URL, e.g. ``"sqlite:///study.db"``, and the study's
        name in it. A search with a storage resumes where it stopped.
    verbose : bool, default False
        Show Optuna's per-trial log.

    Returns
    -------
    TuneResult
    """
    optuna = _import_optuna()
    X_arr = np.asarray(X)
    Y = np.asarray(event_time, dtype=np.float64)
    E = np.asarray(event_indicator)
    splits = _resolve_splits(cv, E, seed)

    def objective(trial: Any) -> float:
        result = _cross_validate(
            build,
            X_arr,
            Y,
            E,
            scoring=scoring,
            train=train,
            cv=splits,
            seed=seed,
            trial=trial,
        )
        return float(np.mean(result.scores["score"].to_numpy()))

    study = optuna.create_study(
        direction=direction or _direction(scoring),
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        study_name=study_name,
        load_if_exists=storage is not None,
    )
    with _OptunaVerbosity(optuna, verbose):
        study.optimize(objective, n_trials=n_trials)

    model = None
    if refit:
        model = build(study.best_trial)
        _train(train, model, Fold(X_arr, Y, E), study.best_trial)
    return TuneResult(
        params=dict(study.best_params),
        score=float(study.best_value),
        model=model,
        trials=trials_table(study),
        study=study,
    )


def nested_cv(
    build: Build,
    X: ArrayLike,
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    *,
    scoring: Scoring,
    train: Train | None = None,
    outer: Splits = 5,
    inner: Splits = 3,
    n_trials: int = 30,
    direction: Direction | None = None,
    seed: int | None = 0,
    verbose: bool = False,
) -> CVResult:
    r"""Tune inside every outer training fold, score on the outer test fold.

    The outer scores are an unbiased estimate of the tuned model's
    performance because the test fold never touches the search. Requires
    ``optuna``.

    Parameters
    ----------
    build, X, event_time, event_indicator, scoring, train, n_trials,
    direction, seed, verbose
        As in :func:`tune`.
    outer : int or iterable of (train_idx, test_idx), default 5
        Outer folds.
    inner : int or iterable of (train_idx, test_idx), default 3
        Folds of the search inside each outer training fold. Explicit
        index pairs are relative to the outer training fold.

    Returns
    -------
    CVResult
        Outer scores, the tuned model of each outer fold, the outer splits,
        the parameters chosen per fold and the inner searches.
    """
    X_arr = np.asarray(X)
    Y = np.asarray(event_time, dtype=np.float64)
    E = np.asarray(event_indicator)

    splits = _resolve_splits(outer, E, seed)
    rows, models, params, trials, studies = [], [], [], [], []
    for k, (train_idx, test_idx) in enumerate(splits):
        train_fold = Fold(X_arr[train_idx], Y[train_idx], E[train_idx])
        best = tune(
            build,
            train_fold.X,
            train_fold.event_time,
            train_fold.event_indicator,
            scoring=scoring,
            train=train,
            cv=inner,
            n_trials=n_trials,
            direction=direction,
            seed=seed,
            refit=True,
            verbose=verbose,
        )
        score = scoring(
            best.model, X_arr[test_idx], Y[test_idx], E[test_idx], train_fold
        )
        rows.append({"fold": k, "score": float(score)})
        models.append(best.model)
        params.append(best.params)
        trials.append(best.trials)
        studies.append(best.study)
    return CVResult(
        scores=pl.DataFrame(rows),
        models=models,
        splits=splits,
        params=params,
        trials=trials,
        studies=studies,
    )


def _direction(scoring: Scoring) -> Direction:
    return "maximize" if getattr(scoring, "greater_is_better", True) else "minimize"


def _import_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "tune and nested_cv need optuna; install it with "
            "pip install 'tausurv[tune]'"
        ) from exc
    return optuna


class _OptunaVerbosity:
    """Silence Optuna's per-trial log for one search, then restore it."""

    def __init__(self, optuna: Any, verbose: bool) -> None:
        self.optuna = optuna
        self.verbose = verbose

    def __enter__(self) -> None:
        self.previous = self.optuna.logging.get_verbosity()
        if not self.verbose:
            self.optuna.logging.set_verbosity(self.optuna.logging.WARNING)

    def __exit__(self, *exc: object) -> None:
        self.optuna.logging.set_verbosity(self.previous)
