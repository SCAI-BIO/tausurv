r"""Survival-aware splitting, cross-validation and hyperparameter tuning.

Every entry point takes the same two callables: ``build(trial=None)`` returns
a fresh model, ``train(model, X, event_time, event_indicator, trial=None)``
fits it in place. ``train`` is optional and defaults to ``model.fit``. In
:func:`tune` and :func:`nested_cv` the ``trial`` is an Optuna trial, so
hyperparameters are sampled with ``trial.suggest_*`` inside the callable that
uses them; in :func:`cross_validate` it is ``None``.
"""

from tausurv.model_selection import scoring
from tausurv.model_selection._cross_validate import Fold, Scorer, cross_validate
from tausurv.model_selection._results import CVResult, Ensemble, TuneResult
from tausurv.model_selection._split import stratified_folds, train_test_split
from tausurv.model_selection._tune import nested_cv, tune

__all__ = [
    "CVResult",
    "Ensemble",
    "Fold",
    "Scorer",
    "TuneResult",
    "cross_validate",
    "nested_cv",
    "scoring",
    "stratified_folds",
    "train_test_split",
    "tune",
]
