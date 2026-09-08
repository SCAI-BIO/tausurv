r"""What a study returns: fold models, splits, scores and searches."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from tausurv.model_selection._fold import Fold
from tausurv.predictor import CompetingRisksPredictor

Split = tuple[NDArray[np.intp], NDArray[np.intp]]


@dataclass
class CVResult(CompetingRisksPredictor):
    r"""Everything a fold-based study produced, usable as a predictor.

    The result predicts *out of fold* on the data the study was run on: row
    $i$ is scored by the model of the fold that held $i$ out, so no subject
    is ever predicted by a model that trained on it. Pass the same ``X`` the
    study saw; a different number of rows is refused, and rows no fold held
    out (possible with explicit splits) come back as NaN. For new data use
    :attr:`ensemble`, the average of the fold models.

    Attributes
    ----------
    scores : polars.DataFrame
        One row per fold, one column per scorer and ``seconds`` of training
        time. For :func:`nested_cv` these are the outer scores, the honest
        estimate after tuning, and the seconds cover the inner search.
    models : list
        The fitted model of each fold, in fold order.
    splits : list of (train_idx, test_idx)
        The indices that produced each fold.
    params : list of dict
        Parameters chosen inside each fold; empty for plain cross-validation.
    trials : list of polars.DataFrame or None
        Optuna's trials per fold, ``number``, ``value``, ``state`` and one
        column per parameter. ``None`` without a search.
    studies : list or None
        The live Optuna studies; not persisted, use :attr:`trials` or a
        ``storage`` URL for that.
    """

    scores: pl.DataFrame
    models: list[Any]
    splits: list[Split]
    params: list[dict[str, Any]] = field(default_factory=list)
    trials: list[pl.DataFrame] | None = None
    studies: list[Any] | None = None

    @property
    def n_samples(self) -> int:
        return int(max(max(tr.max(), te.max()) for tr, te in self.splits)) + 1

    @property
    def n_causes(self) -> int:  # type: ignore[override]
        return int(getattr(self.models[0], "n_causes", 1))

    @property
    def times_(self) -> NDArray[np.float64]:  # type: ignore[override]
        return np.asarray(self.models[0].times_, dtype=np.float64)

    @property
    def ensemble(self) -> Ensemble:
        """Average of the fold models, for data the study did not see."""
        return Ensemble(self.models)

    def evaluate(
        self,
        scoring: Any,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
    ) -> NDArray[np.float64]:
        """Apply a scorer to every fold model on its held-out fold.

        ``scoring`` is any scorer, see :mod:`tausurv.model_selection.scoring`;
        one returning a curve gives a ``(n_folds, n_times)`` array, one
        returning a number gives ``(n_folds,)``. Pass the study's own data.
        Lets a finished study be scored on new metrics without refitting.
        """
        X_arr = np.asarray(X)
        Y = np.asarray(event_time, dtype=np.float64)
        E = np.asarray(event_indicator)
        if X_arr.shape[0] != self.n_samples:
            raise ValueError(
                f"evaluate needs the study's own data with {self.n_samples} rows, "
                f"got {X_arr.shape[0]}"
            )
        results = []
        for model, (train_idx, test_idx) in zip(self.models, self.splits, strict=True):
            train = Fold(X_arr[train_idx], Y[train_idx], E[train_idx])
            results.append(
                scoring(model, X_arr[test_idx], Y[test_idx], E[test_idx], train)
            )
        return np.asarray(results, dtype=np.float64)

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        return self._out_of_fold(X, lambda model, rows: model.predict(rows))

    def _survival_function(
        self, X: ArrayLike, times: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self._out_of_fold(
            X, lambda model, rows: model.predict_survival_function(rows, times)
        )

    def _cif(self, X: ArrayLike, times: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._out_of_fold(X, lambda model, rows: _cif_3d(model, rows, times))

    def _out_of_fold(self, X: ArrayLike, predict: Any) -> NDArray[np.float64]:
        X_arr = np.asarray(X)
        if X_arr.shape[0] != self.n_samples:
            raise ValueError(
                f"out-of-fold prediction needs the study's own data with "
                f"{self.n_samples} rows, got {X_arr.shape[0]}; use .ensemble for "
                f"new data"
            )
        out: NDArray[np.float64] | None = None
        for model, (_, test_idx) in zip(self.models, self.splits, strict=True):
            block = np.asarray(predict(model, X_arr[test_idx]), dtype=np.float64)
            if out is None:
                out = np.full((self.n_samples, *block.shape[1:]), np.nan)
            out[test_idx] = block
        assert out is not None
        return out

    def save(self, path: str | Path) -> None:
        """Write the fold models, splits, scores, params and trials to ``path``.

        Each model goes to ``fold_<k>/`` through its own ``save``; models
        without one cannot be persisted.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for k, model in enumerate(self.models):
            if not hasattr(model, "save"):
                raise TypeError(
                    f"{type(model).__name__} has no save(); the study cannot be "
                    f"persisted"
                )
            model.save(path / f"fold_{k:02d}")
        arrays = {f"train_{k:02d}": tr for k, (tr, _) in enumerate(self.splits)}
        arrays |= {f"test_{k:02d}": te for k, (_, te) in enumerate(self.splits)}
        np.savez(path / "splits.npz", **arrays)  # type: ignore[arg-type]  # stubs type **kwds against allow_pickle
        self.scores.write_parquet(path / "scores.parquet")
        (path / "params.json").write_text(json.dumps(self.params, indent=2))
        for k, table in enumerate(self.trials or []):
            table.write_parquet(path / f"trials_{k:02d}.parquet")
        meta = {"model": type(self.models[0]).__name__, "n_folds": len(self.models)}
        (path / "result.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path, model: Any) -> CVResult:
        """Read a result written by :meth:`save`.

        ``model`` is the fold models' class, which provides ``load``.
        """
        path = Path(path)
        meta = json.loads((path / "result.json").read_text())
        if meta["model"] != model.__name__:
            raise ValueError(
                f"result at {path} holds {meta['model']} models; load it with "
                f"model={meta['model']}, not {model.__name__}"
            )
        n_folds = int(meta["n_folds"])
        models = [model.load(path / f"fold_{k:02d}") for k in range(n_folds)]
        with np.load(path / "splits.npz") as data:
            splits = [
                (data[f"train_{k:02d}"], data[f"test_{k:02d}"]) for k in range(n_folds)
            ]
        trial_files = sorted(path.glob("trials_*.parquet"))
        return cls(
            scores=pl.read_parquet(path / "scores.parquet"),
            models=models,
            splits=splits,
            params=json.loads((path / "params.json").read_text()),
            trials=[pl.read_parquet(f) for f in trial_files] or None,
        )


@dataclass
class TuneResult:
    """Outcome of :func:`tune`.

    Attributes
    ----------
    params : dict
        Best trial's parameters, keyed by the names given to ``suggest_*``.
    score : float
        Best mean cross-validated score.
    model : object or None
        Model built and trained with the best parameters on all the data
        passed to :func:`tune`; ``None`` when ``refit=False``.
    trials : polars.DataFrame
        Every trial: ``number``, ``value``, ``state`` and one column per
        parameter.
    study : optuna.Study or None
        The live search; not persisted, pass ``storage`` to :func:`tune` for
        that.
    """

    params: dict[str, Any]
    score: float
    model: Any
    trials: pl.DataFrame
    study: Any = None

    def save(self, path: str | Path) -> None:
        """Write the model (through its own ``save``), params, score and trials."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            if not hasattr(self.model, "save"):
                raise TypeError(f"{type(self.model).__name__} has no save()")
            self.model.save(path / "model")
        meta = {
            "model": None if self.model is None else type(self.model).__name__,
            "params": self.params,
            "score": self.score,
        }
        (path / "result.json").write_text(json.dumps(meta, indent=2))
        self.trials.write_parquet(path / "trials.parquet")

    @classmethod
    def load(cls, path: str | Path, model: Any = None) -> TuneResult:
        """Read a result written by :meth:`save`.

        ``model`` is the fitted model's class, which provides ``load``.
        """
        path = Path(path)
        meta = json.loads((path / "result.json").read_text())
        fitted = None
        if meta["model"] is not None:
            if model is None or model.__name__ != meta["model"]:
                raise ValueError(
                    f"result at {path} holds a {meta['model']}; "
                    f"pass model={meta['model']}"
                )
            fitted = model.load(path / "model")
        return cls(
            params=meta["params"],
            score=float(meta["score"]),
            model=fitted,
            trials=pl.read_parquet(path / "trials.parquet"),
        )


class Ensemble(CompetingRisksPredictor):
    """Mean prediction of several fitted models with a shared interface."""

    def __init__(self, models: list[Any]) -> None:
        self.models = models

    @property
    def n_causes(self) -> int:  # type: ignore[override]
        return int(getattr(self.models[0], "n_causes", 1))

    @property
    def times_(self) -> NDArray[np.float64]:  # type: ignore[override]
        return np.asarray(self.models[0].times_, dtype=np.float64)

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        return _mean([m.predict(X) for m in self.models])

    def _survival_function(
        self, X: ArrayLike, times: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _mean([m.predict_survival_function(X, times) for m in self.models])

    def _cif(self, X: ArrayLike, times: NDArray[np.float64]) -> NDArray[np.float64]:
        return _mean([_cif_3d(m, X, times) for m in self.models])


def _mean(predictions: list[Any]) -> NDArray[np.float64]:
    return np.asarray(np.mean(predictions, axis=0), dtype=np.float64)


def _cif_3d(
    model: Any, X: ArrayLike, times: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``(n, n_causes, T)`` from any predictor, single-event ones included."""
    cif = np.asarray(model.predict_cif(X, times), dtype=np.float64)
    return cif if cif.ndim == 3 else cif[:, None, :]


def trials_table(study: Any) -> pl.DataFrame:
    """Optuna trials as a frame: ``number``, ``value``, ``state`` and the parameters."""
    rows = [
        {"number": t.number, "value": t.value, "state": t.state.name, **t.params}
        for t in study.trials
    ]
    return pl.DataFrame(rows)
