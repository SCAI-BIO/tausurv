from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.linear._checkpoint import _check_model_name
from tausurv.predictor import SurvivalPredictor


class Coxnet(SurvivalPredictor):
    r"""Cox proportional hazards with an elastic-net penalty.

    A wrapper around scikit-survival's ``CoxnetSurvivalAnalysis``, which
    fits a regularisation path by coordinate descent (Simon et al., 2011)
    and keeps the last model on the path for prediction. Needs the
    ``sksurv`` extra: ``pip install tausurv[sksurv]``.

    Parameters
    ----------
    n_alphas : int, default 100
        Number of penalty strengths along the regularisation path; ignored
        when ``alphas`` is given.
    alphas : float or (k,) array-like, optional
        Explicit penalty strengths. A single float fits one model.
    alpha_min_ratio : float or "auto", default "auto"
        Ratio of the smallest to the largest penalty on the automatic path.
    l1_ratio : float, default 0.5
        Elastic-net mixing in $[0, 1]$: $1$ is the lasso, $0$ ridge.
    penalty_factor : (d,) array-like, optional
        Per-coefficient multipliers of the penalty; $0$ leaves a
        coefficient unpenalised.
    copy_X : bool, default True
        Whether scikit-survival copies the covariates before fitting.
    tol : float, default 1e-7
        Convergence tolerance of the coordinate descent.
    max_iter : int, default 100000
        Iteration cap of the coordinate descent.
    verbose : bool, default False
        Forwarded to scikit-survival.
    standardize : bool, default True
        Centre and scale the covariates before fitting and prediction. The
        coefficients in ``coef_`` are then on the standardised scale.

    Attributes
    ----------
    times_ : (k,) array
        Unique training event times, the model's natural time grid.
    coef_ : (d,) array
        Coefficients of the last model on the path, on the standardised
        scale when ``standardize`` is ``True``.

    References
    ----------
    Simon, N., Friedman, J., Hastie, T., Tibshirani, R. (2011).
    Regularization paths for Cox's proportional hazards model via
    coordinate descent. Journal of Statistical Software, 39(5).
    """

    def __init__(
        self,
        *,
        n_alphas: int = 100,
        alphas: float | ArrayLike | None = None,
        alpha_min_ratio: float | Literal["auto"] = "auto",
        l1_ratio: float = 0.5,
        penalty_factor: ArrayLike | None = None,
        copy_X: bool = True,  # noqa: N803  # scikit-survival's parameter name
        tol: float = 1e-7,
        max_iter: int = 100000,
        verbose: bool = False,
        standardize: bool = True,
    ) -> None:
        if alphas is not None and isinstance(alphas, (float, int)):
            alphas = np.array([alphas], dtype=np.float64)
        self.alphas = alphas
        self.n_alphas = n_alphas if alphas is None else len(np.asarray(alphas))
        self.alpha_min_ratio = alpha_min_ratio
        self.l1_ratio = l1_ratio
        self.penalty_factor = penalty_factor
        self.copy_X = copy_X
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.standardize = standardize

        self._model: Any = None

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
    ) -> Coxnet:
        sksurv_linear, sksurv_util = _import_sksurv()
        covariates = np.asarray(X, dtype=np.float64)
        T = np.asarray(event_time, dtype=np.float64)
        delta = np.asarray(event_indicator)
        self.times_ = np.unique(T[delta == 1])
        model = sksurv_linear.CoxnetSurvivalAnalysis(
            n_alphas=self.n_alphas,
            alphas=self.alphas,
            alpha_min_ratio=self.alpha_min_ratio,
            l1_ratio=self.l1_ratio,
            penalty_factor=self.penalty_factor,
            normalize=False,
            copy_X=self.copy_X,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            fit_baseline_model=True,
        )
        if self.standardize:
            model = _standardised(model)
        model.fit(covariates, sksurv_util.Surv.from_arrays(event=delta == 1, time=T))
        self._model = model
        return self

    def predict_cumulative_hazard(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        r"""$\hat \Lambda(t \mid x)$ from the fitted baseline and risk score."""
        model = self._fitted()
        times_arr = self._resolve_times(times)
        step_functions = model.predict_cumulative_hazard_function(
            np.asarray(X, dtype=np.float64)
        )
        return np.asarray([fn(times_arr) for fn in step_functions], dtype=np.float64)

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""The linear predictor $x^\top \hat\beta$; larger means more risk."""
        model = self._fitted()
        return np.asarray(
            model.predict(np.asarray(X, dtype=np.float64)), dtype=np.float64
        )

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        model = self._fitted()
        step_functions = model.predict_survival_function(
            np.asarray(X, dtype=np.float64)
        )
        return np.asarray([fn(times) for fn in step_functions], dtype=np.float64)

    @property
    def coef_(self) -> NDArray[np.float64]:
        return np.asarray(self._inner().coef_, dtype=np.float64)

    def save(self, path: str | Path) -> None:
        """Write ``config.json`` and ``state.npz`` to ``path``."""
        if self._model is None:
            raise RuntimeError("Coxnet is not fitted; call fit() before save()")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "model": type(self).__name__,
            "n_alphas": self.n_alphas,
            "alphas": _to_jsonable(self.alphas),
            "alpha_min_ratio": self.alpha_min_ratio,
            "l1_ratio": self.l1_ratio,
            "penalty_factor": _to_jsonable(self.penalty_factor),
            "copy_X": self.copy_X,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "verbose": self.verbose,
            "standardize": self.standardize,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))
        inner = self._inner()
        baseline = inner._baseline_models[-1]
        arrays: dict[str, Any] = {
            "coef": np.asarray(inner.coef_),
            "offset": np.asarray(inner.offset_),
            "alphas": np.asarray(inner.alphas_),
            "alpha_min_ratio": np.asarray(inner.alpha_min_ratio_),
            "penalty_factor": np.asarray(inner.penalty_factor_),
            "deviance_ratio": np.asarray(inner.deviance_ratio_),
            "n_features_in": np.asarray(inner.n_features_in_),
            "times": np.asarray(self.times_),
            "baseline_times": np.asarray(baseline.unique_times_),
            "baseline_cum_hazard": np.asarray(
                [m.cum_baseline_hazard_.y for m in inner._baseline_models]
            ),
        }
        if self.standardize:
            scaler = self._model.named_steps["standardscaler"]
            arrays["scaler_mean"] = np.asarray(scaler.mean_)
            arrays["scaler_scale"] = np.asarray(scaler.scale_)
        np.savez_compressed(path / "state.npz", **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Coxnet:
        """Reconstruct a model saved by `save`."""
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        _check_model_name(config.pop("model"), cls, path)
        model = cls(**config)
        sksurv_linear, _ = _import_sksurv()
        from sksurv.functions import StepFunction
        from sksurv.linear_model.coxph import BreslowEstimator

        with np.load(path / "state.npz") as state:
            inner = sksurv_linear.CoxnetSurvivalAnalysis(
                n_alphas=model.n_alphas,
                alphas=model.alphas,
                alpha_min_ratio=model.alpha_min_ratio,
                l1_ratio=model.l1_ratio,
                penalty_factor=model.penalty_factor,
                normalize=False,
                copy_X=model.copy_X,
                tol=model.tol,
                max_iter=model.max_iter,
                verbose=model.verbose,
                fit_baseline_model=True,
            )
            inner.alphas_ = state["alphas"]
            inner.alpha_min_ratio_ = float(state["alpha_min_ratio"])
            inner.penalty_factor_ = state["penalty_factor"]
            inner.coef_ = state["coef"]
            inner.deviance_ratio_ = state["deviance_ratio"]
            inner.offset_ = state["offset"]
            inner.n_features_in_ = int(state["n_features_in"])

            baseline_times = state["baseline_times"]
            cum_hazard = state["baseline_cum_hazard"]
            baselines = []
            for values in cum_hazard:
                estimator = BreslowEstimator()
                estimator.unique_times_ = baseline_times
                estimator.cum_baseline_hazard_ = StepFunction(baseline_times, values)
                estimator.baseline_survival_ = StepFunction(
                    baseline_times, np.exp(-values)
                )
                baselines.append(estimator)
            inner._baseline_models = tuple(baselines)

            if model.standardize:
                model._model = _standardised(inner)
                scaler = model._model.named_steps["standardscaler"]
                scaler.mean_ = state["scaler_mean"]
                scaler.scale_ = state["scaler_scale"]
                scaler.n_features_in_ = int(state["n_features_in"])
            else:
                model._model = inner
            model.times_ = state["times"]
        return model

    def _inner(self) -> Any:
        """The fitted scikit-survival estimator, unwrapped from any pipeline."""
        model = self._fitted()
        if self.standardize:
            model = model.named_steps["coxnetsurvivalanalysis"]
        return model

    def _fitted(self) -> Any:
        if self._model is None:
            raise RuntimeError("Coxnet is not fitted; call fit() before predicting")
        return self._model


def _import_sksurv() -> tuple[Any, Any]:
    try:
        import sksurv.linear_model
        import sksurv.util
    except ImportError as exc:
        raise ImportError(
            "Coxnet needs the optional scikit-survival dependency; "
            "install it with `pip install tausurv[sksurv]`"
        ) from exc
    return sksurv.linear_model, sksurv.util


def _standardised(model: Any) -> Any:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), model)


def _to_jsonable(value: Any) -> Any:
    """JSON-encode a constructor argument that may be an array or scalar."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return np.asarray(value).tolist()
