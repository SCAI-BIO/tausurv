from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.predictor import SurvivalPredictor


class Coxnet(SurvivalPredictor):
    def __init__(
        self,
        *,
        n_alphas: int = 100,
        alphas: ArrayLike | None = None,
        alpha_min_ratio: float | Literal["auto"] = "auto",
        l1_ratio: float = 0.5,
        penalty_factor: ArrayLike | None = None,
        copy_X: bool = True,
        tol: float = 1e-7,
        max_iter: int = 100000,
        verbose: int = 0,
        standardize: bool = False,
    ) -> None:
        r"""Coxnet model for survival analysis using scikit-survival's CoxnetSurvivalAnalysis.

        Parameters
        ----------
        n_alphas : int, default=100
            Number of alphas along the regularization path.
        alphas : ArrayLike or None, default=None
            List of alphas along the regularization path. If None, alphas are set automatically.
        alpha_min_ratio : float or "auto", default="auto"
            Minimum ratio for alpha. If "auto", it is set automatically.
        l1_ratio : float, default=0.5
            The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
        penalty_factor : ArrayLike or None, default=None
            Array of penalty factors for each coefficient.
        copy_X : bool, default=True
            Whether to copy the input data.
        tol : float, default=1e-7
            Tolerance for the optimization.
        max_iter : int, default=100000
            Maximum number of iterations.
        verbose : int, default=0
            Verbosity level.
        standardize : bool, default=False
            Whether to apply internal standard scaling to the features when calling `fit` or `predict` methods.
        """
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.l1_ratio = l1_ratio
        self.penalty_factor = penalty_factor
        self.copy_X = copy_X
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.standardize = standardize

        self._sksurv_linear, self._sksurv_surv = self._import_sksurv()
        self._model = None

    def fit(
            self,
            X: ArrayLike,
            event_time: ArrayLike,
            event_indicator: ArrayLike,
        ) -> "Coxnet":
        self.times_ = np.unique(event_time[event_indicator == 1])
        self._model = self._sksurv_linear.CoxnetSurvivalAnalysis(
            n_alphas=self.n_alphas,
            alphas=self.alphas,
            alpha_min_ratio=self.alpha_min_ratio,
            l1_ratio=self.l1_ratio,
            penalty_factor=self.penalty_factor,
            normalize=False, # will perform standardization instead
            copy_X=self.copy_X,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            fit_baseline_model=True,
        )
        y = self._sksurv_surv.from_arrays(event=event_indicator, time=event_time)
        if self.standardize:
            self._model = self._create_pipeline(self._model)
        self._model.fit(X, y)

        return self

    def predict_cumulative_hazard(
            self,
            X: ArrayLike,
            times: ArrayLike | None = None,
        ) -> NDArray[np.float64]:
        assert self._model is not None, "The model must be fitted before predicting."
        if times is None:
            cumhaz = self._model.predict_cumulative_hazard_function(X, return_array=True)
        else:
            cumhaz_step_fun = self._model.predict_cumulative_hazard_function(X, return_array=False)
            cumhaz = np.asarray([fn(times) for fn in cumhaz_step_fun], dtype=np.float64)
        return np.asarray(cumhaz, dtype=np.float64)

    
    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        assert self._model is not None, "The model must be fitted before predicting."
        return self._model.predict(X)

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        assert self._model is not None, "The model must be fitted before predicting."
        if times is None:
            surv = self._model.predict_survival_function(X, return_array=True)
        else:
            surv_step_fun = self._model.predict_survival_function(X, return_array=False)
            surv = np.asarray([fn(times) for fn in surv_step_fun], dtype=np.float64)
        return np.asarray(surv, dtype=np.float64)

    @property
    def coef_(self) -> NDArray[np.float64]:
        assert self._model is not None, "The model must be fitted before accessing coefficients."
        if self.standardize:
            return self._model.named_steps['coxnetsurvivalanalysis'].coef_
        return self._model.coef_

    @staticmethod
    def _import_sksurv():
        try:
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            import sksurv.linear_model as sksurv_linear
            from sksurv.util import Surv as sksurv_surv
        except ImportError as exc:
            raise ImportError(
                "SksurvCoxPH requires the optional 'scikit-survival' "
                "dependency; install it with `pip install tausurv[sksurv]`"
            ) from exc
        return sksurv_linear, sksurv_surv


    @staticmethod
    def _create_pipeline(model):
        try:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError(
                "Internal standard scaling requires the optional 'scikit-learn' "
                "dependency; install it with `pip install tausurv[sksurv]`"
            ) from exc
        return make_pipeline(StandardScaler(), model)