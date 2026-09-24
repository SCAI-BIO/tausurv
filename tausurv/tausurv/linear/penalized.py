from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd

from tausurv.predictor import SurvivalPredictor


class PenalizedLinear(SurvivalPredictor):
    def __init__(
        self,
        *,
        penalizer: float = 0.1,
        l1_ratio: float = 0.5,
        standardize: bool = True,
    ) -> None:
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.standardize = standardize

        self._model = None
        self._train_mean = None
        self._train_std = None

    def fit(
            self,
            X: ArrayLike,
            event_time: ArrayLike,
            event_indicator: ArrayLike,
        ) -> "PenalizedLinear":
        fitter_class = self._get_fitter()
        self.times_ = np.unique(event_time[event_indicator == 1])
        self._model = fitter_class(
            penalizer=self.penalizer,
            l1_ratio=self.l1_ratio,
        )
        if self.standardize:
            X = self._standardize(X)
        self._model.fit(pd.concat([pd.DataFrame(X), 
                                   pd.DataFrame(event_time, columns=["event_time"]), 
                                   pd.DataFrame(event_indicator, columns=["event_indicator"])], axis=1),
                                   duration_col="event_time", event_col="event_indicator")
        return self

    def predict_cumulative_hazard(
            self,
            X: ArrayLike,
            times: ArrayLike | None = None,
        ) -> NDArray[np.float64]:
        assert self._model is not None, "The model must be fitted before predicting."
        if self.standardize:
            X = self._standardize(X)
        cumhaz = self._model.predict_cumulative_hazard(pd.DataFrame(X), times=times)
        return cumhaz.to_numpy().T

    
    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        assert self._model is not None, "The model must be fitted before predicting."
        if self.standardize:
            X = self._standardize(X)
        return self._model.predict(pd.DataFrame(X))

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        assert self._model is not None, "The model must be fitted before predicting."
        if self.standardize:
            X = self._standardize(X)
        surv = self._model.predict_survival_function(pd.DataFrame(X), times=times)
        return surv.to_numpy().T

    @staticmethod
    def _get_fitter():
        raise NotImplementedError("Subclasses must implement the _get_fitter method.")

    @staticmethod
    def _import_lifelines_fitters():
        try:
            from lifelines import fitters as fitters_module
        except ImportError as exc:
            raise ImportError(
                "SksurvCoxPH requires the optional 'lifelines' "
                "dependency; install it with `pip install tausurv[lifelines]`"
            ) from exc
        return fitters_module


    def _standardize(self, data: ArrayLike):
        if self._train_mean is None or self._train_std is None:
            self._train_mean = np.mean(data, axis=0)
            self._train_std = np.std(data, axis=0)
        return (data - self._train_mean) / self._train_std

class PenalizedCoxPH(PenalizedLinear):
    def _get_fitter(self):
        fitters_module = self._import_lifelines_fitters()
        return fitters_module.coxph_fitter.CoxPHFitter

class PenalizedWeibullAFT(PenalizedLinear):
    def _get_fitter(self):
        fitters_module = self._import_lifelines_fitters()
        return fitters_module.weibull_aft_fitter.WeibullAFTFitter

class PenalizedLogLogisticAFT(PenalizedLinear):
    def _get_fitter(self):
        fitters_module = self._import_lifelines_fitters()
        return fitters_module.log_logistic_aft_fitter.LogLogisticAFTFitter

class PenalizedLogNormalAFT(PenalizedLinear):
    def _get_fitter(self):
        fitters_module = self._import_lifelines_fitters()
        return fitters_module.log_normal_aft_fitter.LogNormalAFTFitter