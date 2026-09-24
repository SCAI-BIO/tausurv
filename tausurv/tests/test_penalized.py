from __future__ import annotations

import numpy as np
import pytest

from tausurv import simulations
from tausurv.linear import (
    PenalizedCoxPH,
    PenalizedLogLogisticAFT,
    PenalizedLogNormalAFT,
    PenalizedWeibullAFT,
)


@pytest.mark.parametrize(
    "model_class",
    [
        PenalizedCoxPH,
        PenalizedWeibullAFT,
        PenalizedLogLogisticAFT,
        PenalizedLogNormalAFT,
    ],
)
def test_penalized_lifelines_wrappers_implement_predictor_contract(model_class):
    pytest.importorskip("lifelines")
    X, event_time, event_indicator = simulations.single_risk(
        n=100, n_features=3, censoring_rate=0.2, seed=0
    )
    model = model_class(penalizer=0.01).fit(X, event_time, event_indicator)
    times = np.linspace(0.05, 2.0, 8)

    survival = model.predict_survival_function(X, times)
    cumulative_hazard = model.predict_cumulative_hazard(X, times)

    assert model.times_.shape == (event_indicator.sum(),)
    assert survival.shape == cumulative_hazard.shape == (X.shape[0], times.size)
    assert np.all((survival >= 0.0) & (survival <= 1.0))
    assert np.all(np.diff(survival, axis=1) <= 1e-12)
    np.testing.assert_allclose(cumulative_hazard, -np.log(survival), rtol=1e-10)


def test_penalized_lifelines_wrapper_reuses_training_standardization():
    pytest.importorskip("lifelines")
    X, event_time, event_indicator = simulations.single_risk(
        n=100, n_features=3, censoring_rate=0.2, seed=1
    )
    model = PenalizedCoxPH(penalizer=0.01).fit(X, event_time, event_indicator)
    times = np.array([0.25, 0.5, 1.0])
    training_mean = model._train_mean.copy()
    training_std = model._train_std.copy()

    model.predict_survival_function(X[:10], times)

    np.testing.assert_array_equal(model._train_mean, training_mean)
    np.testing.assert_array_equal(model._train_std, training_std)