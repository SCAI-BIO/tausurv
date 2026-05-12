from __future__ import annotations

import numpy as np
import pytest

from tausurv import simulations
from tausurv.metrics.concordance import harrell
from tausurv.trees.random_survival_forest import RandomSurvivalForest


def test_rsf_concordance_close_to_sksurv():
    """RSFs are random; we can't expect identical predictions, but trained on
    the same data with the same hyperparameters both should achieve similar
    discrimination."""
    sksurv_ensemble = pytest.importorskip("sksurv.ensemble")
    X, T, E = simulations.single_risk(n=400, n_features=5, censoring_rate=0.3, seed=0)

    rsf_ours = RandomSurvivalForest(
        n_estimators=30, max_depth=5, min_samples_leaf=10, seed=0
    ).fit(X, T, E)

    structured = np.array(
        list(zip(E.astype(bool), T)),
        dtype=[("event", bool), ("time", float)],
    )
    rsf_them = sksurv_ensemble.RandomSurvivalForest(
        n_estimators=30,
        max_depth=5,
        min_samples_leaf=10,
        random_state=0,
        n_jobs=1,
    ).fit(X, structured)

    c_ours = harrell(T, E, rsf_ours.predict(X))
    c_them = harrell(T, E, rsf_them.predict(X))
    assert abs(c_ours - c_them) < 0.05


def test_rsf_risk_scores_rank_correlate_with_sksurv():
    sksurv_ensemble = pytest.importorskip("sksurv.ensemble")
    scipy_stats = pytest.importorskip("scipy.stats")
    X, T, E = simulations.single_risk(n=400, n_features=5, censoring_rate=0.3, seed=0)

    rsf_ours = RandomSurvivalForest(
        n_estimators=30, max_depth=5, min_samples_leaf=10, seed=0
    ).fit(X, T, E)

    structured = np.array(
        list(zip(E.astype(bool), T)),
        dtype=[("event", bool), ("time", float)],
    )
    rsf_them = sksurv_ensemble.RandomSurvivalForest(
        n_estimators=30,
        max_depth=5,
        min_samples_leaf=10,
        random_state=0,
        n_jobs=1,
    ).fit(X, structured)

    rho = scipy_stats.spearmanr(rsf_ours.predict(X), rsf_them.predict(X)).statistic
    assert rho > 0.6
