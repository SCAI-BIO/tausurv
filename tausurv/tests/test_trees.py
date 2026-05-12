from __future__ import annotations

import numpy as np
import pytest

from tausurv import simulations
from tausurv.metrics.concordance import harrell
from tausurv.trees.random_survival_forest import RandomSurvivalForest
from tausurv.trees.survival_tree import SurvivalTree, _log_rank_statistic


def test_log_rank_statistic_separates_two_distinct_groups():
    # Group A: events at t=1,2,3. Group B: events at t=10,11,12.
    Y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    delta = np.array([1, 1, 1, 1, 1, 1], dtype=np.int8)
    left = np.array([True, True, True, False, False, False])
    score = _log_rank_statistic(Y, delta, left)
    assert score > 5.0


def test_log_rank_statistic_zero_for_no_signal():
    # All identical outcomes
    Y = np.array([1.0, 1.0, 1.0, 1.0])
    delta = np.array([1, 1, 1, 1], dtype=np.int8)
    left = np.array([True, False, True, False])
    score = _log_rank_statistic(Y, delta, left)
    assert score == 0.0


def test_survival_tree_predict_shapes_and_monotonicity():
    X, T, E = simulations.single_risk(n=200, seed=0)
    tree = SurvivalTree(max_depth=3, seed=0).fit(X, T, E)
    grid = np.linspace(0.1, 3.0, 10)
    H = tree.predict_cumulative_hazard(X, grid)
    assert H.shape == (200, 10)
    assert (H >= 0).all()
    assert (np.diff(H, axis=1) >= 0).all()
    S = tree.predict_survival_function(X, grid)
    assert ((S >= 0) & (S <= 1)).all()
    assert (np.diff(S, axis=1) <= 0).all()


def test_survival_tree_predict_ranks_outcomes():
    X, T, E = simulations.single_risk(n=500, seed=0)
    tree = SurvivalTree(max_depth=4, min_samples_leaf=10, seed=0).fit(X, T, E)
    risk = tree.predict(X)
    assert risk.shape == (500,)
    # Higher predicted risk -> shorter event time.
    assert np.corrcoef(risk, T)[0, 1] < 0


def test_survival_tree_reproducible_with_seed():
    X, T, E = simulations.single_risk(n=200, seed=0)
    grid = np.linspace(0.1, 3.0, 5)
    t1 = SurvivalTree(seed=42).fit(X, T, E)
    t2 = SurvivalTree(seed=42).fit(X, T, E)
    np.testing.assert_array_equal(
        t1.predict_cumulative_hazard(X, grid),
        t2.predict_cumulative_hazard(X, grid),
    )


def test_random_survival_forest_shapes():
    X, T, E = simulations.single_risk(n=200, seed=0)
    rsf = RandomSurvivalForest(n_estimators=10, max_depth=3, seed=0).fit(X, T, E)
    grid = np.linspace(0.1, 3.0, 8)
    H = rsf.predict_cumulative_hazard(X, grid)
    assert H.shape == (200, 8)
    assert (np.diff(H, axis=1) >= 0).all()


def test_random_survival_forest_concordance_beats_random():
    X, T, E = simulations.single_risk(n=500, censoring_rate=0.3, seed=0)
    rsf = RandomSurvivalForest(
        n_estimators=20, max_depth=4, min_samples_leaf=10, seed=0
    ).fit(X, T, E)
    c = harrell(T, E, rsf.predict(X))
    assert c > 0.65  # well above 0.5 random baseline


def test_survival_tree_rejects_invalid_max_features():
    with pytest.raises(ValueError, match="max_features"):
        SurvivalTree(max_features="quartile").fit(
            np.zeros((10, 3)), np.arange(10.0), np.ones(10, dtype=np.int8)
        )
