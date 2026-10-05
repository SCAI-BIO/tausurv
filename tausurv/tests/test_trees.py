from __future__ import annotations

import numpy as np
import pytest

from tausurv import simulations
from tausurv.metrics.concordance import harrell
from tausurv.trees.random_survival_forest import RandomSurvivalForest
from tausurv.trees.survival_boost import (
    SurvivalBoost,
    _default_time_grid,
    _draw_horizons,
    _horizon_targets,
)
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


def test_survival_boost_shapes():
    pytest.importorskip("sklearn")
    X, T, E = simulations.competing_risks(n=200, seed=0)
    sb = SurvivalBoost(n_iter=20, seed=0).fit(X, T, E)
    grid = np.linspace(0.1, 3.0, 8)
    H = sb.predict_cumulative_hazard(X, grid)
    assert H.shape == (200, 8)
    # Each horizon is predicted separately, so the curves need not be
    # monotone, but every horizon is a proper distribution.
    cif = sb.predict_cif(X, grid)
    S = sb.predict_survival_function(X, grid)
    assert (cif >= 0).all() and (S >= 0).all()
    np.testing.assert_allclose(cif.sum(axis=1) + S, 1.0, atol=1e-12)


def test_survival_boost_concordance_beats_random():
    # A ranking check is the only thing that catches swapped event/time
    # roles in the IPCW target construction — shape and monotonicity
    # assertions pass either way.
    pytest.importorskip("sklearn")
    X, T, E = simulations.competing_risks(n=500, censoring_rate=0.3, seed=0)
    sb = SurvivalBoost(seed=0).fit(X, T, E)
    c = harrell(T, E, sb.predict(X))
    assert c > 0.65  # well above 0.5 random baseline


def test_survival_boost_horizon_targets_truth_table():
    # One subject per branch of the IPCW target rule (Alberge et al. 2025,
    # Algorithm 2). Distinct weight values verify which array each branch
    # selects from.
    event = np.array([True, False, True, False, True])
    Y = np.array([1.0, 2.0, 3.0, 4.0, 4.0])
    horizons = np.array([2.0, 1.0, 3.0, 5.0, 2.0])
    ipcw_duration = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ipcw_horizons = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    target, weight = _horizon_targets(event, Y, horizons, ipcw_duration, ipcw_horizons)

    # event by horizon -> 1, weighted at the observed time (index 2 is the
    # Y == horizon boundary, which counts as observed); still under
    # observation -> 0, weighted at the horizon; censored before the
    # horizon -> 0 with weight 0.
    np.testing.assert_array_equal(target, [True, False, True, False, False])
    np.testing.assert_array_equal(weight, [10.0, 2.0, 30.0, 0.0, 5.0])


def test_survival_boost_draw_horizons_hard_zeros():
    rng = np.random.RandomState(0)
    horizons = _draw_horizons(rng, n=200, t_max=7.0, hard_zero_fraction=0.1)
    assert horizons.shape == (200,)
    assert ((horizons >= 0.0) & (horizons < 7.0)).all()
    assert (horizons == 0.0).sum() == 20


def test_survival_boost_default_time_grid():
    observed = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    # Fewer observations than steps: the sorted observed times themselves.
    np.testing.assert_array_equal(
        _default_time_grid(observed, n_steps=10), [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    # More observations than steps: quantile grid spanning the range.
    grid = _default_time_grid(observed, n_steps=3)
    np.testing.assert_array_equal(grid, [1.0, 3.0, 5.0])
    assert (np.diff(grid) >= 0).all()


def test_survival_boost_rejects_noncontiguous_causes():
    X = np.zeros((10, 2))
    T = np.arange(1.0, 11.0)
    E = np.array([0, 1, 3, 1, 3, 0, 1, 3, 1, 0])  # cause 2 missing
    with pytest.raises(ValueError, match="contiguous"):
        SurvivalBoost().fit(X, T, E)


def test_survival_tree_rejects_invalid_max_features():
    with pytest.raises(ValueError, match="max_features"):
        SurvivalTree(max_features="quartile").fit(
            np.zeros((10, 3)), np.arange(10.0), np.ones(10, dtype=np.int8)
        )
