from __future__ import annotations

import numpy as np
import pytest

from tausurv import simulations


def test_single_risk_shapes_and_dtype():
    X, T, E = simulations.single_risk(n=500, seed=42)
    assert X.shape == (500, 5)
    assert T.shape == (500,)
    assert E.shape == (500,)
    assert E.dtype == np.int8


def test_single_risk_event_indicator_is_binary():
    _, _, E = simulations.single_risk(n=500, seed=42)
    assert set(np.unique(E).tolist()) <= {0, 1}


def test_single_risk_censoring_rate_approximate():
    _, _, E = simulations.single_risk(n=2000, censoring_rate=0.3, seed=42)
    actual = float(1.0 - E.mean())
    assert abs(actual - 0.3) < 0.1


def test_single_risk_reproducible_with_seed():
    a = simulations.single_risk(n=100, seed=42)
    b = simulations.single_risk(n=100, seed=42)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_competing_risk_shapes():
    X, T, E = simulations.competing_risks(n=500, n_causes=3, seed=42)
    assert X.shape == (500, 5)
    assert T.shape == (500,)
    assert E.shape == (500,)


def test_competing_risk_event_indicator_uses_cause_coding():
    _, _, E = simulations.competing_risks(n=2000, n_causes=3, seed=42)
    assert set(np.unique(E).tolist()) <= {0, 1, 2, 3}
    for k in range(1, 4):
        assert (E == k).sum() > 0


def test_competing_risk_rejects_invalid_n_causes():
    with pytest.raises(ValueError, match="n_causes"):
        simulations.competing_risks(n=100, n_causes=0, seed=42)
