from __future__ import annotations

import numpy as np
import pytest

from tausurv import bin_index, time_grid


def _observed(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.exponential(10.0, n)
    C = rng.exponential(14.0, n)
    Y = np.minimum(T, C)
    delta = (T <= C).astype(np.int64)
    return Y, delta


def test_grid_is_increasing_and_spans_data():
    Y, delta = _observed()
    grid = time_grid(Y, delta, n_bins=15)
    assert grid.ndim == 1 and len(grid) <= 15
    assert np.all(np.diff(grid) > 0)
    assert grid[-1] == Y.max()


def test_grid_balances_events_across_bins():
    rng = np.random.default_rng(1)
    Y = rng.exponential(5.0, 1000)
    delta = np.ones(1000, dtype=np.int64)
    grid = time_grid(Y, delta, n_bins=10)
    counts = np.bincount(bin_index(grid, Y), minlength=len(grid))
    assert np.all(counts == 100)


def test_grid_collapses_tied_quantiles():
    Y = np.repeat([1.0, 2.0], 100)
    delta = np.ones(200, dtype=np.int64)
    grid = time_grid(Y, delta, n_bins=10)
    assert np.array_equal(grid, [1.0, 1.5, 2.0])


def test_grid_requires_events():
    with pytest.raises(ValueError):
        time_grid(np.ones(5), np.zeros(5))


def test_grid_accepts_cause_labels():
    Y, delta = _observed(seed=3)
    causes = delta * (1 + np.arange(len(delta)) % 2)
    assert np.array_equal(time_grid(Y, causes), time_grid(Y, delta))


def test_bin_index_right_closed():
    grid = np.array([1.0, 2.0, 3.0])
    times = [0.5, 1.0, 1.5, 2.0, 3.0, 3.5]
    assert bin_index(grid, times).tolist() == [0, 0, 1, 1, 2, 3]


def test_bin_index_matches_loss_binning():
    torch = pytest.importorskip("torch")
    Y, delta = _observed(n=200, seed=4)
    grid = time_grid(Y, delta, n_bins=8)
    K = len(grid)
    queries = np.concatenate([Y, grid, grid[-1:] + 1.0])
    np_idx = bin_index(grid, queries).clip(max=K - 1)
    torch_idx = torch.searchsorted(
        torch.as_tensor(grid), torch.as_tensor(queries), right=False
    ).clamp(max=K - 1)
    assert np.array_equal(np_idx, torch_idx.numpy())


def test_grid_wires_into_discrete_time_training():
    torch = pytest.importorskip("torch")
    from tausurv.nn import LogisticHazard
    from tausurv.nn.functional import logistic_hazard_nll

    rng = np.random.default_rng(5)
    Y, delta = _observed(n=100, seed=5)
    X = rng.normal(size=(100, 3))
    grid = time_grid(Y, delta, n_bins=8)
    model = LogisticHazard(in_features=3, n_bins=len(grid)).set_time_grid(grid)
    loss = logistic_hazard_nll(
        model(torch.as_tensor(X, dtype=torch.float32)),
        torch.as_tensor(Y),
        torch.as_tensor(delta),
        torch.as_tensor(grid),
    )
    assert torch.isfinite(loss)
    assert np.array_equal(model.times_, grid)
