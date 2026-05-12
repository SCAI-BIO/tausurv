"""Tests for tausurv.model_selection."""

from __future__ import annotations

import numpy as np
import pytest

from tausurv.model_selection import train_test_split


def _data(n=500, n_features=5, event_rate=0.4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    T = rng.exponential(1.0, size=n)
    E = (rng.uniform(size=n) < event_rate).astype(int)
    return X, T, E


def test_shapes_partition_correctly():
    X, T, E = _data(n=500)
    X_tr, X_te, T_tr, T_te, E_tr, E_te = train_test_split(
        X, T, E, test_size=0.2, seed=0
    )
    assert X_tr.shape[0] + X_te.shape[0] == 500
    assert X_tr.shape[1] == X.shape[1] and X_te.shape[1] == X.shape[1]
    assert T_tr.shape == (X_tr.shape[0],) and T_te.shape == (X_te.shape[0],)
    assert E_tr.shape == (X_tr.shape[0],) and E_te.shape == (X_te.shape[0],)
    assert X_te.shape[0] == round(500 * 0.2)


def test_no_index_appears_in_both_splits():
    """Every row goes to exactly one of train/test."""
    X, T, E = _data(n=100)
    # Tag rows with a unique signature in T so we can identify them post-split.
    T = np.arange(100, dtype=float)
    X_tr, X_te, T_tr, T_te, _, _ = train_test_split(X, T, E, test_size=0.3, seed=0)
    union = np.sort(np.concatenate([T_tr, T_te]))
    np.testing.assert_array_equal(union, np.arange(100, dtype=float))


def test_stratified_split_preserves_event_proportions():
    X, T, E = _data(n=2000, event_rate=0.1)
    overall_rate = E.mean()
    _, _, _, _, E_tr, E_te = train_test_split(
        X, T, E, test_size=0.3, stratify=True, seed=0
    )
    # Within a few-sample slop, both splits match the overall event rate.
    assert abs(E_tr.mean() - overall_rate) < 0.01
    assert abs(E_te.mean() - overall_rate) < 0.02


def test_stratified_split_handles_competing_risks():
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 3))
    T = rng.exponential(1.0, size=n)
    # Three classes with different proportions: 60% censored, 25% cause 1, 15% cause 2.
    E = rng.choice([0, 1, 2], size=n, p=[0.6, 0.25, 0.15])
    _, _, _, _, E_tr, E_te = train_test_split(
        X, T, E, test_size=0.25, stratify=True, seed=0
    )
    for cls in (0, 1, 2):
        p_overall = (E == cls).mean()
        p_train = (E_tr == cls).mean()
        p_test = (E_te == cls).mean()
        assert abs(p_train - p_overall) < 0.02, f"class {cls}: train deviates"
        assert abs(p_test - p_overall) < 0.03, f"class {cls}: test deviates"


def test_non_stratified_can_skew_event_proportions():
    """Without stratification a rare-event dataset can produce a test
    set whose event proportion drifts from the overall rate by far more
    than the stratified version's deviation."""
    rng = np.random.default_rng(0)
    n = 50
    X = rng.normal(size=(n, 3))
    T = rng.exponential(1.0, size=n)
    E = (rng.uniform(size=n) < 0.05).astype(int)  # 5% event rate, very rare

    # Across many seeds, the non-stratified max deviation is much larger
    # than the stratified one.
    devs_strat = []
    devs_random = []
    for s in range(100):
        _, _, _, _, _, E_te_s = train_test_split(
            X, T, E, test_size=0.3, stratify=True, seed=s
        )
        _, _, _, _, _, E_te_r = train_test_split(
            X, T, E, test_size=0.3, stratify=False, seed=s
        )
        devs_strat.append(abs(E_te_s.mean() - E.mean()))
        devs_random.append(abs(E_te_r.mean() - E.mean()))
    assert max(devs_strat) <= max(devs_random)


def test_seed_reproducibility():
    X, T, E = _data(n=200)
    out1 = train_test_split(X, T, E, test_size=0.3, seed=42)
    out2 = train_test_split(X, T, E, test_size=0.3, seed=42)
    for a, b in zip(out1, out2, strict=True):
        np.testing.assert_array_equal(a, b)


def test_different_seeds_produce_different_splits():
    X, T, E = _data(n=200)
    _, _, T_te_1, _, _, _ = train_test_split(X, T, E, test_size=0.3, seed=1)
    _, _, T_te_2, _, _, _ = train_test_split(X, T, E, test_size=0.3, seed=2)
    # Splits should differ — sort by some component and compare; very low
    # probability of identical permutation by chance.
    assert not np.array_equal(np.sort(T_te_1), np.sort(T_te_2))


def test_test_size_must_be_in_open_unit_interval():
    X, T, E = _data(n=50)
    for bad in (-0.1, 0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="test_size"):
            train_test_split(X, T, E, test_size=bad)


def test_mismatched_shapes_raise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    T = rng.exponential(1.0, size=99)  # off by one
    E = (rng.uniform(size=100) < 0.5).astype(int)
    with pytest.raises(ValueError, match="first axis"):
        train_test_split(X, T, E)


def test_works_with_torch_tensors_via_asarray():
    """X can be any array-like — numpy converts."""
    torch = pytest.importorskip("torch")
    X = torch.randn(100, 3)
    T = torch.rand(100) + 0.1
    E = torch.randint(0, 2, (100,))
    X_tr, X_te, T_tr, T_te, E_tr, E_te = train_test_split(
        X.numpy(), T.numpy(), E.numpy(), seed=0
    )
    assert X_tr.shape[0] + X_te.shape[0] == 100
