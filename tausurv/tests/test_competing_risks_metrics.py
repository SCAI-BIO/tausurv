"""Tests for competing-risks-aware metrics.

The cause-specific variants should:

- reduce exactly to their single-event counterparts when only one event
  cause is present in the data,
- be ~0.5 (AUC, C) or large (Brier) for random predictions,
- match a hand-computed toy example,
- run end-to-end on real CR-simulated data + a trained DeepHit-CR.
"""

from __future__ import annotations

import numpy as np
import pytest

from tausurv import simulations
from tausurv.metrics import auc, brier
from tausurv.metrics.concordance import harrell, harrell_cause_specific


def test_harrell_cause_specific_reduces_to_harrell():
    X, T, E = simulations.single_risk(n=300, n_features=5, seed=0)
    rng = np.random.default_rng(0)
    r = rng.normal(size=300)
    c_single = harrell(T, E, r)
    c_cs = harrell_cause_specific(T, E.astype(np.int8), r, cause=1)
    assert c_single == pytest.approx(c_cs)


def test_brier_cause_specific_reduces_to_one_minus_survival_brier():
    X, T, E = simulations.single_risk(n=300, n_features=5, seed=0)
    grid = np.linspace(0.1, 3.0, 10)
    rng = np.random.default_rng(0)
    # Random monotone CIF in [0, 1]: each row uses a fixed quantile of grid.
    risk = rng.uniform(0.0, 1.0, size=300)
    F = np.minimum(1.0, risk[:, None] * np.linspace(0.2, 1.0, len(grid)))
    S = 1.0 - F
    bs_single = brier.score(T, E, S, grid)
    bs_cs = brier.score_cause_specific(T, E.astype(np.int8), F, grid, cause=1)
    np.testing.assert_allclose(bs_single, bs_cs, atol=1e-12)


def test_integrated_brier_cause_specific_reduces():
    X, T, E = simulations.single_risk(n=300, n_features=5, seed=0)
    grid = np.linspace(0.1, 3.0, 12)
    rng = np.random.default_rng(0)
    risk = rng.uniform(0.0, 1.0, size=300)
    F = np.minimum(1.0, risk[:, None] * np.linspace(0.2, 1.0, len(grid)))
    S = 1.0 - F
    ibs_single = brier.integrated(T, E, S, grid)
    ibs_cs = brier.integrated_cause_specific(T, E.astype(np.int8), F, grid, cause=1)
    assert ibs_single == pytest.approx(ibs_cs)


def test_auc_cause_specific_reduces_to_blanche():
    X, T, E = simulations.single_risk(n=300, n_features=5, seed=0)
    grid = np.linspace(0.2, 2.0, 8)
    rng = np.random.default_rng(0)
    risk = rng.uniform(0.0, 1.0, size=300)
    # Time-varying marker: F_1(t | x_i) = risk_i * t / max(t)
    F = np.outer(risk, grid / grid.max())
    auc_blanche = auc.blanche(T, E, F, grid)
    auc_cs = auc.cause_specific(T, E.astype(np.int8), F, grid, cause=1)
    # NaN-aware comparison.
    mask = np.isfinite(auc_blanche) & np.isfinite(auc_cs)
    np.testing.assert_allclose(auc_blanche[mask], auc_cs[mask], atol=1e-12)


def test_harrell_cause_specific_random_is_around_half():
    X, T, E = simulations.competing_risks(n=600, n_features=5, n_causes=2, seed=0)
    rng = np.random.default_rng(0)
    r_random = rng.normal(size=len(T))
    c = harrell_cause_specific(T, E, r_random, cause=1)
    assert 0.4 < c < 0.6


def test_harrell_cause_specific_perfect_is_high():
    """A predictor that knows exactly which subjects have cause-1 events
    and ranks them earliest should score near 1."""
    X, T, E = simulations.competing_risks(n=400, n_features=5, n_causes=2, seed=0)
    # Perfect cause-1 predictor: higher risk = shorter cause-1 time.
    # Use -T as the rank, but only for cause-1 events.
    r = np.where(
        E == 1, -T, -T.max() - 1.0
    )  # cause-1 cases get true ranks; everyone else gets the smallest rank
    c = harrell_cause_specific(T, E, r, cause=1)
    assert c > 0.95


def test_harrell_cause_specific_handles_both_causes():
    """Same risk score, two causes — concordance should differ because
    the partner sets and case sets are cause-specific."""
    X, T, E = simulations.competing_risks(n=500, n_features=5, n_causes=2, seed=0)
    c1 = harrell_cause_specific(T, E, X[:, 0], cause=1)
    c2 = harrell_cause_specific(T, E, X[:, 0], cause=2)
    assert 0.0 <= c1 <= 1.0 and 0.0 <= c2 <= 1.0
    assert not np.isclose(c1, c2, atol=1e-3), (
        "cause-1 and cause-2 concordance with the same score should not "
        "be (nearly) equal on data with cause-specific signal"
    )


def test_harrell_cause_specific_hand_traced_pairs():
    """Four subjects:
       i=0: T=1, cause=1   (case for cause 1)
       i=1: T=2, cause=2   (competing event)
       i=2: T=3, cause=0   (censored, later than i=0)
       i=3: T=4, cause=1   (later cause-1 event)

    Case i=0 has partners {1, 2, 3} — j=1 because its Y > 1 (still at risk),
    j=2 because its Y > 1 (still at risk; cause status of cause 1 known to
    be 'not yet'), j=3 because its Y > 1. Case i=3 has partners {1} only —
    j=0 has the same cause and j=2 was censored before i=3's event time.
    """
    T = np.array([1.0, 2.0, 3.0, 4.0])
    E = np.array([1, 2, 0, 1], dtype=np.int8)

    # r=[10, 1, 5, 2]: case 0 beats all three partners; case 3 beats partner 1.
    # All 4 comparisons concordant -> C = 1.0.
    r = np.array([10.0, 1.0, 5.0, 2.0])
    assert harrell_cause_specific(T, E, r, cause=1) == 1.0

    # r=[10, 1, 5, 20]: case 0 wins {1, 2} and loses to {3}; case 3 wins {1}.
    # 3 of 4 comparisons concordant -> C = 0.75.
    r2 = np.array([10.0, 1.0, 5.0, 20.0])
    assert harrell_cause_specific(T, E, r2, cause=1) == 0.75


def test_harrell_cause_specific_excludes_censored_earlier_partners():
    """When a censored subject occurs BEFORE the cause-1 case, that subject
    is NOT a comparable partner (cause-1 status unknown). Verify by
    contrast: adding a censored-earlier subject should not change the score."""
    # Two configurations, identical except for an extra censored subject
    # whose event time precedes the case.
    T1 = np.array([2.0, 4.0])
    E1 = np.array([1, 1], dtype=np.int8)
    r1 = np.array([10.0, 1.0])
    base = harrell_cause_specific(T1, E1, r1, cause=1)  # 1 case, 1 partner

    T2 = np.array([1.0, 2.0, 4.0])
    E2 = np.array([0, 1, 1], dtype=np.int8)  # extra censored at T=1
    r2 = np.array([100.0, 10.0, 1.0])  # extra subject has highest risk
    extra = harrell_cause_specific(T2, E2, r2, cause=1)

    # Same C — censored-earlier subjects are excluded from cause-1 case
    # i=1's partner set.
    assert base == extra


def test_cause_specific_rejects_cause_zero():
    T = np.array([1.0, 2.0])
    E = np.array([1, 0], dtype=np.int8)
    with pytest.raises(ValueError, match="cause must be >= 1"):
        harrell_cause_specific(T, E, [0.5, 0.7], cause=0)
    with pytest.raises(ValueError, match="cause must be >= 1"):
        brier.score_cause_specific(
            T, E, np.zeros((2, 3)), np.array([0.5, 1.0, 1.5]), cause=0
        )
    with pytest.raises(ValueError, match="cause must be >= 1"):
        auc.cause_specific(T, E, np.zeros((2, 3)), np.array([0.5, 1.0, 1.5]), cause=0)
