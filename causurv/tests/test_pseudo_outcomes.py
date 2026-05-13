"""Tests for the DR-survival pseudo-outcome from Frauen et al. 2025.

The DR pseudo-outcome is the centerpiece of the orthogonal-survival
learners; if its math drifts, every learner downstream becomes wrong
without an obvious symptom. These tests pin the math via:

(1) shape + finiteness on real data,
(2) algebraic identities that the formula must satisfy
    (e.g., martingale correction collapses to zero in the
    no-confounding/no-censoring oracle case).
"""

from __future__ import annotations

import numpy as np
import pytest

from causurv.nuisances import cross_fit
from causurv.nuisances.pseudo_outcomes import (
    PSEUDO_OUTCOME_FOR,
    dr_survival,
    r_survival,
)


def _data_and_cf(n=300, seed=0, n_folds=3, t_max=10):
    from sklearn.linear_model import LogisticRegression
    from tausurv.linear import CoxPH

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    T = rng.integers(1, t_max + 1, size=n).astype(np.float64)
    E = (rng.uniform(size=n) < 0.7).astype(np.int8)
    A = (rng.uniform(size=n) < 0.5).astype(np.int8)
    cf = cross_fit(
        X, T, E, A,
        outcome_factory=lambda: CoxPH(),
        propensity_factory=lambda: LogisticRegression(max_iter=500),
        censoring_factory=lambda: CoxPH(),
        times=np.arange(1, t_max + 1, dtype=np.float64),
        n_folds=n_folds,
        per_arm_censoring=True,
        seed=seed,
    )
    return X, T, E, A, cf


def test_returns_per_subject_pseudo_outcome_and_weight():
    X, T, E, A, cf = _data_and_cf()
    phi, rho = dr_survival(
        cf, event_time=T, event_indicator=E, treatment=A, target_time=5.0
    )
    n = X.shape[0]
    assert phi.shape == (n,)
    assert rho.shape == (n,)


def test_rho_is_constant_one_for_dr():
    X, T, E, A, cf = _data_and_cf()
    _, rho = dr_survival(
        cf, event_time=T, event_indicator=E, treatment=A, target_time=5.0
    )
    np.testing.assert_array_equal(rho, np.ones_like(rho))


def test_finite_under_default_propensity_clip():
    X, T, E, A, cf = _data_and_cf()
    phi, _ = dr_survival(
        cf, event_time=T, event_indicator=E, treatment=A, target_time=5.0
    )
    assert np.isfinite(phi).all()


def test_rejects_missing_per_arm_censoring():
    """dr_survival needs censoring nuisances; cross-fitting without
    censoring_factory should fail at pseudo-outcome time."""
    from sklearn.linear_model import LogisticRegression
    from tausurv.linear import CoxPH

    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 5))
    T = rng.integers(1, 11, size=n).astype(np.float64)
    E = (rng.uniform(size=n) < 0.7).astype(np.int8)
    A = (rng.uniform(size=n) < 0.5).astype(np.int8)
    cf = cross_fit(
        X, T, E, A,
        outcome_factory=lambda: CoxPH(),
        propensity_factory=lambda: LogisticRegression(max_iter=500),
        # no censoring_factory
        times=np.arange(1, 11, dtype=np.float64),
        n_folds=3,
        seed=0,
    )
    with pytest.raises(ValueError, match="per-arm censoring"):
        dr_survival(
            cf, event_time=T, event_indicator=E, treatment=A, target_time=5.0
        )


def test_rejects_target_time_off_grid():
    X, T, E, A, cf = _data_and_cf()
    with pytest.raises(ValueError, match="not in cross-fit times grid"):
        dr_survival(
            cf, event_time=T, event_indicator=E, treatment=A, target_time=99.0
        )


def test_pseudo_outcome_collapses_to_plug_in_when_correction_is_zero():
    r"""Algebraic identity: the DR pseudo-outcome equals the plug-in
    difference $S_1 - S_0$ exactly when the orthogonal correction
    $(A - \pi)\,\xi_S\,S_a / (\pi(1-\pi))$ is zero.

    We construct nuisances where the correction term is forced to zero
    via $\xi_S = 0$: trivially-no-event survival ($S \equiv 1$ across
    the grid implies $\lambda^S \equiv 0$, plus zero observed events).
    Then $\varphi = S_1 - S_0 = 0$ (here both arms have $S \equiv 1$).
    """
    from causurv.nuisances.types import CrossFitNuisances

    n, T_grid_size = 20, 5
    cf = CrossFitNuisances(
        outcome={},
        propensity=None,
        censoring={},
        oof_outcome={
            0: np.ones((n, T_grid_size)),
            1: np.ones((n, T_grid_size)),
        },
        oof_propensity=np.column_stack([np.full(n, 0.5), np.full(n, 0.5)]),
        oof_censoring={
            0: np.ones((n, T_grid_size)),
            1: np.ones((n, T_grid_size)),
        },
        times=np.arange(1, T_grid_size + 1, dtype=np.float64),
        fold_assignment=np.zeros(n, dtype=np.int_),
        n_folds=1,
    )
    T = np.full(n, T_grid_size + 1.0)  # all subjects survive past the grid
    E = np.zeros(n, dtype=np.int8)
    A = np.zeros(n, dtype=np.int_)
    phi, _ = dr_survival(
        cf, event_time=T, event_indicator=E, treatment=A, target_time=3.0
    )
    np.testing.assert_allclose(phi, np.zeros(n), atol=1e-9)


def test_r_survival_returns_pseudo_and_nonconstant_weight():
    """R-learner weight is (A - π)^2, not constant 1."""
    X, T, E, A, cf = _data_and_cf()
    phi, weight = r_survival(
        cf, event_time=T, event_indicator=E, treatment=A, target_time=5.0
    )
    assert phi.shape == weight.shape == (X.shape[0],)
    assert np.isfinite(phi).all() and np.isfinite(weight).all()
    # Weight = (A - π)^2 ∈ [0, 1]; should be non-constant on realistic data.
    assert weight.min() >= 0.0
    assert weight.max() <= 1.0
    assert weight.std() > 0.0


def test_r_survival_weight_equals_a_minus_pi_squared():
    """Sanity: with the propensity OOF clipped per the function, the
    weight returned by r_survival should equal (A - clipped_pi)^2."""
    X, T, E, A, cf = _data_and_cf()
    _, weight = r_survival(
        cf, event_time=T, event_indicator=E, treatment=A, target_time=5.0,
        propensity_clip=1e-2,
    )
    pi = np.clip(cf.oof_propensity[:, 1], 1e-2, 1.0 - 1e-2)
    expected = (A.astype(float) - pi) ** 2
    np.testing.assert_allclose(weight, expected)


def test_dr_and_r_disagree_but_target_same_estimand():
    """DR and R compute different pseudo-outcomes for the same task —
    the weighted-MSE minimisers coincide at the population level but
    the per-sample (pseudo, weight) tuples differ."""
    X, T, E, A, cf = _data_and_cf()
    phi_dr, _ = dr_survival(
        cf, event_time=T, event_indicator=E, treatment=A, target_time=5.0
    )
    phi_r, _ = r_survival(
        cf, event_time=T, event_indicator=E, treatment=A, target_time=5.0
    )
    assert not np.allclose(phi_dr, phi_r)


def test_pseudo_outcome_for_dispatch_table():
    """The PSEUDO_OUTCOME_FOR table exposes named pseudo-outcome
    functions for OrthoLearner's `weighting=` dispatch."""
    assert PSEUDO_OUTCOME_FOR["DR"] is dr_survival
    assert PSEUDO_OUTCOME_FOR["R"] is r_survival


def test_correction_term_picks_up_nuisance_data_mismatch():
    r"""When the OOF survival predicts a drop ($\lambda^S > 0$) but no
    events are observed, the DR correction term *should* fire — the
    pseudo-outcome moves away from the plug-in $S_1 - S_0$ to balance
    the discrepancy. This is the orthogonality property in action.
    """
    from causurv.nuisances.types import CrossFitNuisances

    n, T_grid_size = 20, 5
    # Both arms predict 0.5 survival → hazard 0.5 at time 1; no events
    # actually observed → DR pseudo-outcome moves away from zero.
    cf = CrossFitNuisances(
        outcome={},
        propensity=None,
        censoring={},
        oof_outcome={
            0: np.full((n, T_grid_size), 0.5),
            1: np.full((n, T_grid_size), 0.5),
        },
        oof_propensity=np.column_stack([np.full(n, 0.5), np.full(n, 0.5)]),
        oof_censoring={
            0: np.ones((n, T_grid_size)),
            1: np.ones((n, T_grid_size)),
        },
        times=np.arange(1, T_grid_size + 1, dtype=np.float64),
        fold_assignment=np.zeros(n, dtype=np.int_),
        n_folds=1,
    )
    T = np.full(n, T_grid_size + 1.0)
    E = np.zeros(n, dtype=np.int8)
    A = np.zeros(n, dtype=np.int_)
    phi, _ = dr_survival(
        cf, event_time=T, event_indicator=E, treatment=A, target_time=3.0
    )
    # S_1 - S_0 = 0 (both arms equal); correction picks up the model
    # error (predicted hazard but no observed events) → phi != 0.
    assert (np.abs(phi) > 0.1).all()


def test_propensity_clip_bounds_pseudo_outcome_magnitude():
    """Without clipping, extreme propensities would blow up; clip=0.01
    bounds the (1/π(1-π)) factor to ≤ 1/(0.01 * 0.99) ≈ 101. Strong
    smoke check that values stay bounded on realistic data."""
    X, T, E, A, cf = _data_and_cf(n=400)
    phi, _ = dr_survival(
        cf, event_time=T, event_indicator=E, treatment=A, target_time=5.0,
        propensity_clip=0.01,
    )
    # Pseudo-outcome is bounded by 1 (plug-in) + (1/(π(1-π))) * 1 (max
    # of ξ_S * S_a in [0, 1]). With clip=0.01 the bound is ~1 + 101.
    assert np.abs(phi).max() < 150.0
