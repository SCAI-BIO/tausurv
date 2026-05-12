"""Tests for the S-learner.

Verifies the meta-learner mechanics — factory pattern works with any
``SurvivalPredictor``, fit/predict shapes, cause= dispatch for CR base
models. Recovery-style tests (does S-learner produce sensible HTE on
simulated data?) live in ``scripts/``, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from causurv.learners import SLearner
from causurv.predictor import HTEPredictor


def _data(n=200, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    T = rng.exponential(1.0, size=n).astype(np.float64) + 0.1
    E = (rng.uniform(size=n) < 0.7).astype(np.int8)
    A = (rng.uniform(size=n) < 0.5).astype(np.int8)
    return X, T, E, A


def test_inherits_hte_predictor():
    from tausurv.linear import CoxPH

    sl = SLearner(outcome_factory=lambda: CoxPH())
    assert isinstance(sl, HTEPredictor)


def test_fit_returns_self_and_caches_X_for_default_ate():
    """fit() returns self (sklearn convention) and caches X so
    predict_ate() can default to the training sample."""
    from tausurv.linear import CoxPH

    X, T, E, A = _data()
    sl = SLearner(outcome_factory=lambda: CoxPH())
    out = sl.fit(X, T, E, A)
    assert out is sl
    # predict_ate without X uses the cached training set.
    ate = sl.predict_ate(times=np.array([0.5, 1.0]))
    assert ate.shape == (2,)


def test_potential_outcomes_shape_binary_treatment():
    from tausurv.linear import CoxPH

    X, T, E, A = _data(n=150, d=3)
    sl = SLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)
    times = np.linspace(0.2, 3.0, 10)
    arms = sl.predict_potential_outcomes(X[:30], times)
    assert len(arms) == 2  # binary
    for arm in arms:
        assert arm.shape == (30, 10)
        assert ((arm >= 0) & (arm <= 1)).all()


def test_potential_outcomes_pass_treatment_as_feature():
    """The S-learner appends A as the last feature. Verify by comparing
    its predictions for arm=0 vs the base model directly queried with
    A=0 appended."""
    from tausurv.linear import CoxPH

    X, T, E, A = _data()
    sl = SLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)
    times = np.linspace(0.2, 3.0, 5)

    # Manual: query the underlying base model with X | A=0.
    X_test = X[:10]
    XA_arm0 = np.hstack([X_test, np.zeros((10, 1))])
    s0_direct = sl._model.predict_survival_function(XA_arm0, times)

    # Via the meta-learner.
    s0_meta = sl.predict_potential_outcomes(X_test, times)[0]
    np.testing.assert_allclose(s0_meta, s0_direct)


def test_hte_call_through_inherits_contrast_dispatch():
    """The base class' predict_hte should work end-to-end."""
    from tausurv.linear import CoxPH

    X, T, E, A = _data()
    sl = SLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)
    times = np.linspace(0.2, 3.0, 5)
    hte = sl.predict_hte(X[:5], times, contrast="survival_diff")
    assert hte.contrast == "survival_diff"
    assert hte.values.shape == (5, 5)


def test_ate_uses_stored_training_X_when_X_omitted():
    from tausurv.linear import CoxPH

    X, T, E, A = _data(n=80)
    sl = SLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)
    times = np.linspace(0.2, 3.0, 4)
    ate = sl.predict_ate(times=times)
    assert ate.shape == (4,)


def test_works_with_aft_base_model():
    """Different base model — verifies the factory pattern is not
    Cox-specific."""
    from tausurv.linear import WeibullAFT

    X, T, E, A = _data(n=120)
    sl = SLearner(outcome_factory=lambda: WeibullAFT()).fit(X, T, E, A)
    times = np.linspace(0.2, 3.0, 6)
    hte = sl.predict_hte(X[:5], times)
    assert hte.values.shape == (5, 6)


def test_works_with_rsf_base_model():
    from tausurv.trees import RandomSurvivalForest

    X, T, E, A = _data(n=120, d=3)
    sl = SLearner(outcome_factory=lambda: RandomSurvivalForest(n_estimators=10, seed=0))
    sl.fit(X, T, E, A)
    times = np.linspace(0.2, 3.0, 5)
    hte = sl.predict_hte(X[:5], times)
    assert hte.values.shape == (5, 5)


def test_validates_dimensions():
    from tausurv.linear import CoxPH

    sl = SLearner(outcome_factory=lambda: CoxPH())
    X = np.zeros(10)  # 1D, invalid
    with pytest.raises(ValueError, match="X must be 2D"):
        sl.fit(X, np.ones(10), np.ones(10, dtype=np.int8), np.zeros(10, dtype=np.int8))

    X = np.zeros((10, 3))
    with pytest.raises(ValueError, match="treatment must be shape"):
        sl.fit(X, np.ones(10), np.ones(10, dtype=np.int8), np.zeros(5, dtype=np.int8))

    with pytest.raises(ValueError, match="non-negative"):
        sl.fit(X, np.ones(10), np.ones(10, dtype=np.int8), -np.ones(10, dtype=np.int8))


def test_predict_before_fit_raises():
    from tausurv.linear import CoxPH

    sl = SLearner(outcome_factory=lambda: CoxPH())
    with pytest.raises(RuntimeError, match="fit"):
        sl.predict_potential_outcomes(np.zeros((3, 4)))


def test_multi_arm_detected_from_treatment_values():
    """If A has values {0, 1, 2}, n_arms is 3 — the API generalises
    without changes."""
    from tausurv.linear import CoxPH

    rng = np.random.default_rng(0)
    n, d = 150, 3
    X = rng.normal(size=(n, d))
    T = rng.exponential(1.0, size=n) + 0.1
    E = (rng.uniform(size=n) < 0.7).astype(np.int8)
    A = rng.choice([0, 1, 2], size=n).astype(np.int8)

    sl = SLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)
    assert sl._n_arms == 3
    arms = sl.predict_potential_outcomes(X[:5])
    assert len(arms) == 3
