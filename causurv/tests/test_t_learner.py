"""Tests for the T-learner.

Verifies the meta-learner mechanics — one base model per arm, fits on
arm-specific subsets, predicts on shared time grid. Recovery-style
tests live in ``scripts/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from causurv.learners import TLearner
from causurv.predictor import HTEPredictor


def _data(n=400, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    T = rng.exponential(1.0, size=n).astype(np.float64) + 0.1
    E = (rng.uniform(size=n) < 0.7).astype(np.int8)
    A = (rng.uniform(size=n) < 0.5).astype(np.int8)
    return X, T, E, A


def test_inherits_hte_predictor():
    from tausurv.linear import CoxPH

    tl = TLearner(outcome_factory=lambda: CoxPH())
    assert isinstance(tl, HTEPredictor)


def test_fit_creates_one_model_per_arm():
    from tausurv.linear import CoxPH

    X, T, E, A = _data()
    tl = TLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)
    assert set(tl._models) == {0, 1}


def test_arm_models_see_only_their_arms_data():
    """A model_0 fit on A==0 data must NOT be the same as model_1 fit on A==1.

    Verify by comparing on-disk coefficients (Cox is deterministic
    given data) between an arm's model in T-learner and a separately
    fit Cox on the same arm's subset.
    """
    from tausurv.linear import CoxPH

    X, T, E, A = _data()
    tl = TLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)
    # Independently fit CoxPH on arm 0.
    cox_arm0 = CoxPH().fit(X[A == 0], T[A == 0], E[A == 0])
    np.testing.assert_allclose(tl._models[0].coef_, cox_arm0.coef_, atol=1e-10)


def test_potential_outcomes_shape_binary():
    from tausurv.linear import CoxPH

    X, T, E, A = _data(n=200, d=3)
    tl = TLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)
    times = np.linspace(0.2, 3.0, 8)
    arms = tl.predict_potential_outcomes(X[:30], times)
    assert len(arms) == 2
    for arm in arms:
        assert arm.shape == (30, 8)
        assert ((arm >= 0) & (arm <= 1)).all()


def test_both_arms_predict_on_same_time_grid():
    """The two arms' predictions must share the same time axis when
    `times` is explicitly given — that's the whole point of the union
    of times_."""
    from tausurv.linear import CoxPH

    X, T, E, A = _data()
    tl = TLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)
    times = np.array([0.5, 1.0, 1.5, 2.0])
    s0, s1 = tl.predict_potential_outcomes(X[:10], times)
    assert s0.shape == s1.shape == (10, 4)


def test_times_is_union_of_arm_times():
    """`self.times_` is the union of per-arm training event times so
    default-grid predictions cover both arms' supports."""
    from tausurv.linear import CoxPH

    X, T, E, A = _data(n=300)
    tl = TLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)

    # Construct the expected union manually.
    times_arm0 = CoxPH().fit(X[A == 0], T[A == 0], E[A == 0]).times_
    times_arm1 = CoxPH().fit(X[A == 1], T[A == 1], E[A == 1]).times_
    expected = np.unique(np.concatenate([times_arm0, times_arm1]))
    np.testing.assert_array_equal(tl.times_, expected)


def test_predict_hte_dispatch_returns_hteestimates():
    from tausurv.linear import CoxPH

    X, T, E, A = _data()
    tl = TLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)
    times = np.linspace(0.5, 2.0, 4)
    hte = tl.predict_hte(X[:5], estimand="survival_diff", times=times)
    assert hte.values.shape == (5, 4)
    assert hte.contrast == "survival_diff"


def test_predict_hte_matches_manual_subtraction():
    """Verify the meta-learner correctly chains to ``apply_contrast`` —
    survival_diff is just S_treat - S_ref."""
    from tausurv.linear import CoxPH

    X, T, E, A = _data()
    tl = TLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)
    times = np.linspace(0.5, 2.0, 4)
    s0, s1 = tl.predict_potential_outcomes(X[:5], times)
    hte = tl.predict_hte(X[:5], estimand="survival_diff", times=times)
    np.testing.assert_allclose(hte.values, s1 - s0)


def test_works_with_different_base_models():
    """Same TLearner code, different bases — factory pattern."""
    from tausurv.linear import WeibullAFT
    from tausurv.trees import RandomSurvivalForest

    X, T, E, A = _data(n=200)
    times = np.linspace(0.5, 2.0, 4)

    for factory in (
        lambda: WeibullAFT(),
        lambda: RandomSurvivalForest(n_estimators=10, seed=0),
    ):
        tl = TLearner(outcome_factory=factory).fit(X, T, E, A)
        arms = tl.predict_potential_outcomes(X[:5], times)
        assert len(arms) == 2
        for arm in arms:
            assert arm.shape == (5, 4)


def test_competing_risks_base_via_cause():
    """With a CR-aware base (FineGray), predict_cif via cause= works."""
    from tausurv.linear import FineGray
    from tausurv import simulations

    X, T, E = simulations.competing_risk(n=400, n_features=4, n_causes=2, seed=0)
    rng = np.random.default_rng(0)
    A = (rng.uniform(size=X.shape[0]) < 0.5).astype(np.int8)

    tl = TLearner(outcome_factory=lambda: FineGray(cause=1)).fit(X, T, E, A)
    grid = np.linspace(0.1, 2.0, 6)
    cif_arms = tl.predict_potential_outcomes(X[:10], grid, cause=1)
    assert len(cif_arms) == 2
    for arm in cif_arms:
        assert arm.shape == (10, 6)
        assert ((arm >= 0) & (arm <= 1)).all()


def test_multi_arm_extension():
    """A ∈ {0, 1, 2} — same code, three models, three-arm potential outcomes."""
    from tausurv.linear import CoxPH

    rng = np.random.default_rng(0)
    n, d = 300, 4
    X = rng.normal(size=(n, d))
    T = rng.exponential(1.0, size=n) + 0.1
    E = (rng.uniform(size=n) < 0.7).astype(np.int8)
    A = rng.choice([0, 1, 2], size=n).astype(np.int8)

    tl = TLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)
    assert set(tl._models) == {0, 1, 2}
    arms = tl.predict_potential_outcomes(X[:5])
    assert len(arms) == 3
    # Predict 2-vs-0 contrast.
    from causurv.estimands import SurvivalDiff
    hte = tl.predict_hte(
        X[:5],
        estimand=SurvivalDiff(times=tl.times_, treatment=2, reference=0),
    )
    assert hte.treatment == 2 and hte.reference == 0


def test_validates_dimensions():
    from tausurv.linear import CoxPH

    tl = TLearner(outcome_factory=lambda: CoxPH())
    X = np.zeros(10)  # 1D
    with pytest.raises(ValueError, match="X must be 2D"):
        tl.fit(X, np.ones(10), np.ones(10, dtype=np.int8), np.zeros(10, dtype=np.int8))

    X = np.zeros((10, 3))
    with pytest.raises(ValueError, match="first axis"):
        tl.fit(X, np.ones(10), np.ones(10, dtype=np.int8), np.zeros(5, dtype=np.int8))

    with pytest.raises(ValueError, match="non-negative"):
        tl.fit(X, np.ones(10), np.ones(10, dtype=np.int8), -np.ones(10, dtype=np.int8))


def test_predict_before_fit_raises():
    from tausurv.linear import CoxPH

    tl = TLearner(outcome_factory=lambda: CoxPH())
    with pytest.raises(RuntimeError, match="fit"):
        tl.predict_potential_outcomes(np.zeros((3, 4)))


def test_ate_without_X_uses_stored_training_X():
    from tausurv.linear import CoxPH

    X, T, E, A = _data(n=120)
    tl = TLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)
    times = np.linspace(0.2, 2.0, 5)
    ate = tl.predict_ate(estimand="survival_diff", times=times)
    assert ate.shape == (5,)


def test_independent_arm_models_not_s_learner():
    """T-learner ≠ S-learner: T-learner's model_0 is fit ONLY on A==0
    subjects; the model does not know about A at all."""
    from tausurv.linear import CoxPH

    X, T, E, A = _data(n=200)
    tl = TLearner(outcome_factory=lambda: CoxPH()).fit(X, T, E, A)

    # Model_0's coef_ has X.shape[1] entries (no treatment column).
    assert tl._models[0].coef_.shape == (X.shape[1],)
