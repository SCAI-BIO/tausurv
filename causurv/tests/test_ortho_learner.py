"""Tests for OrthoLearner + DRLearner (Frauen 2025 DR-survival).

Mechanics-only: shape, contracts, validation, reproducibility.
Oracle-recovery benchmarks (does PEHE converge on the SurvITE simulator?)
live in ``scripts/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from causurv.estimands import CIFDiff, RMSTDiff, SurvivalDiff
from causurv.learners import DRLearner, OrthoLearner, RLearner
from causurv.predictor import HTEEstimates, HTEPredictor


def _data(n=300, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    T = rng.integers(1, 16, size=n).astype(np.float64)
    E = (rng.uniform(size=n) < 0.7).astype(np.int8)
    A = (rng.uniform(size=n) < 0.5).astype(np.int8)
    return X, T, E, A


def _factories():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from tausurv.linear import CoxPH

    return dict(
        outcome_factory=lambda: CoxPH(),
        propensity_factory=lambda: LogisticRegression(max_iter=500),
        censoring_factory=lambda: CoxPH(),
        second_stage_factory=lambda: RandomForestRegressor(
            n_estimators=20, max_depth=4, random_state=0
        ),
    )


def _model(times=(5, 10), **kw):
    factories = _factories()
    defaults = dict(
        estimand=SurvivalDiff(times=list(times)),
        n_folds=3,
        seed=0,
        **factories,
    )
    defaults.update(kw)
    return OrthoLearner(**defaults)


def test_inherits_hte_predictor():
    assert isinstance(_model(), HTEPredictor)


def test_fit_returns_self():
    X, T, E, A = _data()
    ol = _model()
    assert ol.fit(X, T, E, A) is ol


def test_fit_sets_times_attribute():
    X, T, E, A = _data()
    ol = _model(times=(5, 10, 15)).fit(X, T, E, A)
    np.testing.assert_array_equal(ol.times_, np.array([5.0, 10.0, 15.0]))


def test_predict_hte_returns_hteestimates_with_right_shape():
    X, T, E, A = _data(n=200)
    ol = _model(times=(5, 10)).fit(X, T, E, A)
    hte = ol.predict_hte(X[:30])
    assert isinstance(hte, HTEEstimates)
    assert hte.values.shape == (30, 2)
    assert hte.contrast == "survival_diff"
    assert hte.treatment == 1 and hte.reference == 0
    np.testing.assert_array_equal(hte.estimand.times, np.array([5.0, 10.0]))


def test_predict_potential_outcomes_raises():
    X, T, E, A = _data(n=200)
    ol = _model().fit(X, T, E, A)
    with pytest.raises(NotImplementedError, match="contrast"):
        ol.predict_potential_outcomes(X[:5])


def test_predict_ate_uses_stored_training_X():
    X, T, E, A = _data(n=200)
    ol = _model(times=(5, 10)).fit(X, T, E, A)
    ate = ol.predict_ate()  # defaults to training X
    assert ate.shape == (2,)


def test_predict_before_fit_raises():
    ol = _model()
    with pytest.raises(RuntimeError, match="fit"):
        ol.predict_hte(np.zeros((3, 6)))


def test_rejects_non_survival_diff_estimand():
    factories = _factories()
    with pytest.raises(TypeError, match="SurvivalDiff"):
        OrthoLearner(estimand=RMSTDiff(times=[5, 10], horizon=10), **factories)
    with pytest.raises(TypeError, match="SurvivalDiff"):
        OrthoLearner(estimand=CIFDiff(times=[5, 10], cause=1), **factories)


def test_rejects_non_binary_treatment_reference():
    factories = _factories()
    with pytest.raises(ValueError, match="binary-treatment"):
        OrthoLearner(
            estimand=SurvivalDiff(times=[5, 10], treatment=0, reference=1),
            **factories,
        )


def test_rejects_non_binary_data():
    rng = np.random.default_rng(0)
    n, d = 200, 6
    X = rng.normal(size=(n, d))
    T = rng.integers(1, 16, size=n).astype(np.float64)
    E = (rng.uniform(size=n) < 0.7).astype(np.int8)
    A = rng.choice([0, 1, 2], size=n).astype(np.int8)  # 3 arms
    ol = _model()
    with pytest.raises(ValueError, match="binary-treatment"):
        ol.fit(X, T, E, A)


def test_rejects_target_times_below_one():
    factories = _factories()
    with pytest.raises(ValueError, match="times must be positive"):
        OrthoLearner(estimand=SurvivalDiff(times=[0, 5]), **factories)


def test_rejects_unsupported_weighting():
    factories = _factories()
    with pytest.raises(ValueError, match="weighting must be one of"):
        OrthoLearner(
            estimand=SurvivalDiff(times=[5]), weighting="C", **factories
        )


def test_dr_learner_is_an_ortho_learner_with_dr_weighting():
    factories = _factories()
    dr = DRLearner(estimand=SurvivalDiff(times=[5]), **factories)
    assert isinstance(dr, OrthoLearner)
    assert dr.weighting == "DR"


def test_r_learner_is_an_ortho_learner_with_r_weighting():
    factories = _factories()
    r = RLearner(estimand=SurvivalDiff(times=[5]), **factories)
    assert isinstance(r, OrthoLearner)
    assert r.weighting == "R"


def test_r_learner_predicts_with_right_shape():
    X, T, E, A = _data(n=200)
    r = RLearner(
        estimand=SurvivalDiff(times=[5, 10]), n_folds=3, seed=0, **_factories()
    ).fit(X, T, E, A)
    hte = r.predict_hte(X[:30])
    assert isinstance(hte, HTEEstimates)
    assert hte.values.shape == (30, 2)
    assert hte.contrast == "survival_diff"


def test_r_and_dr_produce_different_predictions():
    """Both target tau_t(x), but the pseudo-outcomes differ → second-stage
    fits differ → predictions differ on finite samples."""
    X, T, E, A = _data(n=200)
    common = dict(
        estimand=SurvivalDiff(times=[5, 10]), n_folds=3, seed=0, **_factories()
    )
    dr = DRLearner(**common).fit(X, T, E, A)
    r = RLearner(**common).fit(X, T, E, A)
    assert not np.allclose(dr.predict_hte(X[:20]).values, r.predict_hte(X[:20]).values)


def test_seeded_runs_are_reproducible():
    X, T, E, A = _data(n=200)
    a = _model(seed=42).fit(X, T, E, A).predict_hte(X[:10])
    b = _model(seed=42).fit(X, T, E, A).predict_hte(X[:10])
    np.testing.assert_allclose(a.values, b.values, atol=1e-10)


def test_oof_nuisances_carried_through_fit():
    """OrthoLearner caches the CrossFitNuisances so downstream consumers
    (diagnostics, plots) can inspect the OOF estimates."""
    from causurv.nuisances import CrossFitNuisances

    X, T, E, A = _data(n=200)
    ol = _model(times=(5, 10)).fit(X, T, E, A)
    assert isinstance(ol._cf, CrossFitNuisances)
    assert sorted(ol._cf.oof_outcome) == [0, 1]
    assert sorted(ol._cf.oof_censoring) == [0, 1]


def test_predict_hte_values_finite():
    """Even on small data, second-stage predictions must be finite —
    the propensity_clip default should bound the DR pseudo-outcome."""
    X, T, E, A = _data(n=300)
    ol = _model(times=(5, 10)).fit(X, T, E, A)
    hte = ol.predict_hte(X[:50])
    assert np.isfinite(hte.values).all()
