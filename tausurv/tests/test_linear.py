from __future__ import annotations

import numpy as np
import pytest

from tausurv import simulations
from tausurv.linear import CoxPH


def test_coxph_recovers_simulated_coefficients():
    X, T, E = simulations.single_risk(n=5000, n_features=5, censoring_rate=0.2, seed=0)
    cph = CoxPH().fit(X, T, E)
    expected = simulations._default_coefficients(5)
    np.testing.assert_allclose(cph.coef_, expected, atol=0.1)


def test_coxph_predict_survival_is_decreasing_and_in_unit_interval():
    X, T, E = simulations.single_risk(n=200, seed=0)
    cph = CoxPH().fit(X, T, E)
    grid = np.linspace(0.05, 5.0, 20)
    S = cph.predict_survival_function(X, grid)
    assert S.shape == (200, 20)
    assert ((S >= 0) & (S <= 1)).all()
    # Survival is right-continuous step from baseline H; non-increasing in t.
    assert (np.diff(S, axis=1) <= 0).all()


def test_coxph_predict_correlates_with_event_time():
    X, T, E = simulations.single_risk(n=500, seed=0)
    cph = CoxPH().fit(X, T, E)
    risk = cph.predict(X)
    assert risk.shape == (500,)
    # Higher predicted risk -> shorter event time.
    assert np.corrcoef(risk, T)[0, 1] < 0


def test_coxph_predict_cif_complements_survival():
    X, T, E = simulations.single_risk(n=50, seed=0)
    cph = CoxPH().fit(X, T, E)
    grid = np.linspace(0.1, 3.0, 5)
    S = cph.predict_survival_function(X, grid)
    F = cph.predict_cif(X, grid)
    np.testing.assert_allclose(F, 1.0 - S)


def test_coxph_predict_cif_rejects_invalid_cause():
    X, T, E = simulations.single_risk(n=50, seed=0)
    cph = CoxPH().fit(X, T, E)
    with pytest.raises(ValueError, match="single-event"):
        cph.predict_cif(X, [1.0], cause=2)


def test_coxph_rejects_non_binary_event_indicator():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    T = rng.exponential(1, 100)
    E = rng.choice([0, 1, 2], 100)
    with pytest.raises(ValueError, match="single-event"):
        CoxPH().fit(X, T, E)


def test_coxph_matches_sksurv():
    sksurv_linear = pytest.importorskip("sksurv.linear_model")
    X, T, E = simulations.single_risk(n=500, n_features=5, censoring_rate=0.3, seed=0)

    cph = CoxPH().fit(X, T, E)

    structured = np.array(
        list(zip(E.astype(bool), T)),
        dtype=[("event", bool), ("time", float)],
    )
    sksurv_cph = sksurv_linear.CoxPHSurvivalAnalysis(ties="breslow")
    sksurv_cph.fit(X, structured)

    np.testing.assert_allclose(cph.coef_, sksurv_cph.coef_, atol=5e-3)


def test_coxph_save_load_preserves_predictions(tmp_path):
    X, T, E = simulations.single_risk(n=200, seed=0)
    cph = CoxPH().fit(X, T, E)
    cph.save(tmp_path / "cph")
    loaded = CoxPH.load(tmp_path / "cph")

    assert loaded.max_iter == cph.max_iter and loaded.tol == cph.tol
    grid = np.linspace(0.05, 5.0, 20)
    np.testing.assert_array_equal(loaded.times_, cph.times_)
    np.testing.assert_allclose(
        loaded.predict_survival_function(X, grid),
        cph.predict_survival_function(X, grid),
    )
    np.testing.assert_allclose(loaded.predict(X), cph.predict(X))


def test_coxph_unfitted_save_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="not fitted"):
        CoxPH().save(tmp_path / "cph")


def test_coxph_standard_errors_match_lifelines():
    lifelines = pytest.importorskip("lifelines")
    pd = pytest.importorskip("pandas")
    # Continuous times, so Breslow and lifelines' Efron tie handling coincide.
    X, T, E = simulations.single_risk(n=500, n_features=5, censoring_rate=0.3, seed=1)

    cph = CoxPH(tol=1e-9).fit(X, T, E)

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["T"], df["E"] = T, E
    cpf = lifelines.CoxPHFitter().fit(df, duration_col="T", event_col="E")

    np.testing.assert_allclose(cph.coef_, cpf.params_.to_numpy(), atol=1e-4)
    np.testing.assert_allclose(
        cph.standard_errors_, cpf.standard_errors_.to_numpy(), rtol=1e-4
    )
    np.testing.assert_allclose(
        cph.covariance_, cpf.variance_matrix_.to_numpy(), rtol=1e-4, atol=1e-8
    )


def test_coxph_confidence_intervals_are_wald():
    X, T, E = simulations.single_risk(n=300, n_features=3, seed=2)
    cph = CoxPH().fit(X, T, E)

    lo, hi = cph.confidence_intervals(level=0.95)
    assert np.all(lo < cph.coef_) and np.all(cph.coef_ < hi)
    np.testing.assert_allclose(hi - lo, 2 * 1.959963984540054 * cph.standard_errors_)

    lo90, hi90 = cph.confidence_intervals(level=0.90)
    assert np.all(hi90 - lo90 < hi - lo)


def test_coxph_confidence_intervals_reject_bad_level():
    X, T, E = simulations.single_risk(n=100, n_features=2, seed=3)
    cph = CoxPH().fit(X, T, E)
    with pytest.raises(ValueError, match="level"):
        cph.confidence_intervals(level=1.5)


def test_coxph_save_load_preserves_standard_errors(tmp_path):
    X, T, E = simulations.single_risk(n=200, seed=0)
    cph = CoxPH().fit(X, T, E)
    cph.save(tmp_path / "cph")
    loaded = CoxPH.load(tmp_path / "cph")

    np.testing.assert_allclose(loaded.covariance_, cph.covariance_)
    np.testing.assert_allclose(loaded.standard_errors_, cph.standard_errors_)
