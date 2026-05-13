"""Tests for the SurvITE benchmark simulator.

Covers (a) generator mechanics — shapes, dtypes, scenario flags,
reproducibility — and (b) the oracle predictor — survival monotonicity,
HTE arithmetic, integration with ``HTEPredictor``. Recovery-style
checks (does T-learner converge to the oracle HTE?) live in
``scripts/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from causurv.predictor import HTEEstimates, HTEPredictor
from causurv.simulations import SurvITE


def test_default_is_s4_treated_and_censored():
    sim = SurvITE()
    assert sim.scenario == "S4"
    assert sim.treated is True
    assert sim.censored is True


@pytest.mark.parametrize(
    "scenario,treated,censored",
    [
        ("S1", False, False),
        ("S2", False, True),
        ("S3", True, False),
        ("S4", True, True),
    ],
)
def test_scenario_flags(scenario, treated, censored):
    sim = SurvITE(scenario=scenario)
    assert (sim.treated, sim.censored) == (treated, censored)


def test_invalid_scenario_raises():
    with pytest.raises(ValueError, match="scenario must be one of"):
        SurvITE(scenario="S5")


def test_n_features_must_be_at_least_4():
    with pytest.raises(ValueError, match="n_features must be >= 4"):
        SurvITE(n_features=3)


def test_t_max_must_be_at_least_2():
    with pytest.raises(ValueError, match="t_max must be >= 2"):
        SurvITE(t_max=1)


def test_rho_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="rho must be in"):
        SurvITE(rho=-0.1)
    with pytest.raises(ValueError, match="rho must be in"):
        SurvITE(rho=1.0)


def test_treatment_features_depend_on_overlap_flag():
    assert SurvITE(overlap_treat=True, n_features=10)._P == (0, 1)
    assert SurvITE(overlap_treat=False, n_features=10)._P == (8, 9)
    # The non-overlap case must shift with n_features.
    assert SurvITE(overlap_treat=False, n_features=6)._P == (4, 5)


def test_inherits_hte_predictor():
    assert isinstance(SurvITE(), HTEPredictor)


def test_generate_shapes_and_dtypes():
    sim = SurvITE(scenario="S4", n_features=8, t_max=20)
    X, T, E, A = sim.generate(n=300, seed=0)
    assert X.shape == (300, 8) and X.dtype == np.float64
    assert T.shape == (300,) and T.dtype == np.float64
    assert E.shape == (300,) and E.dtype == np.int8
    assert A.shape == (300,) and A.dtype == np.int8
    assert set(np.unique(E)).issubset({0, 1})
    assert set(np.unique(A)).issubset({0, 1})


def test_generate_event_times_are_integers_within_horizon():
    sim = SurvITE(scenario="S4", t_max=15)
    _, T, _, _ = sim.generate(n=200, seed=0)
    assert (T == np.floor(T)).all()
    assert T.min() >= 1.0
    assert T.max() <= 15.0


def test_s1_has_no_treatment_and_no_censoring():
    """S1 ⇒ A ≡ 0, and E == 1 except for administrative censoring at t_max."""
    sim = SurvITE(scenario="S1", t_max=20)
    _, T, E, A = sim.generate(n=400, seed=0)
    assert (A == 0).all()
    # All non-administratively-censored events have E == 1.
    assert ((T < sim.t_max) <= (E == 1)).all()


def test_s2_has_no_treatment_but_can_censor():
    sim = SurvITE(scenario="S2", t_max=20)
    _, _, E, A = sim.generate(n=400, seed=0)
    assert (A == 0).all()
    assert (E == 0).any()  # some censoring occurs


def test_s3_treats_but_doesnt_censor_informatively():
    """In S3 the only censoring source is the administrative t_max."""
    sim = SurvITE(scenario="S3", t_max=20)
    _, T, E, _ = sim.generate(n=400, seed=0)
    # Any censored subject must be censored at t_max (admin only).
    assert (T[E == 0] >= sim.t_max).all()


def test_s4_has_both_treatment_and_informative_censoring():
    sim = SurvITE(scenario="S4")
    _, _, E, A = sim.generate(n=400, seed=0)
    assert (A == 1).any() and (A == 0).any()
    assert (E == 0).any()


def test_seed_reproducibility():
    sim = SurvITE(scenario="S4", n_features=6, t_max=15)
    out_a = sim.generate(n=120, seed=42)
    out_b = sim.generate(n=120, seed=42)
    for arr_a, arr_b in zip(out_a, out_b):
        np.testing.assert_array_equal(arr_a, arr_b)


def test_different_seeds_yield_different_data():
    sim = SurvITE(scenario="S4")
    X1, *_ = sim.generate(n=100, seed=0)
    X2, *_ = sim.generate(n=100, seed=1)
    assert not np.allclose(X1, X2)


def test_oracle_survival_shape_matches_times():
    sim = SurvITE(scenario="S4")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, sim.n_features))
    times = np.array([1.0, 5.0, 10.0, 20.0, 30.0])
    S0, S1 = sim.predict_potential_outcomes(X, times)
    assert S0.shape == S1.shape == (50, 5)


def test_oracle_survival_is_in_unit_interval_and_monotone():
    sim = SurvITE(scenario="S4")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, sim.n_features))
    S0, S1 = sim.predict_potential_outcomes(X)
    assert ((S0 >= 0.0) & (S0 <= 1.0)).all()
    assert ((S1 >= 0.0) & (S1 <= 1.0)).all()
    # Monotone non-increasing in t.
    assert (np.diff(S0, axis=1) <= 1e-12).all()
    assert (np.diff(S1, axis=1) <= 1e-12).all()


def test_oracle_survival_at_t_below_one_is_one():
    sim = SurvITE(scenario="S4")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, sim.n_features))
    S0, S1 = sim.predict_potential_outcomes(X, times=[0.0, 0.5])
    np.testing.assert_array_equal(S0, np.ones((20, 2)))
    np.testing.assert_array_equal(S1, np.ones((20, 2)))


def test_oracle_survival_default_times_is_integer_grid():
    sim = SurvITE(scenario="S4", t_max=10)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, sim.n_features))
    S0, S1 = sim.predict_potential_outcomes(X)
    assert S0.shape == S1.shape == (5, 10)


def test_oracle_potential_outcomes_rejects_cause():
    sim = SurvITE()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, sim.n_features))
    with pytest.raises(ValueError, match="single-event"):
        sim.predict_potential_outcomes(X, cause=1)


def test_oracle_predict_hte_returns_hteestimates():
    from causurv.estimands import SurvivalDiff

    sim = SurvITE(scenario="S4")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, sim.n_features))
    times = np.array([5.0, 10.0, 20.0])
    hte = sim.predict_hte(X, estimand=SurvivalDiff(times=times))
    assert isinstance(hte, HTEEstimates)
    assert hte.values.shape == (10, 3)
    assert hte.contrast == "survival_diff"
    assert (hte.treatment, hte.reference) == (1, 0)


def test_oracle_predict_hte_equals_manual_subtraction():
    from causurv.estimands import SurvivalDiff

    sim = SurvITE(scenario="S4")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, sim.n_features))
    times = np.linspace(1.0, 20.0, 5)
    S0, S1 = sim.predict_potential_outcomes(X, times)
    hte = sim.predict_hte(X, estimand=SurvivalDiff(times=times))
    np.testing.assert_allclose(hte.values, S1 - S0)


def test_oracle_rejects_non_binary_arm_request():
    from causurv.estimands import SurvivalDiff

    sim = SurvITE()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, sim.n_features))
    with pytest.raises(ValueError, match="binary-treatment"):
        sim.predict_hte(
            X, estimand=SurvivalDiff(times=[5.0], treatment=2, reference=0)
        )


def test_oracle_treatment_changes_hazard_when_x2_nonnegative():
    """The hazard adjustment is `A * ((X[:,2] >= 0) + 0.5)` — under A=1 the
    sigmoid argument is smaller, so the hazard is *lower* and survival
    *higher* on subjects with X[:, 2] >= 0."""
    sim = SurvITE(scenario="S4")
    X = np.zeros((4, sim.n_features))
    X[:, 2] = np.array([1.0, 1.0, -1.0, -1.0])
    S0, S1 = sim.predict_potential_outcomes(X, times=[10.0])
    # Rows 0,1 (X[:,2] >= 0): treatment helps ⇒ S1 > S0.
    assert (S1[:2, 0] > S0[:2, 0]).all()


def test_hazard_is_in_unit_interval():
    sim = SurvITE(scenario="S4")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, sim.n_features))
    for a in (0, 1):
        h = sim.hazard(X, treatment=a, event=True)
        assert h.shape == (50, sim.t_max)
        assert ((h >= 0.0) & (h <= 1.0)).all()


def test_censoring_hazard_last_column_is_one():
    sim = SurvITE(scenario="S4")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, sim.n_features))
    h = sim.hazard(X, event=False)
    np.testing.assert_array_equal(h[:, -1], np.ones(10))


def test_fit_raises():
    sim = SurvITE()
    with pytest.raises(NotImplementedError, match="simulator/oracle"):
        sim.fit(None, None, None, None)


def test_predict_ate_uses_last_generated_X():
    sim = SurvITE(scenario="S4")
    sim.generate(n=200, seed=0)
    ate = sim.predict_ate(estimand="survival_diff", times=[5.0, 10.0, 20.0])
    assert ate.shape == (3,)


def test_predict_ate_without_generate_raises():
    sim = SurvITE()
    with pytest.raises(RuntimeError, match="prior generate"):
        sim.predict_ate(estimand="survival_diff", times=[5.0])
