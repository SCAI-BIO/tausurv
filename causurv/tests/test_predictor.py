"""Tests for the HTEPredictor protocol via a stub subclass.

Verifies that :meth:`predict_hte` and :meth:`predict_ate` correctly
resolve estimands (instance / class / string), dispatch contrasts, and
slice arms by treatment/reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from causurv.estimands import (
    CIFDiff,
    RMSTDiff,
    SurvivalDiff,
    SurvivalRatio,
)
from causurv.nuisances import Nuisances
from causurv.predictor import HTEEstimates, HTEPredictor


class _StubLearner(HTEPredictor):
    """Returns fixed potential outcomes; no actual fitting."""

    def __init__(self, *, arms_survival, arms_cif=None, times):
        self._arms_survival = arms_survival
        self._arms_cif = arms_cif or {}
        self.times_ = np.asarray(times, dtype=np.float64)
        self._fit_X = np.zeros((arms_survival[0].shape[0], 2))

    def fit(self, X, T, E, A):
        return self

    def predict_potential_outcomes(self, X, times=None, *, cause=None):
        if cause is None:
            return tuple(self._arms_survival)
        return tuple(self._arms_cif[cause])


def _stub_binary():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    s_ref = np.array(
        [
            [0.95, 0.85, 0.70, 0.50],
            [0.90, 0.75, 0.55, 0.30],
        ]
    )
    s_treat = np.array(
        [
            [0.97, 0.90, 0.80, 0.65],
            [0.95, 0.85, 0.70, 0.50],
        ]
    )
    return _StubLearner(arms_survival=[s_ref, s_treat], times=times)


def _stub_with_cif():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    s_ref = np.array([[0.9, 0.7, 0.5, 0.3]])
    s_treat = np.array([[0.95, 0.85, 0.7, 0.55]])
    f_ref = {1: [np.array([[0.05, 0.15, 0.3, 0.5]])]}
    f_treat = {1: [np.array([[0.03, 0.10, 0.2, 0.35]])]}
    cifs = {
        1: [f_ref[1][0], f_treat[1][0]],
    }
    return _StubLearner(
        arms_survival=[s_ref, s_treat], arms_cif=cifs, times=times
    )


def test_predict_hte_returns_hteestimates_with_metadata():
    m = _stub_binary()
    X = np.zeros((2, 3))
    out = m.predict_hte(X, estimand=SurvivalDiff(times=m.times_))
    assert isinstance(out, HTEEstimates)
    assert out.contrast == "survival_diff"
    assert out.treatment == 1 and out.reference == 0
    assert out.cause is None
    assert out.values.shape == (2, 4)


def test_predict_hte_survival_diff_arithmetic():
    m = _stub_binary()
    X = np.zeros((2, 3))
    out = m.predict_hte(X, estimand=SurvivalDiff(times=m.times_))
    expected = m._arms_survival[1] - m._arms_survival[0]
    np.testing.assert_allclose(out.values, expected)


def test_predict_hte_rmst_diff_collapses_time_axis():
    m = _stub_binary()
    X = np.zeros((2, 3))
    out = m.predict_hte(X, estimand=RMSTDiff(times=m.times_, horizon=3.0))
    assert out.values.shape == (2,)
    assert out.is_time_collapsed  # rmst_diff collapses the time axis


def test_predict_hte_accepts_string_form():
    m = _stub_binary()
    X = np.zeros((2, 3))
    out = m.predict_hte(X, estimand="survival_diff", times=m.times_)
    expected = m._arms_survival[1] - m._arms_survival[0]
    np.testing.assert_allclose(out.values, expected)


def test_predict_hte_accepts_class_form():
    m = _stub_binary()
    X = np.zeros((2, 3))
    out = m.predict_hte(X, estimand=SurvivalRatio, times=m.times_)
    assert out.contrast == "survival_ratio"


def test_predict_hte_rejects_extra_kwargs_with_instance():
    m = _stub_binary()
    X = np.zeros((2, 3))
    with pytest.raises(TypeError, match="no other estimand-related kwargs"):
        m.predict_hte(X, estimand=SurvivalDiff(times=m.times_), times=m.times_)


def test_predict_hte_string_missing_required_param_raises():
    """estimand='rmst_diff' without horizon → dataclass enforces it."""
    m = _stub_binary()
    X = np.zeros((2, 3))
    with pytest.raises(TypeError, match="horizon"):
        m.predict_hte(X, estimand="rmst_diff", times=m.times_)


def test_predict_hte_string_unknown_estimand_raises():
    m = _stub_binary()
    X = np.zeros((2, 3))
    with pytest.raises(ValueError, match="estimand must be one of"):
        m.predict_hte(X, estimand="nonsense", times=m.times_)


def test_predict_hte_estimand_is_required():
    m = _stub_binary()
    X = np.zeros((2, 3))
    with pytest.raises(TypeError, match="estimand"):
        m.predict_hte(X)


def test_predict_hte_cif_diff_requires_cause_via_dataclass():
    m = _stub_with_cif()
    X = np.zeros((1, 3))
    with pytest.raises(TypeError, match="cause"):
        # Missing cause kwarg — caught by the CIFDiff dataclass.
        m.predict_hte(X, estimand="cif_diff", times=m.times_)


def test_predict_hte_cif_path_uses_cif_arrays():
    m = _stub_with_cif()
    X = np.zeros((1, 3))
    out = m.predict_hte(X, estimand=CIFDiff(times=m.times_, cause=1))
    expected = m._arms_cif[1][1] - m._arms_cif[1][0]
    np.testing.assert_allclose(out.values, expected)
    assert out.cause == 1


def test_predict_hte_rejects_treatment_index_out_of_range():
    m = _stub_binary()
    X = np.zeros((2, 3))
    with pytest.raises(ValueError, match="out of"):
        m.predict_hte(X, estimand=SurvivalDiff(times=m.times_, treatment=5))


def test_predict_ate_is_mean_of_predict_hte():
    m = _stub_binary()
    X = np.zeros((2, 3))
    est = SurvivalDiff(times=m.times_)
    ate = m.predict_ate(X, estimand=est)
    cate = m.predict_hte(X, estimand=est)
    np.testing.assert_allclose(ate, cate.values.mean(axis=0))


def test_predict_ate_without_X_uses_stored_X():
    m = _stub_binary()
    ate = m.predict_ate(estimand=SurvivalDiff(times=m.times_))
    assert ate.shape == (4,)  # one ATE per time point


def test_nuisances_dataclass_holds_per_arm_outcome_models():
    nu = Nuisances(
        outcome={0: object(), 1: object(), 2: object()},
        propensity=object(),
    )
    assert 2 in nu.outcome


def test_hte_estimates_carries_full_estimand():
    """The result object stores the spec that produced it — convenient
    for downstream tooling (plotting, metrics, serialisation)."""
    m = _stub_binary()
    est = SurvivalDiff(times=m.times_, treatment=1, reference=0)
    out = m.predict_hte(np.zeros((2, 3)), estimand=est)
    assert out.estimand is est
    # Convenience accessors mirror the estimand:
    assert out.contrast == est.contrast
    assert out.treatment == est.treatment
    assert out.reference == est.reference
    assert out.cause == getattr(est, "cause", None)


def test_hte_estimates_is_time_collapsed_for_rmst():
    m = _stub_binary()
    out_pointwise = m.predict_hte(
        np.zeros((2, 3)), estimand=SurvivalDiff(times=m.times_)
    )
    out_rmst = m.predict_hte(
        np.zeros((2, 3)), estimand=RMSTDiff(times=m.times_, horizon=3.0)
    )
    assert not out_pointwise.is_time_collapsed
    assert out_rmst.is_time_collapsed
