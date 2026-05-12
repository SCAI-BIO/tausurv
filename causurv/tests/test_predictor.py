"""Tests for the HTEPredictor protocol via a stub subclass.

Verifies that :meth:`predict_hte` and :meth:`predict_ate` correctly
dispatch contrasts, validate cause/contrast combinations, slice arms
by ``treatment``/``reference``, and resolve default times.
"""

from __future__ import annotations

import numpy as np
import pytest

from causurv.nuisances import Nuisances
from causurv.predictor import HTEEstimates, HTEPredictor


class _StubLearner(HTEPredictor):
    """Returns fixed potential outcomes; no actual fitting."""

    def __init__(self, *, arms_survival, arms_cif=None, times):
        # arms_survival: list of (n, T) arrays, one per arm.
        # arms_cif: dict {cause: list of (n, T) arrays} for CR.
        self._arms_survival = arms_survival
        self._arms_cif = arms_cif or {}
        self.times_ = np.asarray(times, dtype=np.float64)
        self._fit_X = np.zeros((arms_survival[0].shape[0], 2))

    def fit(self, X, T, E, A):
        return self

    def predict_potential_outcomes(self, X, times=None, *, cause=None):
        # Ignores X for this stub (it's about the protocol, not fitting).
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
    X = np.zeros((2, 3))  # ignored by stub
    out = m.predict_hte(X)
    assert isinstance(out, HTEEstimates)
    assert out.contrast == "survival_diff"
    assert out.treatment == 1 and out.reference == 0
    assert out.cause is None
    assert out.values.shape == (2, 4)


def test_predict_hte_survival_diff_arithmetic():
    m = _stub_binary()
    X = np.zeros((2, 3))
    out = m.predict_hte(X, contrast="survival_diff")
    expected = m._arms_survival[1] - m._arms_survival[0]
    np.testing.assert_allclose(out.values, expected)


def test_predict_hte_rmst_diff_collapses_time_axis():
    m = _stub_binary()
    X = np.zeros((2, 3))
    out = m.predict_hte(X, contrast="rmst_diff", horizon=3.0)
    assert out.values.shape == (2,)
    assert out.times is None  # time-collapsed contrast


def test_predict_hte_cif_diff_requires_cause():
    m = _stub_with_cif()
    X = np.zeros((1, 3))
    with pytest.raises(ValueError, match="requires `cause`"):
        m.predict_hte(X, contrast="cif_diff")


def test_predict_hte_survival_diff_rejects_cause():
    m = _stub_with_cif()
    X = np.zeros((1, 3))
    with pytest.raises(ValueError, match="cannot take `cause`"):
        m.predict_hte(X, contrast="survival_diff", cause=1)


def test_predict_hte_cif_path_uses_cif_arrays():
    m = _stub_with_cif()
    X = np.zeros((1, 3))
    out = m.predict_hte(X, contrast="cif_diff", cause=1)
    expected = m._arms_cif[1][1] - m._arms_cif[1][0]
    np.testing.assert_allclose(out.values, expected)
    assert out.cause == 1


def test_predict_hte_rejects_treatment_index_out_of_range():
    m = _stub_binary()
    X = np.zeros((2, 3))
    with pytest.raises(ValueError, match="out of"):
        m.predict_hte(X, treatment=5)


def test_predict_ate_is_mean_of_predict_hte():
    m = _stub_binary()
    X = np.zeros((2, 3))
    ate = m.predict_ate(X)
    cate = m.predict_hte(X)
    np.testing.assert_allclose(ate, cate.values.mean(axis=0))


def test_predict_ate_without_X_uses_stored_X():
    m = _stub_binary()
    ate = m.predict_ate()
    assert ate.shape == (4,)  # one ATE per time point


def test_resolve_times_falls_back_to_attribute():
    m = _stub_binary()
    X = np.zeros((2, 3))
    out_with_default = m.predict_hte(X)
    out_explicit = m.predict_hte(X, times=m.times_)
    np.testing.assert_array_equal(out_with_default.values, out_explicit.values)


def test_resolve_times_raises_when_unset():
    m = _stub_binary()
    del m.times_
    with pytest.raises(RuntimeError, match="times_"):
        m.predict_hte(np.zeros((1, 3)))


def test_nuisances_dataclass_holds_per_arm_outcome_models():
    """The Nuisances dataclass should accept arbitrary arm labels (not
    just 0 and 1) — so multi-arm extension is non-breaking."""
    nu = Nuisances(
        outcome={0: object(), 1: object(), 2: object()},
        propensity=object(),
    )
    assert 2 in nu.outcome


