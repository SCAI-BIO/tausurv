"""Tests for the CauseSpecificPredictor composite competing-risks model."""

from __future__ import annotations

import numpy as np
import pytest

from tausurv import simulations
from tausurv.linear import CoxPH
from tausurv.predictor import CauseSpecificPredictor


def _cr_data(n=500, n_causes=2, seed=0):
    return simulations.competing_risk(
        n=n, n_features=5, n_causes=n_causes, seed=seed
    )


def test_n_causes_must_be_positive():
    with pytest.raises(ValueError, match="n_causes"):
        CauseSpecificPredictor(CoxPH, n_causes=0)


def test_event_indicator_out_of_range():
    X, T, E = _cr_data(n=50, n_causes=2)
    E = E.copy()
    E[0] = 3  # only causes 1, 2 expected
    model = CauseSpecificPredictor(CoxPH, n_causes=2)
    with pytest.raises(ValueError, match=r"event_indicator must be in"):
        model.fit(X, T, E)


def test_predict_cif_shape():
    X, T, E = _cr_data(n=300, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2).fit(X, T, E)
    grid = np.linspace(0.1, 3.0, 10)
    F = model.predict_cif(X[:20], grid)
    assert F.shape == (20, 2, 10)


def test_predict_cif_single_cause_shape():
    X, T, E = _cr_data(n=300, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2).fit(X, T, E)
    grid = np.linspace(0.1, 3.0, 10)
    F = model.predict_cif(X[:20], grid, cause=1)
    assert F.shape == (20, 10)


def test_cif_bounds():
    X, T, E = _cr_data(n=300, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2).fit(X, T, E)
    grid = np.linspace(0.1, 3.0, 10)
    F = model.predict_cif(X[:20], grid)
    assert ((F >= 0.0) & (F <= 1.0)).all()


def test_cif_non_decreasing_in_t():
    X, T, E = _cr_data(n=300, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2).fit(X, T, E)
    grid = np.linspace(0.05, 3.0, 20)
    F = model.predict_cif(X[:30], grid)
    assert (np.diff(F, axis=-1) >= -1e-12).all()


def test_survival_non_increasing_in_t():
    X, T, E = _cr_data(n=300, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2).fit(X, T, E)
    grid = np.linspace(0.05, 3.0, 20)
    S = model.predict_survival_function(X[:30], grid)
    assert (np.diff(S, axis=1) <= 1e-12).all()


def test_sum_of_cifs_equals_one_minus_survival():
    r"""$\sum_k F_k(t \mid x) = 1 - S(t \mid x)$ exactly by construction.

    With $S(t_i) = \prod_{j \le i}(1 - \Delta\Lambda(t_j))$, each increment
    satisfies $S(t_i^-)\,\Delta\Lambda(t_i) = S(t_i^-) - S(t_i)$, so the
    per-cause CIFs sum to $1 - S$ exactly.
    """
    X, T, E = _cr_data(n=300, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2).fit(X, T, E)
    grid = model.times_
    F = model._cif(X[:30], grid)  # (n, K, T)
    S = model.predict_survival_function(X[:30], grid)  # (n, T)
    np.testing.assert_allclose(F.sum(axis=1), 1.0 - S, atol=1e-10)


def test_survival_approximates_product_of_cause_survivals():
    r"""$S(t|x) \approx \prod_k S_k(t|x)$.

    The discrete product $\prod_i (1 - \Delta\Lambda(t_i))$ approximates the
    product of continuous survivals $\prod_k \exp(-\Lambda_k)$.
    """
    X, T, E = _cr_data(n=300, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2).fit(X, T, E)
    grid = model.times_
    S1 = model.models_[0].predict_survival_function(X[:30], grid)
    S2 = model.models_[1].predict_survival_function(X[:30], grid)
    S_direct = S1 * S2
    S_from_mixin = model.predict_survival_function(X[:30], grid)
    np.testing.assert_allclose(S_from_mixin, S_direct, atol=0.15)


def test_single_cause_approximates_wrapped_model():
    r"""With one cause, $F \approx 1 - S_{\text{wrapped}}$ (discrete product)."""
    X, T, E = _cr_data(n=300, n_causes=1)
    model = CauseSpecificPredictor(CoxPH, n_causes=1).fit(X, T, E)
    grid = model.times_
    F = model.predict_cif(X[:20], grid, cause=1)
    S_wrapped = model.models_[0].predict_survival_function(X[:20], grid)
    np.testing.assert_allclose(F, 1.0 - S_wrapped, atol=0.05)


def test_times_set_after_fit():
    X, T, E = _cr_data(n=200, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2)
    assert not hasattr(model, "times_")
    model.fit(X, T, E)
    assert hasattr(model, "times_")
    assert model.times_.ndim == 1
    assert np.all(np.diff(model.times_) > 0)  # sorted, unique
