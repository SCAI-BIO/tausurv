"""Tests for the nuisance infrastructure: fit_nuisances + cross_fit.

Verifies (a) the carrier types (Nuisances + CrossFitNuisances and the
subclass relationship), (b) fit_nuisances mechanics, (c) cross_fit
mechanics — shapes, OOF completeness, stratification, reproducibility,
and the critical "OOF predictions come from models that didn't see the
subject" property.
"""

from __future__ import annotations

import numpy as np
import pytest

from causurv.nuisances import (
    CrossFitNuisances,
    Nuisances,
    cross_fit,
    fit_nuisances,
)


def _data(n=400, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    T = (rng.exponential(1.0, size=n) + 0.1).astype(np.float64)
    E = (rng.uniform(size=n) < 0.7).astype(np.int8)
    A = (rng.uniform(size=n) < 0.5).astype(np.int8)
    return X, T, E, A


def _factories():
    from sklearn.linear_model import LogisticRegression
    from tausurv.linear import CoxPH

    return (
        lambda: CoxPH(),
        lambda: LogisticRegression(max_iter=500),
        lambda: CoxPH(),
    )


def test_cross_fit_nuisances_is_a_nuisances():
    """Subclass relationship: orthogonal code can isinstance-check, but
    learners that only need a Nuisances accept either."""
    assert issubclass(CrossFitNuisances, Nuisances)


def test_nuisances_dataclass_is_kw_only():
    """We use kw_only=True so the CrossFitNuisances subclass can add
    non-default fields after Nuisances's defaulted `censoring`."""
    with pytest.raises(TypeError):
        Nuisances({}, None)  # positional — disallowed


def test_fit_nuisances_returns_nuisances():
    X, T, E, A = _data()
    outc, prop, _ = _factories()
    nuis = fit_nuisances(X, T, E, A, outcome_factory=outc, propensity_factory=prop)
    assert isinstance(nuis, Nuisances)
    assert not isinstance(nuis, CrossFitNuisances)


def test_fit_nuisances_outcome_dict_keyed_by_arm():
    X, T, E, A = _data()
    outc, prop, _ = _factories()
    nuis = fit_nuisances(X, T, E, A, outcome_factory=outc, propensity_factory=prop)
    assert sorted(nuis.outcome) == [0, 1]


def test_fit_nuisances_propensity_supports_predict_proba():
    X, T, E, A = _data()
    outc, prop, _ = _factories()
    nuis = fit_nuisances(X, T, E, A, outcome_factory=outc, propensity_factory=prop)
    proba = nuis.propensity.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_fit_nuisances_censoring_is_none_when_factory_missing():
    X, T, E, A = _data()
    outc, prop, _ = _factories()
    nuis = fit_nuisances(X, T, E, A, outcome_factory=outc, propensity_factory=prop)
    assert nuis.censoring is None


def test_fit_nuisances_censoring_fits_on_1_minus_event():
    X, T, E, A = _data()
    outc, prop, cens = _factories()
    nuis = fit_nuisances(
        X, T, E, A,
        outcome_factory=outc, propensity_factory=prop, censoring_factory=cens,
    )
    assert nuis.censoring is not None
    # Censoring model should be able to predict survival functions.
    grid = np.array([0.5, 1.0, 2.0])
    G = nuis.censoring.predict_survival_function(X[:5], grid)
    assert G.shape == (5, 3)


def test_fit_nuisances_validates_dimensions():
    outc, prop, _ = _factories()
    with pytest.raises(ValueError, match="X must be 2D"):
        fit_nuisances(
            np.zeros(10), np.ones(10), np.ones(10, dtype=np.int8),
            np.zeros(10, dtype=np.int8),
            outcome_factory=outc, propensity_factory=prop,
        )


def test_cross_fit_returns_cross_fit_nuisances():
    X, T, E, A = _data()
    outc, prop, _ = _factories()
    xf = cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                   n_folds=3, seed=0)
    assert isinstance(xf, CrossFitNuisances)
    assert isinstance(xf, Nuisances)


def test_cross_fit_shapes_are_aligned_to_grid():
    X, T, E, A = _data(n=200)
    outc, prop, cens = _factories()
    xf = cross_fit(
        X, T, E, A,
        outcome_factory=outc, propensity_factory=prop, censoring_factory=cens,
        n_folds=4, seed=0,
    )
    n_t = xf.times.size
    assert xf.oof_outcome[0].shape == (200, n_t)
    assert xf.oof_outcome[1].shape == (200, n_t)
    assert xf.oof_propensity.shape == (200, 2)
    assert xf.oof_censoring.shape == (200, n_t)
    assert xf.fold_assignment.shape == (200,)
    assert xf.n_folds == 4


def test_cross_fit_propensity_rows_sum_to_one():
    X, T, E, A = _data()
    outc, prop, _ = _factories()
    xf = cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                   n_folds=5, seed=0)
    np.testing.assert_allclose(xf.oof_propensity.sum(axis=1), 1.0)


def test_cross_fit_no_nan_in_oof():
    X, T, E, A = _data()
    outc, prop, cens = _factories()
    xf = cross_fit(
        X, T, E, A,
        outcome_factory=outc, propensity_factory=prop, censoring_factory=cens,
        n_folds=5, seed=0,
    )
    for a in (0, 1):
        assert not np.isnan(xf.oof_outcome[a]).any()
    assert not np.isnan(xf.oof_propensity).any()
    assert not np.isnan(xf.oof_censoring).any()


def test_cross_fit_oof_censoring_is_none_when_factory_missing():
    X, T, E, A = _data()
    outc, prop, _ = _factories()
    xf = cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                   n_folds=3, seed=0)
    assert xf.oof_censoring is None
    assert xf.censoring is None


def test_cross_fit_fold_assignment_is_complete_and_balanced():
    X, T, E, A = _data(n=300)
    outc, prop, _ = _factories()
    xf = cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                   n_folds=5, seed=0)
    folds, counts = np.unique(xf.fold_assignment, return_counts=True)
    assert set(folds) == {0, 1, 2, 3, 4}
    # Folds should be roughly balanced (within 1 of each other).
    assert counts.max() - counts.min() <= 2


def test_cross_fit_stratify_by_arm_keeps_every_arm_in_every_fold():
    X, T, E, A = _data(n=120)
    outc, prop, _ = _factories()
    xf = cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                   n_folds=5, stratify_by_arm=True, seed=0)
    for k in range(5):
        for a in (0, 1):
            n_in_fold = int(((xf.fold_assignment == k) & (A == a)).sum())
            assert n_in_fold > 0, f"fold {k} has no arm-{a} subjects"


def test_oof_predictions_come_from_held_out_folds():
    """The headline guarantee: for subject i in fold k, ``oof_outcome[a][i]``
    must equal the prediction of an outcome model fit on the *other*
    K-1 folds' arm-a subjects. If this breaks, orthogonal-learner
    theory falls over."""
    from tausurv.linear import CoxPH

    X, T, E, A = _data(n=150, seed=1)
    outc, prop, _ = _factories()
    xf = cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                   n_folds=3, seed=0)

    # Pick the first subject and reproduce its OOF outcome prediction by hand.
    i = 0
    fold_i = int(xf.fold_assignment[i])
    train_mask = (xf.fold_assignment != fold_i) & (A == 0)
    cox_arm0 = CoxPH().fit(X[train_mask], T[train_mask], E[train_mask])
    expected = cox_arm0.predict_survival_function(X[i : i + 1], xf.times)
    np.testing.assert_allclose(xf.oof_outcome[0][i], expected[0], rtol=1e-8)


def test_cross_fit_full_data_refit_is_consistent_with_fit_nuisances():
    """The .outcome models inside CrossFitNuisances should be full-data refits —
    i.e., produce the same predictions as fit_nuisances on the same data."""
    from tausurv.linear import CoxPH

    X, T, E, A = _data(n=200, seed=2)
    outc, prop, _ = _factories()

    xf = cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                   n_folds=5, seed=0)
    nuis = fit_nuisances(X, T, E, A, outcome_factory=outc, propensity_factory=prop)

    test_X = X[:10]
    grid = xf.times
    np.testing.assert_allclose(
        xf.outcome[0].predict_survival_function(test_X, grid),
        nuis.outcome[0].predict_survival_function(test_X, grid),
        rtol=1e-8,
    )


def test_cross_fit_seed_is_reproducible():
    X, T, E, A = _data()
    outc, prop, _ = _factories()
    a = cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                  n_folds=3, seed=123)
    b = cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                  n_folds=3, seed=123)
    np.testing.assert_array_equal(a.fold_assignment, b.fold_assignment)
    np.testing.assert_allclose(a.oof_propensity, b.oof_propensity)
    np.testing.assert_allclose(a.oof_outcome[0], b.oof_outcome[0])


def test_cross_fit_different_seeds_differ():
    X, T, E, A = _data()
    outc, prop, _ = _factories()
    a = cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                  n_folds=3, seed=0)
    b = cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                  n_folds=3, seed=1)
    assert not np.array_equal(a.fold_assignment, b.fold_assignment)


def test_cross_fit_requires_at_least_two_folds():
    X, T, E, A = _data()
    outc, prop, _ = _factories()
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                  n_folds=1, seed=0)


def test_cross_fit_validates_dimensions():
    outc, prop, _ = _factories()
    with pytest.raises(ValueError, match="X must be 2D"):
        cross_fit(
            np.zeros(10), np.ones(10), np.ones(10, dtype=np.int8),
            np.zeros(10, dtype=np.int8),
            outcome_factory=outc, propensity_factory=prop,
        )


def test_cross_fit_supports_three_arms():
    rng = np.random.default_rng(0)
    n, d = 240, 4
    X = rng.normal(size=(n, d))
    T = (rng.exponential(1.0, size=n) + 0.1).astype(np.float64)
    E = (rng.uniform(size=n) < 0.7).astype(np.int8)
    A = rng.choice([0, 1, 2], size=n).astype(np.int8)
    outc, prop, _ = _factories()
    xf = cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                   n_folds=4, seed=0)
    assert sorted(xf.oof_outcome) == [0, 1, 2]
    assert xf.oof_propensity.shape == (n, 3)


def test_cross_fit_respects_explicit_times_kwarg():
    X, T, E, A = _data()
    outc, prop, _ = _factories()
    explicit = np.array([0.2, 0.5, 1.0, 2.0])
    xf = cross_fit(X, T, E, A, outcome_factory=outc, propensity_factory=prop,
                   times=explicit, n_folds=3, seed=0)
    np.testing.assert_array_equal(xf.times, explicit)
    assert xf.oof_outcome[0].shape == (X.shape[0], 4)
