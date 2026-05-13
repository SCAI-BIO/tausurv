from __future__ import annotations

import numpy as np
import pytest

from causurv.estimands import SurvivalDiff
from causurv.learners import CausalSurvivalForest
from tausurv.core import fit_gradient_forest


@pytest.fixture
def csf_factories():
    from sklearn.linear_model import LogisticRegression
    from tausurv.linear import CoxPH

    return (
        lambda: CoxPH(),
        lambda: LogisticRegression(max_iter=500),
        lambda: CoxPH(),
    )


def _hte_data(n=400, p=4, hazard_ratio=0.5, seed=0):
    """Treatment halves the hazard regardless of X (constant HTE > 0)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    A = (rng.uniform(size=n) < 0.5).astype(np.int8)
    rate = 1.0 * (1 - (1 - hazard_ratio) * A)
    T = rng.exponential(1.0 / rate)
    C = rng.exponential(5.0, size=n)
    Y = np.minimum(T, C)
    E = (T <= C).astype(np.int8)
    return X, Y, E, A


def _heterogeneous_data(n=600, p=4, seed=0):
    """Treatment effect varies with X[:, 0]: large for X[0]>0, none otherwise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    A = (rng.uniform(size=n) < 0.5).astype(np.int8)
    benefit = (X[:, 0] > 0).astype(np.float64)  # only half the population benefits
    rate = 1.0 * (1 - 0.6 * benefit * A)
    T = rng.exponential(1.0 / rate)
    C = rng.exponential(5.0, size=n)
    Y = np.minimum(T, C)
    E = (T <= C).astype(np.int8)
    return X, Y, E, A


def test_gradient_forest_recovers_signal():
    """A gradient forest fit on (X, pseudo = X[:,0]) should predict
    ≈ X[:,0] for any test point. Uses no honesty / no subsample so
    every tree sees the full signal."""
    n, p = 500, 3
    rng = np.random.default_rng(0)
    X = np.asfortranarray(rng.normal(size=(n, p)).astype(np.float64))
    pseudo = X[:, 0].copy().astype(np.float64)
    forest = fit_gradient_forest(
        X, pseudo,
        n_trees=100, min_samples_leaf=5,
        max_features="all", bootstrap=False, subsample_fraction=1.0,
        honesty=False, seed=42,
    )
    W = forest.forest_weights(X)
    preds = W @ pseudo
    assert np.corrcoef(preds, pseudo)[0, 1] > 0.99


def test_gradient_forest_weights_sum_to_one():
    """In no-subsample mode, every tree's target leaf contains the full
    estimation set, so weights sum to exactly 1.0 per query."""
    n, p = 200, 3
    rng = np.random.default_rng(0)
    X = np.asfortranarray(rng.normal(size=(n, p)).astype(np.float64))
    pseudo = X[:, 0].astype(np.float64)
    forest = fit_gradient_forest(
        X, pseudo,
        n_trees=10, min_samples_leaf=5,
        max_features="all", bootstrap=False, subsample_fraction=1.0,
        honesty=False, seed=0,
    )
    W = forest.forest_weights(X[:5])
    np.testing.assert_allclose(W.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12)


def test_causal_survival_forest_fits_and_predicts(csf_factories):
    outc, prop, cens = csf_factories
    X, T, E, A = _hte_data(n=400)
    csf = CausalSurvivalForest(
        outcome_factory=outc,
        propensity_factory=prop,
        censoring_factory=cens,
        estimand=SurvivalDiff(times=[1, 3, 5], treatment=1, reference=0),
        n_trees=50,
        n_folds=3,
        seed=0,
    ).fit(X, T, E, A)

    hte = csf.predict_hte(X[:10])
    assert hte.values.shape == (10, 3)
    assert np.all(np.isfinite(hte.values))


def test_causal_survival_forest_recovers_positive_ate(csf_factories):
    """Hazard halved under treatment → positive survival-difference ATE
    at every target time, at every reasonable signal strength."""
    outc, prop, cens = csf_factories
    X, T, E, A = _hte_data(n=600, hazard_ratio=0.5)
    csf = CausalSurvivalForest(
        outcome_factory=outc,
        propensity_factory=prop,
        censoring_factory=cens,
        estimand=SurvivalDiff(times=[1, 3, 5], treatment=1, reference=0),
        n_trees=100,
        n_folds=3,
        seed=0,
    ).fit(X, T, E, A)
    ate = csf.predict_ate()
    assert (ate > 0).all(), f"expected positive ATE at every horizon, got {ate}"


def test_causal_survival_forest_separates_heterogeneous_subgroups(csf_factories):
    """When treatment effect is concentrated in X[:, 0] > 0, the HTE
    estimate at high-X[0] points should exceed that at low-X[0] points."""
    outc, prop, cens = csf_factories
    X, T, E, A = _heterogeneous_data(n=800)
    csf = CausalSurvivalForest(
        outcome_factory=outc,
        propensity_factory=prop,
        censoring_factory=cens,
        estimand=SurvivalDiff(times=[1, 3], treatment=1, reference=0),
        n_trees=200,
        max_features="all",
        n_folds=5,
        seed=0,
    ).fit(X, T, E, A)
    hte = csf.predict_hte(X).values  # (n, 2)
    high = X[:, 0] > 0.5
    low = X[:, 0] < -0.5
    # At each horizon: mean HTE in the high-benefit subgroup > low-benefit.
    for j in range(2):
        assert hte[high, j].mean() > hte[low, j].mean(), (
            f"horizon {j}: high mean {hte[high, j].mean():.3f} "
            f"<= low mean {hte[low, j].mean():.3f}"
        )


def test_causal_survival_forest_deterministic_under_seed(csf_factories):
    outc, prop, cens = csf_factories
    X, T, E, A = _hte_data(n=200)
    common = dict(
        outcome_factory=outc, propensity_factory=prop, censoring_factory=cens,
        estimand=SurvivalDiff(times=[1, 3], treatment=1, reference=0),
        n_trees=20, n_folds=3, seed=42,
    )
    a = CausalSurvivalForest(**common).fit(X, T, E, A)
    b = CausalSurvivalForest(**common).fit(X, T, E, A)
    np.testing.assert_array_equal(
        a.predict_hte(X[:30]).values,
        b.predict_hte(X[:30]).values,
    )


def test_predict_potential_outcomes_raises(csf_factories):
    outc, prop, cens = csf_factories
    X, T, E, A = _hte_data(n=200)
    csf = CausalSurvivalForest(
        outcome_factory=outc,
        propensity_factory=prop,
        censoring_factory=cens,
        estimand=SurvivalDiff(times=[1, 3], treatment=1, reference=0),
        n_trees=20,
        n_folds=3,
        seed=0,
    ).fit(X, T, E, A)
    with pytest.raises(NotImplementedError):
        csf.predict_potential_outcomes(X[:5])
