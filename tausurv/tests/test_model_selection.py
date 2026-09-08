"""Tests for tausurv.model_selection."""

from __future__ import annotations

import numpy as np
import pytest
from tausurv.model_selection import train_test_split


def _data(n=500, n_features=5, event_rate=0.4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    T = rng.exponential(1.0, size=n)
    E = (rng.uniform(size=n) < event_rate).astype(int)
    return X, T, E


def test_shapes_partition_correctly():
    X, T, E = _data(n=500)
    X_tr, X_te, T_tr, T_te, E_tr, E_te = train_test_split(
        X, T, E, test_size=0.2, seed=0
    )
    assert X_tr.shape[0] + X_te.shape[0] == 500
    assert X_tr.shape[1] == X.shape[1] and X_te.shape[1] == X.shape[1]
    assert T_tr.shape == (X_tr.shape[0],) and T_te.shape == (X_te.shape[0],)
    assert E_tr.shape == (X_tr.shape[0],) and E_te.shape == (X_te.shape[0],)
    assert X_te.shape[0] == round(500 * 0.2)


def test_no_index_appears_in_both_splits():
    """Every row goes to exactly one of train/test."""
    X, T, E = _data(n=100)
    # Tag rows with a unique signature in T so we can identify them post-split.
    T = np.arange(100, dtype=float)
    X_tr, X_te, T_tr, T_te, _, _ = train_test_split(X, T, E, test_size=0.3, seed=0)
    union = np.sort(np.concatenate([T_tr, T_te]))
    np.testing.assert_array_equal(union, np.arange(100, dtype=float))


def test_stratified_split_preserves_event_proportions():
    X, T, E = _data(n=2000, event_rate=0.1)
    overall_rate = E.mean()
    _, _, _, _, E_tr, E_te = train_test_split(
        X, T, E, test_size=0.3, stratify=True, seed=0
    )
    # Within a few-sample slop, both splits match the overall event rate.
    assert abs(E_tr.mean() - overall_rate) < 0.01
    assert abs(E_te.mean() - overall_rate) < 0.02


def test_stratified_split_handles_competing_risks():
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 3))
    T = rng.exponential(1.0, size=n)
    # Three classes with different proportions: 60% censored, 25% cause 1, 15% cause 2.
    E = rng.choice([0, 1, 2], size=n, p=[0.6, 0.25, 0.15])
    _, _, _, _, E_tr, E_te = train_test_split(
        X, T, E, test_size=0.25, stratify=True, seed=0
    )
    for cls in (0, 1, 2):
        p_overall = (E == cls).mean()
        p_train = (E_tr == cls).mean()
        p_test = (E_te == cls).mean()
        assert abs(p_train - p_overall) < 0.02, f"class {cls}: train deviates"
        assert abs(p_test - p_overall) < 0.03, f"class {cls}: test deviates"


def test_non_stratified_can_skew_event_proportions():
    """Without stratification a rare-event dataset can produce a test
    set whose event proportion drifts from the overall rate by far more
    than the stratified version's deviation."""
    rng = np.random.default_rng(0)
    n = 50
    X = rng.normal(size=(n, 3))
    T = rng.exponential(1.0, size=n)
    E = (rng.uniform(size=n) < 0.05).astype(int)  # 5% event rate, very rare

    # Across many seeds, the non-stratified max deviation is much larger
    # than the stratified one.
    devs_strat = []
    devs_random = []
    for s in range(100):
        _, _, _, _, _, E_te_s = train_test_split(
            X, T, E, test_size=0.3, stratify=True, seed=s
        )
        _, _, _, _, _, E_te_r = train_test_split(
            X, T, E, test_size=0.3, stratify=False, seed=s
        )
        devs_strat.append(abs(E_te_s.mean() - E.mean()))
        devs_random.append(abs(E_te_r.mean() - E.mean()))
    assert max(devs_strat) <= max(devs_random)


def test_seed_reproducibility():
    X, T, E = _data(n=200)
    out1 = train_test_split(X, T, E, test_size=0.3, seed=42)
    out2 = train_test_split(X, T, E, test_size=0.3, seed=42)
    for a, b in zip(out1, out2, strict=True):
        np.testing.assert_array_equal(a, b)


def test_different_seeds_produce_different_splits():
    X, T, E = _data(n=200)
    _, _, T_te_1, _, _, _ = train_test_split(X, T, E, test_size=0.3, seed=1)
    _, _, T_te_2, _, _, _ = train_test_split(X, T, E, test_size=0.3, seed=2)
    # Splits should differ — sort by some component and compare; very low
    # probability of identical permutation by chance.
    assert not np.array_equal(np.sort(T_te_1), np.sort(T_te_2))


def test_test_size_must_be_in_open_unit_interval():
    X, T, E = _data(n=50)
    for bad in (-0.1, 0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="test_size"):
            train_test_split(X, T, E, test_size=bad)


def test_mismatched_shapes_raise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    T = rng.exponential(1.0, size=99)  # off by one
    E = (rng.uniform(size=100) < 0.5).astype(int)
    with pytest.raises(ValueError, match="first axis"):
        train_test_split(X, T, E)


def test_works_with_torch_tensors_via_asarray():
    """X can be any array-like — numpy converts."""
    torch = pytest.importorskip("torch")
    X = torch.randn(100, 3)
    T = torch.rand(100) + 0.1
    E = torch.randint(0, 2, (100,))
    X_tr, X_te, T_tr, T_te, E_tr, E_te = train_test_split(
        X.numpy(), T.numpy(), E.numpy(), seed=0
    )
    assert X_tr.shape[0] + X_te.shape[0] == 100


from tausurv.linear import CoxPH  # noqa: E402
from tausurv.metrics import concordance  # noqa: E402
from tausurv.model_selection import (  # noqa: E402
    CVResult,
    Scorer,
    TuneResult,
    cross_validate,
    nested_cv,
    scoring,
    stratified_folds,
    tune,
)
from tausurv.trees import RandomSurvivalForest  # noqa: E402

from tausurv import simulations  # noqa: E402


def test_stratified_folds_partition_and_stratify():
    E = np.array([0] * 40 + [1] * 20 + [2] * 10)
    folds = stratified_folds(E, n_splits=5, seed=0)

    tests = np.concatenate([te for _, te in folds])
    np.testing.assert_array_equal(np.sort(tests), np.arange(70))
    for tr, te in folds:
        assert np.intersect1d(tr, te).size == 0
        assert (E[te] == 2).sum() == 2
        assert (E[te] == 1).sum() == 4


def test_stratified_folds_reject_single_split():
    with pytest.raises(ValueError, match="n_splits"):
        stratified_folds(np.array([0, 1, 1]), n_splits=1)


def test_cross_validate_returns_one_row_per_fold_and_scorer():
    X, T, E = simulations.single_risk(n=300, n_features=4, seed=0)
    times = np.linspace(0.2, 2.0, 5)

    scores = cross_validate(
        lambda: CoxPH(),
        X,
        T,
        E,
        scoring={
            "harrell": scoring.harrell(),
            "uno": scoring.uno(tau=2.0),
            "ibs": scoring.integrated_brier(times),
        },
        cv=4,
        seed=0,
    )

    assert scores.scores.columns == ["fold", "harrell", "uno", "ibs", "seconds"]
    assert scores.scores.height == 4
    assert scores.scores["harrell"].min() > 0.5
    assert 0.0 < scores.scores["ibs"].max() < 0.5
    assert len(scores.models) == 4 and len(scores.splits) == 4


def test_cross_validate_single_scorer_column_is_score():
    X, T, E = simulations.single_risk(n=200, n_features=3, seed=1)
    scores = cross_validate(lambda: CoxPH(), X, T, E, scoring=scoring.harrell(), cv=3)
    assert scores.scores.columns == ["fold", "harrell_c", "seconds"]


def test_cross_validate_custom_scorer_and_train():
    X, T, E = simulations.single_risk(n=200, n_features=3, seed=2)
    seen = []

    def train(model, X_tr, T_tr, E_tr, trial=None):
        seen.append(trial)
        model.fit(X_tr, T_tr, E_tr)

    def n_test(model, X_te, T_te, E_te, train_fold):
        assert train_fold.X.shape[0] + X_te.shape[0] == 200
        return float(X_te.shape[0])

    scores = cross_validate(lambda: CoxPH(), X, T, E, scoring=n_test, train=train, cv=4)
    assert seen == [None] * 4
    assert scores.scores["score"].sum() == 200


def test_cross_validate_accepts_explicit_index_pairs():
    X, T, E = simulations.single_risk(n=100, n_features=3, seed=3)
    idx = np.arange(100)
    splits = [(idx[:70], idx[70:]), (idx[30:], idx[:30])]
    scores = cross_validate(
        lambda: CoxPH(), X, T, E, scoring=scoring.harrell(), cv=splits
    )
    assert scores.scores.height == 2


def test_tune_samples_in_build_and_train():
    optuna = pytest.importorskip("optuna")
    X, T, E = simulations.single_risk(n=200, n_features=3, seed=4)

    def build(trial=None):
        return CoxPH(
            tol=trial.suggest_float("tol", 1e-8, 1e-4, log=True) if trial else 1e-6
        )

    def train(model, X_tr, T_tr, E_tr, trial=None):
        assert trial is not None
        trial.suggest_int("unused", 1, 3)
        model.fit(X_tr, T_tr, E_tr)

    best = tune(
        build, X, T, E, scoring=scoring.harrell(), train=train, cv=3, n_trials=4
    )

    assert set(best.params) == {"tol", "unused"}
    assert isinstance(best.study, optuna.Study)
    assert len(best.study.trials) == 4
    assert best.model.tol == best.params["tol"]
    assert best.model.predict(X).shape == (200,)


def test_tune_minimises_when_scorer_says_so():
    pytest.importorskip("optuna")
    X, T, E = simulations.single_risk(n=200, n_features=3, seed=5)
    times = np.linspace(0.2, 2.0, 5)
    best = tune(
        lambda trial=None: CoxPH(),
        X,
        T,
        E,
        scoring=scoring.integrated_brier(times),
        cv=3,
        n_trials=2,
        refit=False,
    )
    assert best.study.direction.name == "MINIMIZE"
    assert best.model is None


def test_nested_cv_reports_outer_scores_and_params():
    pytest.importorskip("optuna")
    X, T, E = simulations.single_risk(n=240, n_features=3, seed=6)

    def build(trial=None):
        return CoxPH(
            tol=trial.suggest_float("tol", 1e-8, 1e-4, log=True) if trial else 1e-6
        )

    result = nested_cv(
        build, X, T, E, scoring=scoring.harrell(), outer=3, inner=2, n_trials=3
    )

    assert result.scores.columns == ["fold", "harrell_c", "seconds"]
    assert (result.scores["seconds"] > 0).all()
    assert result.scores.height == 3
    assert len(result.params) == 3 and all("tol" in p for p in result.params)
    assert len(result.studies) == 3 and len(result.models) == 3
    assert all("tol" in t.columns for t in result.trials)


def test_tune_with_neural_model_smoke():
    pytest.importorskip("optuna")
    pytest.importorskip("torch")
    from tausurv.nn import DeepSurv

    X, T, E = simulations.single_risk(n=120, n_features=3, seed=7)

    def build(trial=None):
        hidden = trial.suggest_int("hidden_dim", 4, 8) if trial else 8
        return DeepSurv(in_features=3, hidden_dim=hidden, n_blocks=1)

    def train(model, X_tr, T_tr, E_tr, trial=None):
        lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True) if trial else 1e-3
        model.fit(X_tr, T_tr, E_tr, lr=lr, epochs=2)

    best = tune(
        build, X, T, E, scoring=scoring.harrell(), train=train, cv=2, n_trials=2
    )
    assert set(best.params) == {"hidden_dim", "lr"}
    assert best.model.predict(X).shape == (120,)


def test_cvresult_predicts_out_of_fold_and_refuses_other_sizes():
    X, T, E = simulations.single_risk(n=120, n_features=3, seed=8)
    cv = cross_validate(lambda: CoxPH(), X, T, E, scoring=scoring.harrell(), cv=3)

    risk = cv.predict(X)
    for model, (_, test_idx) in zip(cv.models, cv.splits, strict=True):
        np.testing.assert_allclose(risk[test_idx], model.predict(X[test_idx]))

    times = np.array([0.5, 1.0])
    S = cv.predict_survival_function(X, times)
    assert S.shape == (120, 2)
    np.testing.assert_allclose(cv.predict_cif(X, times, cause=1), 1 - S)

    with pytest.raises(ValueError, match="ensemble"):
        cv.predict(X[:10])


def test_cvresult_ensemble_averages_fold_models():
    X, T, E = simulations.single_risk(n=120, n_features=3, seed=9)
    cv = cross_validate(lambda: CoxPH(), X, T, E, scoring=scoring.harrell(), cv=3)
    X_new = X[:7]

    expected = np.mean([m.predict(X_new) for m in cv.models], axis=0)
    np.testing.assert_allclose(cv.ensemble.predict(X_new), expected)
    S = cv.ensemble.predict_survival_function(X_new, np.array([0.5, 1.0]))
    assert S.shape == (7, 2)


def test_cvresult_save_load_round_trip(tmp_path):
    X, T, E = simulations.single_risk(n=120, n_features=3, seed=10)
    cv = cross_validate(lambda: CoxPH(), X, T, E, scoring=scoring.harrell(), cv=3)
    cv.save(tmp_path / "cv")
    loaded = CVResult.load(tmp_path / "cv", model=CoxPH)

    assert loaded.scores.equals(cv.scores)
    for (a, b), (c, d) in zip(loaded.splits, cv.splits, strict=True):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
    np.testing.assert_allclose(loaded.predict(X), cv.predict(X))
    assert loaded.params == cv.params and loaded.trials is None

    with pytest.raises(ValueError, match="CoxPH"):
        CVResult.load(tmp_path / "cv", model=RandomSurvivalForest)


def test_cvresult_competing_risks_shapes():
    pytest.importorskip("torch")
    from tausurv.nn import DeepHit

    X, T, E = simulations.competing_risk(n=150, n_features=4, n_causes=2, seed=11)

    def train(model, X_tr, T_tr, E_tr, trial=None):
        model.fit(X_tr, T_tr, E_tr, epochs=2)

    cv = cross_validate(
        lambda: DeepHit(in_features=4, n_causes=2, n_bins=5, hidden_dim=8, n_blocks=1),
        X,
        T,
        E,
        train=train,
        scoring=scoring.harrell(),
        cv=3,
    )
    times = cv.times_
    assert cv.n_causes == 2
    assert cv.predict_cif(X, times).shape == (150, 2, len(times))
    assert cv.predict_cif(X, times, cause=2).shape == (150, len(times))
    assert cv.ensemble.predict_cif(X[:5], times, cause=1).shape == (5, len(times))


def test_tune_result_save_load_round_trip(tmp_path):
    pytest.importorskip("optuna")
    X, T, E = simulations.single_risk(n=150, n_features=3, seed=12)

    def build(trial=None):
        return CoxPH(
            tol=trial.suggest_float("tol", 1e-8, 1e-4, log=True) if trial else 1e-6
        )

    best = tune(build, X, T, E, scoring=scoring.harrell(), cv=2, n_trials=3)
    best.save(tmp_path / "best")
    loaded = TuneResult.load(tmp_path / "best", model=CoxPH)

    assert loaded.params == best.params and loaded.score == best.score
    assert loaded.trials.equals(best.trials) and loaded.study is None
    assert set(best.trials.columns) >= {"number", "value", "state", "tol"}
    np.testing.assert_allclose(loaded.model.predict(X), best.model.predict(X))


def test_tune_with_storage_resumes(tmp_path):
    pytest.importorskip("optuna")
    X, T, E = simulations.single_risk(n=150, n_features=3, seed=13)
    url = f"sqlite:///{tmp_path / 'study.db'}"

    def build(trial=None):
        return CoxPH(
            tol=trial.suggest_float("tol", 1e-8, 1e-4, log=True) if trial else 1e-6
        )

    first = tune(
        build,
        X,
        T,
        E,
        scoring=scoring.harrell(),
        cv=2,
        n_trials=2,
        storage=url,
        study_name="s",
        refit=False,
    )
    second = tune(
        build,
        X,
        T,
        E,
        scoring=scoring.harrell(),
        cv=2,
        n_trials=2,
        storage=url,
        study_name="s",
        refit=False,
    )
    assert first.trials.height == 2 and second.trials.height == 4


def test_custom_loss_wrapped_in_scorer_is_minimised():
    pytest.importorskip("optuna")
    X, T, E = simulations.single_risk(n=150, n_features=3, seed=14)

    def one_minus_c(model, X_te, T_te, E_te, train_fold):
        return 1.0 - concordance.harrell(T_te, E_te, model.predict(X_te))

    best = tune(
        lambda trial=None: CoxPH(),
        X,
        T,
        E,
        scoring=Scorer(one_minus_c, greater_is_better=False),
        cv=2,
        n_trials=2,
        refit=False,
    )
    assert best.study.direction.name == "MINIMIZE"


def test_out_of_fold_rows_never_held_out_are_nan():
    X, T, E = simulations.single_risk(n=60, n_features=3, seed=15)
    idx = np.arange(60)
    splits = [(idx[:40], idx[40:50]), (idx[10:], idx[:10])]
    cv = cross_validate(lambda: CoxPH(), X, T, E, scoring=scoring.harrell(), cv=splits)

    risk = cv.predict(X)
    held_out = np.r_[0:10, 40:50]
    assert np.isfinite(risk[held_out]).all()
    assert np.isnan(np.delete(risk, held_out)).all()


def test_evaluate_scalar_and_curve_scorers():
    X, T, E = simulations.single_risk(n=150, n_features=3, seed=16)
    cv = cross_validate(lambda: CoxPH(), X, T, E, scoring=scoring.harrell(), cv=3)
    times = np.linspace(0.2, 1.5, 6)

    c = cv.evaluate(scoring.harrell(), X, T, E)
    np.testing.assert_allclose(c, cv.scores["harrell_c"].to_numpy())

    aucs = cv.evaluate(scoring.auc_over_time(times), X, T, E)
    briers = cv.evaluate(scoring.brier_over_time(times), X, T, E)
    assert aucs.shape == (3, 6) and briers.shape == (3, 6)
    assert np.all((aucs >= 0) & (aucs <= 1)) and np.all((briers >= 0) & (briers <= 1))

    with pytest.raises(ValueError, match="own data"):
        cv.evaluate(scoring.harrell(), X[:10], T[:10], E[:10])


def test_progress_can_be_switched_off(capsys):
    X, T, E = simulations.single_risk(n=80, n_features=3, seed=17)
    cross_validate(
        lambda: CoxPH(), X, T, E, scoring=scoring.harrell(), cv=2, progress=False
    )
    assert "folds" not in capsys.readouterr().err


def test_nested_progress_reports_one_bar_of_fits(capsys):
    pytest.importorskip("optuna")
    X, T, E = simulations.single_risk(n=120, n_features=3, seed=18)
    nested_cv(
        lambda trial=None: CoxPH(),
        X,
        T,
        E,
        scoring=scoring.harrell(),
        outer=2,
        inner=2,
        n_trials=3,
    )
    err = capsys.readouterr().err
    assert err.count("fits:") >= 1 and "14/14" in err and "outer=2/2" in err
