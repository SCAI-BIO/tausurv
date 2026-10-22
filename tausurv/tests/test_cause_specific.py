"""Tests for the CauseSpecificPredictor composite competing-risks model."""

from __future__ import annotations

import functools
import json
import shutil

import numpy as np
import pytest

from tausurv import simulations
from tausurv.linear import CoxPH
from tausurv.predictor import CauseSpecificPredictor, SurvivalPredictor


def _cr_data(n=500, n_causes=2, seed=0):
    return simulations.competing_risks(n=n, n_features=5, n_causes=n_causes, seed=seed)


class _NoSavePredictor(SurvivalPredictor):
    """Wrapped-model stand-in without ``save``/``load``."""

    def fit(self, X, event_time, event_indicator):
        self.times_ = np.unique(np.asarray(event_time, dtype=np.float64))
        return self

    def _survival_function(self, X, times):
        return np.ones((len(X), len(times)))


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


def test_save_load_round_trip_predictions(tmp_path):
    X, T, E = _cr_data(n=300, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2).fit(X, T, E)
    grid = np.linspace(0.1, 3.0, 15)
    cif_before = model.predict_cif(X[:20], grid)
    survival_before = model.predict_survival_function(X[:20], grid)

    model.save(tmp_path / "cs")
    restored = CauseSpecificPredictor.load(tmp_path / "cs")

    np.testing.assert_allclose(restored.predict_cif(X[:20], grid), cif_before)
    np.testing.assert_allclose(
        restored.predict_survival_function(X[:20], grid), survival_before
    )


def test_save_load_layout_and_metadata(tmp_path):
    X, T, E = _cr_data(n=200, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2).fit(X, T, E)
    out = tmp_path / "cs"
    model.save(out)

    assert {p.name for p in out.iterdir()} == {
        "config.json",
        "times.npz",
        "cause_00",
        "cause_01",
    }
    cox_path = f"{CoxPH.__module__}.{CoxPH.__qualname__}"
    config = json.loads((out / "config.json").read_text())
    assert config == {
        "model": "CauseSpecificPredictor",
        "n_causes": 2,
        "cause_models": [cox_path, cox_path],
    }
    for k in range(2):
        assert (out / f"cause_{k:02d}" / "config.json").is_file()
        assert (out / f"cause_{k:02d}" / "state.npz").is_file()

    restored = CauseSpecificPredictor.load(out)
    assert restored.n_causes == 2
    assert len(restored.models_) == 2
    assert all(type(model_k) is CoxPH for model_k in restored.models_)
    np.testing.assert_array_equal(restored.times_, model.times_)
    assert restored.model_factory is None


def test_loaded_model_cannot_refit(tmp_path):
    X, T, E = _cr_data(n=200, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2).fit(X, T, E)
    model.save(tmp_path / "cs")
    restored = CauseSpecificPredictor.load(tmp_path / "cs")

    with pytest.raises(RuntimeError, match="model_factory"):
        restored.fit(X, T, E)


def test_save_unfitted_raises(tmp_path):
    model = CauseSpecificPredictor(CoxPH, n_causes=2)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.save(tmp_path / "cs")


def test_save_rejects_wrapped_model_without_save(tmp_path):
    X, T, E = _cr_data(n=100, n_causes=1)
    model = CauseSpecificPredictor(_NoSavePredictor, n_causes=1).fit(X, T, E)
    with pytest.raises(TypeError, match="no save"):
        model.save(tmp_path / "cs")


def test_load_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        CauseSpecificPredictor.load(tmp_path / "missing")


def test_load_wrong_model_name_raises(tmp_path):
    X, T, E = _cr_data(n=100, n_causes=1)
    model = CauseSpecificPredictor(CoxPH, n_causes=1).fit(X, T, E)
    out = tmp_path / "cs"
    model.save(out)
    config = json.loads((out / "config.json").read_text())
    config["model"] = "CoxPH"
    (out / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="CauseSpecificPredictor"):
        CauseSpecificPredictor.load(out)


def test_load_tampered_class_path_raises(tmp_path):
    X, T, E = _cr_data(n=100, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2).fit(X, T, E)
    out = tmp_path / "cs"
    model.save(out)
    config = json.loads((out / "config.json").read_text())
    config["cause_models"][1] = "tausurv.linear.NoSuchModel"
    (out / "config.json").write_text(json.dumps(config))

    with pytest.raises(ImportError, match="NoSuchModel"):
        CauseSpecificPredictor.load(out)


def test_load_class_without_load_raises(tmp_path):
    X, T, E = _cr_data(n=100, n_causes=1)
    model = CauseSpecificPredictor(CoxPH, n_causes=1).fit(X, T, E)
    out = tmp_path / "cs"
    model.save(out)
    config = json.loads((out / "config.json").read_text())
    config["cause_models"][0] = "tausurv.step.StepFunction"
    (out / "config.json").write_text(json.dumps(config))

    with pytest.raises(TypeError, match="no load"):
        CauseSpecificPredictor.load(out)


def test_load_missing_cause_dir_raises(tmp_path):
    X, T, E = _cr_data(n=100, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2).fit(X, T, E)
    out = tmp_path / "cs"
    model.save(out)
    shutil.rmtree(out / "cause_01")

    with pytest.raises(FileNotFoundError, match="cause_01"):
        CauseSpecificPredictor.load(out)


def test_load_n_causes_mismatch_raises(tmp_path):
    X, T, E = _cr_data(n=100, n_causes=2)
    model = CauseSpecificPredictor(CoxPH, n_causes=2).fit(X, T, E)
    out = tmp_path / "cs"
    model.save(out)
    config = json.loads((out / "config.json").read_text())
    config["n_causes"] = 3
    (out / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="cause_models"):
        CauseSpecificPredictor.load(out)


def test_partial_factory_round_trip(tmp_path):
    X, T, E = _cr_data(n=200, n_causes=2)
    factory = functools.partial(CoxPH, max_iter=37)
    model = CauseSpecificPredictor(factory, n_causes=2).fit(X, T, E)
    grid = np.linspace(0.1, 3.0, 10)
    expected = model.predict_cif(X[:10], grid)
    model.save(tmp_path / "cs")
    restored = CauseSpecificPredictor.load(tmp_path / "cs")

    np.testing.assert_allclose(restored.predict_cif(X[:10], grid), expected)
    assert restored.model_factory is None


def test_lambda_factory_round_trip(tmp_path):
    X, T, E = _cr_data(n=200, n_causes=2)
    model = CauseSpecificPredictor(lambda: CoxPH(), n_causes=2).fit(X, T, E)
    grid = np.linspace(0.1, 3.0, 10)
    expected = model.predict_cif(X[:10], grid)
    model.save(tmp_path / "cs")
    restored = CauseSpecificPredictor.load(tmp_path / "cs")

    np.testing.assert_allclose(restored.predict_cif(X[:10], grid), expected)
    assert restored.model_factory is None
