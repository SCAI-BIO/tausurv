from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from tausurv import simulations
from tausurv.nn import DeepHit, DeepSurv, LogisticHazard, Trainer, fit, functional


def test_deepsurv_save_and_load_round_trip(tmp_path):
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=32, n_blocks=3, dropout=0.2)
    x = torch.randn(8, 5)
    out_before = model(x)

    model.save(tmp_path / "ds")

    # Files exist with expected shape.
    config = json.loads((tmp_path / "ds" / "config.json").read_text())
    assert config["in_features"] == 5
    assert config["hidden_dim"] == 32
    assert config["n_blocks"] == 3
    assert (tmp_path / "ds" / "model.pt").exists()

    # Round-trip reproduces predictions exactly.
    restored = DeepSurv.load(tmp_path / "ds")
    restored.eval()
    model.eval()
    torch.testing.assert_close(restored(x), model(x))


def test_deephit_save_load_preserves_competing_risks_config(tmp_path):
    torch.manual_seed(0)
    model = DeepHit(in_features=4, n_bins=8, n_causes=3, hidden_dim=16)
    model.save(tmp_path / "dh")

    config = json.loads((tmp_path / "dh" / "config.json").read_text())
    assert config["n_causes"] == 3
    assert config["n_bins"] == 8

    restored = DeepHit.load(tmp_path / "dh")
    assert restored.n_causes == 3
    x = torch.randn(2, 4)
    model.eval()
    restored.eval()
    torch.testing.assert_close(restored(x), model(x))


def test_logistic_hazard_save_load(tmp_path):
    torch.manual_seed(0)
    model = LogisticHazard(in_features=5, n_bins=10, hidden_dim=16)
    model.save(tmp_path / "lh")

    restored = LogisticHazard.load(tmp_path / "lh")
    x = torch.randn(3, 5)
    model.eval()
    restored.eval()
    torch.testing.assert_close(restored(x), model(x))


def test_trainer_writes_config_and_model_files_alongside_state(tmp_path):
    """Every checkpoint directory written by Trainer should contain state.pt
    (resume-compatible) AND config.json + model.pt (load-compatible)."""
    torch.manual_seed(0)
    X, T, E = simulations.single_risk(n=200, seed=0)
    X_t, T_t, E_t = (torch.tensor(a, dtype=torch.float32) for a in (X, T, E))

    model = DeepSurv(in_features=5, hidden_dim=16, dropout=0.0)
    fit(
        model,
        functional.cox_nll,
        (X_t, T_t, E_t),
        (X_t, T_t, E_t),
        epochs=4,
        lr=1e-2,
        out_dir=tmp_path,
        save_every=2,
        verbose=False,
    )

    # best/, final/, step-*/ all have all three files
    expected_files = {"state.pt", "config.json", "model.pt"}
    for kind_dir in ("best", "final", "step-000002", "step-000004"):
        files = {p.name for p in (tmp_path / kind_dir).iterdir()}
        assert expected_files <= files, f"{kind_dir} missing: {expected_files - files}"


def test_trained_model_loaded_via_load_gives_same_predictions(tmp_path):
    """The full pipeline: train, save (via Trainer), load via load."""
    torch.manual_seed(0)
    X, T, E = simulations.single_risk(n=200, seed=0)
    X_t, T_t, E_t = (torch.tensor(a, dtype=torch.float32) for a in (X, T, E))

    model = DeepSurv(in_features=5, hidden_dim=16, dropout=0.0)
    fit(
        model,
        functional.cox_nll,
        (X_t, T_t, E_t),
        epochs=10,
        lr=1e-2,
        out_dir=tmp_path,
        verbose=False,
    )

    # Snapshot predictions from the trained model.
    model.eval()
    with torch.no_grad():
        ref = model(X_t)

    # Re-load via load — no need to remember hidden_dim etc.
    restored = DeepSurv.load(tmp_path / "final")
    restored.eval()
    with torch.no_grad():
        out = restored(X_t)

    torch.testing.assert_close(out, ref)
