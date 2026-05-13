from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tausurv import simulations
from tausurv.metrics.concordance import harrell
from tausurv.nn import DeepSurv, Trainer, fit, functional
from tausurv.nn.training.schedule import get_lr


def _data():
    X, T, E = simulations.single_risk(n=300, n_features=5, seed=0)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(T, dtype=torch.float32),
        torch.tensor(E, dtype=torch.float32),
    )


def test_schedule_constant_is_flat():
    assert get_lr("constant", 0, 100, 1e-3) == 1e-3
    assert get_lr("constant", 99, 100, 1e-3) == 1e-3


def test_schedule_cosine_decays_to_zero():
    base = 1e-3
    assert math.isclose(get_lr("cosine", 0, 100, base), base, rel_tol=1e-2)
    assert get_lr("cosine", 99, 100, base) < 0.01 * base


def test_fit_runs_and_reduces_loss():
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    train = _data()
    history = fit(
        model,
        functional.cox_nll,
        train,
        epochs=30,
        lr=1e-2,
        verbose=False,
    )
    assert len(history["train_loss"]) == 30
    assert history["train_loss"][-1] < history["train_loss"][0] - 0.05
    assert history["best_value"] is None  # no val data -> no best tracking


def test_fit_with_val_data_tracks_best():
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    train = _data()
    val = _data()
    history = fit(
        model,
        functional.cox_nll,
        train,
        val,
        epochs=20,
        lr=1e-2,
        verbose=False,
    )
    assert all(v is not None for v in history["val_loss"])
    assert history["best_epoch"] >= 0
    assert history["best_value"] is not None


def test_fit_early_stopping_triggers(tmp_path):
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    train = _data()
    val = _data()
    history = fit(
        model,
        functional.cox_nll,
        train,
        val,
        epochs=100,
        lr=1e-2,
        patience=2,
        verbose=False,
    )
    # Should stop before 100 epochs.
    assert len(history["train_loss"]) < 100


def test_fit_writes_live_logs_and_checkpoints(tmp_path):
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    train = _data()
    val = _data()
    history = fit(
        model,
        functional.cox_nll,
        train,
        val,
        epochs=6,
        lr=1e-2,
        out_dir=tmp_path,
        save_every=3,
        verbose=False,
    )
    # JSONL log exists and has at least 6 train records
    jsonl = (tmp_path / "train.jsonl").read_text().strip().splitlines()
    records = [json.loads(line) for line in jsonl]
    train_events = [r for r in records if r["event"] == "train"]
    assert len(train_events) == 6
    # best/ and final/ dirs exist
    assert (tmp_path / "best" / "state.pt").exists()
    assert (tmp_path / "final" / "state.pt").exists()
    # step-NNNNNN dirs exist
    step_dirs = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("step-"))
    assert step_dirs == ["step-000003", "step-000006"]


def test_fit_resume_continues_history_and_preserves_best(tmp_path):
    """Resume should: (a) extend history with pre+post epochs, (b) preserve best-tracking
    (i.e., not reset to inf), (c) write a "resumed" message to the log."""
    torch.manual_seed(0)
    model1 = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    history1 = fit(
        model1,
        functional.cox_nll,
        _data(),
        _data(),
        epochs=5,
        lr=1e-2,
        out_dir=tmp_path,
        verbose=False,
    )
    best_before = history1["best_value"]
    assert len(history1["train_loss"]) == 5
    assert history1["best_epoch"] >= 0

    # Resume into a fresh model (weights loaded from checkpoint).
    model2 = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    history2 = fit(
        model2,
        functional.cox_nll,
        _data(),
        _data(),
        epochs=8,
        lr=1e-2,
        out_dir=tmp_path,
        resume_from=tmp_path / "final",
        verbose=False,
    )
    # History continued: pre-resume's 5 epochs are at the front, plus 3 new.
    assert len(history2["train_loss"]) == 8
    np.testing.assert_array_equal(history2["train_loss"][:5], history1["train_loss"])
    # Best-value tracking preserved across resume (didn't reset to inf).
    assert history2["best_value"] <= best_before


def test_fit_resume_uninterrupted_equivalence(tmp_path):
    """8 epochs straight vs 5+resume+3 give the same trajectory (RNG restored)."""
    torch.manual_seed(0)
    m_all = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    h_all = fit(m_all, functional.cox_nll, _data(), epochs=8, lr=1e-2, verbose=False)

    torch.manual_seed(0)
    m_part = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    fit(
        m_part,
        functional.cox_nll,
        _data(),
        epochs=5,
        lr=1e-2,
        out_dir=tmp_path / "split",
        verbose=False,
    )

    m_part2 = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    h_resume = fit(
        m_part2,
        functional.cox_nll,
        _data(),
        epochs=8,
        lr=1e-2,
        out_dir=tmp_path / "split2",
        resume_from=tmp_path / "split" / "final",
        verbose=False,
    )

    np.testing.assert_allclose(h_all["train_loss"], h_resume["train_loss"], atol=1e-5)


def test_fit_with_scalar_val_metric():
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    train = _data()
    val = _data()

    def harrell_score(predictions, T, E):
        risk = predictions.cpu().numpy()
        return harrell(T.cpu().numpy(), E.cpu().numpy().astype(int), risk)

    history = fit(
        model,
        functional.cox_nll,
        train,
        val,
        epochs=15,
        lr=1e-2,
        val_metric=harrell_score,
        val_direction="max",
        verbose=False,
    )
    # Scalar metric is logged under "value".
    assert "value" in history["val_metrics"]
    assert len(history["val_metrics"]["value"]) == 15
    assert history["best_value"] > 0.5
    assert history["best_metric"] == "value"


def test_fit_with_dict_val_metric_logs_all_and_tracks_one():
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    train = _data()
    val = _data()

    from tausurv.metrics import brier

    def survival_metrics(predictions, T, E):
        T_np = T.cpu().numpy()
        E_np = E.cpu().numpy().astype(int)
        risk = predictions.cpu().numpy()
        # Build a survival prediction for Brier via a flat exp-based form.
        time_grid = np.linspace(0.05, float(np.quantile(T_np, 0.9)), 25)
        H = time_grid[None, :]
        survival = np.exp(-H * np.exp(risk)[:, None] * 0.5)
        ibs = brier.integrated(T_np, E_np, survival, time_grid)
        return {
            "harrell": harrell(T_np, E_np, risk),
            "ibs": ibs,
        }

    history = fit(
        model,
        functional.cox_nll,
        train,
        val,
        epochs=10,
        lr=1e-2,
        val_metric=survival_metrics,
        best_from="harrell",
        val_direction="max",
        verbose=False,
    )
    # Both metrics are tracked, but only one drives best.
    assert "harrell" in history["val_metrics"]
    assert "ibs" in history["val_metrics"]
    assert len(history["val_metrics"]["harrell"]) == 10
    assert len(history["val_metrics"]["ibs"]) == 10
    assert history["best_metric"] == "harrell"
    assert history["best_value"] > 0.5


def test_trainer_class_matches_fit_function():
    """Trainer.fit() and fit() function produce identical histories."""
    torch.manual_seed(0)
    model1 = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    h1 = fit(model1, functional.cox_nll, _data(), epochs=10, lr=1e-2, verbose=False)

    torch.manual_seed(0)
    model2 = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    trainer = Trainer(model2, loss_fn=functional.cox_nll, lr=1e-2)
    h2 = trainer.fit(_data(), epochs=10, verbose=False)

    np.testing.assert_allclose(h1["train_loss"], h2["train_loss"])


def test_trainer_subclass_can_override_train_step():
    """Demonstrate extensibility: a subclass adds L2 regularization on predictions."""
    torch.manual_seed(0)

    class RegularizedTrainer(Trainer):
        def __init__(self, model, *, l2_lambda: float = 0.01, **kwargs):
            super().__init__(model, **kwargs)
            self.l2_lambda = l2_lambda

        def train_step(self, batch):
            self.model.train()
            X = batch[0]
            targets = batch[1:]
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(X)
            loss = self.loss_fn(predictions, *targets)
            loss = loss + self.l2_lambda * (predictions**2).mean()
            if not torch.isfinite(loss):
                return loss
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            return loss

    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    trainer = RegularizedTrainer(
        model, loss_fn=functional.cox_nll, lr=1e-2, l2_lambda=0.001
    )
    history = trainer.fit(_data(), epochs=20, verbose=False)
    assert len(history["train_loss"]) == 20
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_trainer_subclass_can_override_only_compute_loss():
    """Subclasses customising the loss math should override only
    :meth:`compute_loss` — the inherited :meth:`train_step` handles
    zero_grad, backward, grad-clip, and optimiser step."""
    torch.manual_seed(0)

    class RegularizedTrainer(Trainer):
        def __init__(self, model, *, l2_lambda: float = 0.01, **kwargs):
            super().__init__(model, **kwargs)
            self.l2_lambda = l2_lambda

        def compute_loss(self, batch):
            X, *targets = batch
            predictions = self.model(X)
            base = self.loss_fn(predictions, *targets)
            return base + self.l2_lambda * (predictions**2).mean()

    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    trainer = RegularizedTrainer(
        model, loss_fn=functional.cox_nll, lr=1e-2, l2_lambda=0.001
    )
    history = trainer.fit(_data(), epochs=20, verbose=False)
    assert len(history["train_loss"]) == 20
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_trainer_compute_loss_handles_grad_clip_from_base_train_step():
    """A compute_loss-only subclass still benefits from grad_clip set
    on the base Trainer — the inherited train_step applies it."""
    torch.manual_seed(0)

    class CustomLossTrainer(Trainer):
        def compute_loss(self, batch):
            X, *targets = batch
            return self.loss_fn(self.model(X), *targets)

    model = DeepSurv(in_features=5, hidden_dim=16, dropout=0.0)
    trainer = CustomLossTrainer(
        model, loss_fn=functional.cox_nll, lr=1e-2, grad_clip=1.0
    )
    history = trainer.fit(_data(), epochs=5, verbose=False)
    assert all(math.isfinite(x) for x in history["train_loss"])


def test_trainer_subclass_without_loss_fn_overrides_both_steps():
    """A fully-custom trainer can skip loss_fn entirely by overriding both steps."""
    torch.manual_seed(0)

    class CustomTrainer(Trainer):
        def train_step(self, batch):
            self.model.train()
            X = batch[0]
            targets = batch[1:]
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(X)
            loss = functional.cox_nll(predictions, *targets)
            if not torch.isfinite(loss):
                return loss
            loss.backward()
            self.optimizer.step()
            return loss

        def eval_step(self, batch):
            self.model.eval()
            X = batch[0]
            targets = batch[1:]
            with torch.no_grad():
                predictions = self.model(X)
                loss = functional.cox_nll(predictions, *targets)
            return predictions, loss

    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    trainer = CustomTrainer(model, loss_fn=None, lr=1e-2)  # no loss_fn passed
    history = trainer.fit(_data(), _data(), epochs=10, verbose=False)
    assert len(history["train_loss"]) == 10


def test_trainer_raises_when_default_step_needs_loss_fn():
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    trainer = Trainer(model, loss_fn=None, lr=1e-2)
    with pytest.raises(NotImplementedError, match="loss_fn"):
        trainer.fit(_data(), epochs=1, verbose=False)


def test_trainer_predict_runs_in_eval_mode_no_grad():
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=16, dropout=0.5)
    trainer = Trainer(model, loss_fn=functional.cox_nll, lr=1e-2)
    X, *_ = _data()
    # Two predict calls on the same X should return identical results
    # (eval mode → dropout off → deterministic).
    p1 = trainer.predict(X)
    p2 = trainer.predict(X)
    torch.testing.assert_close(p1, p2)
    assert not p1.requires_grad  # no_grad context


def test_trainer_override_train_epoch_for_two_stage():
    """Demonstrate: a subclass that does two optimizer.step()s per epoch."""
    torch.manual_seed(0)

    class TwoStageTrainer(Trainer):
        def train_epoch(self, train_tensors, *, batch_size=None):
            # Stage 1: warmup half the parameters with smaller LR.
            base_lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.param_groups[0]["lr"] = base_lr * 0.1
            loss1 = self.train_step(train_tensors)
            # Stage 2: full LR.
            self.optimizer.param_groups[0]["lr"] = base_lr
            loss2 = self.train_step(train_tensors)
            return float((loss1 + loss2).item() / 2)

    model = DeepSurv(in_features=5, hidden_dim=16, dropout=0.0)
    trainer = TwoStageTrainer(model, loss_fn=functional.cox_nll, lr=1e-2)
    history = trainer.fit(_data(), epochs=5, verbose=False)
    assert len(history["train_loss"]) == 5


def test_trainer_override_evaluate_for_custom_metrics():
    """Demonstrate: a subclass that runs evaluation twice and averages."""
    torch.manual_seed(0)

    class BaggingEvalTrainer(Trainer):
        def evaluate(self, val_tensors, val_metric=None):
            # Run eval twice (e.g., MC-dropout style) and average the losses.
            loss1, _ = super().evaluate(val_tensors, val_metric=None)
            loss2, m = super().evaluate(val_tensors, val_metric)
            return (loss1 + loss2) / 2, m

    model = DeepSurv(in_features=5, hidden_dim=16, dropout=0.0)
    trainer = BaggingEvalTrainer(model, loss_fn=functional.cox_nll, lr=1e-2)
    history = trainer.fit(_data(), _data(), epochs=3, verbose=False)
    assert len(history["val_loss"]) == 3


def test_fit_writes_startup_summary(tmp_path):
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=16, dropout=0.0)
    fit(
        model,
        functional.cox_nll,
        _data(),
        epochs=2,
        lr=1e-2,
        out_dir=tmp_path,
        verbose=False,
    )
    log = (tmp_path / "train.log").read_text()
    assert "tausurv training" in log
    assert "DeepSurv" in log
    assert "AdamW" in log


def test_fit_dict_metric_rejects_unknown_best_from():
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    train = _data()
    val = _data()

    def metrics(predictions, *targets):
        return {"harrell": 0.6}

    with pytest.raises(ValueError, match="best_from"):
        fit(
            model,
            functional.cox_nll,
            train,
            val,
            epochs=2,
            val_metric=metrics,
            best_from="ibs",
            verbose=False,
        )


def test_trainer_accepts_optimizer_instance():
    """A pre-constructed Optimizer is used as-is; ``lr`` arg is ignored."""
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    opt = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.7)

    trainer = Trainer(model, loss_fn=functional.cox_nll, optimizer=opt, lr=999.0)
    assert trainer.optimizer is opt
    assert trainer.base_lr == pytest.approx(5e-3)
    history = trainer.fit(_data(), epochs=5, verbose=False)
    assert history["lr"][0] == pytest.approx(5e-3)


def test_trainer_accepts_optimizer_factory():
    """A callable factory is invoked with model.parameters()."""
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)

    def make_opt(params):
        return torch.optim.Adam(params, lr=2e-3, betas=(0.85, 0.99))

    trainer = Trainer(model, loss_fn=functional.cox_nll, optimizer=make_opt)
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert trainer.optimizer.defaults["betas"] == (0.85, 0.99)
    assert trainer.base_lr == pytest.approx(2e-3)
    history = trainer.fit(_data(), epochs=5, verbose=False)
    assert len(history["train_loss"]) == 5


def test_trainer_rejects_unknown_optimizer_type():
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    with pytest.raises(TypeError, match="optimizer"):
        Trainer(model, loss_fn=functional.cox_nll, optimizer=42)


def test_trainer_accepts_lr_scheduler_instance():
    """A torch LRScheduler instance is stepped once per epoch by the Trainer."""
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)

    trainer = Trainer(model, loss_fn=functional.cox_nll, optimizer=opt, schedule=sched)
    history = trainer.fit(_data(), epochs=6, verbose=False)
    lrs = history["lr"]
    # StepLR halves every 2 epochs starting from 1e-2.
    expected = [1e-2, 1e-2, 5e-3, 5e-3, 2.5e-3, 2.5e-3]
    np.testing.assert_allclose(lrs, expected, rtol=1e-6)


def test_trainer_rejects_scheduler_bound_to_different_optimizer():
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    other = torch.optim.SGD(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(other, step_size=1, gamma=0.5)
    with pytest.raises(ValueError, match="same optimizer"):
        Trainer(model, loss_fn=functional.cox_nll, schedule=sched)


def test_trainer_resume_with_lr_scheduler_instance(tmp_path):
    """Resume should restore the scheduler position so lr continues decaying correctly."""
    torch.manual_seed(0)
    # Reference: 6 epochs straight with a StepLR(2, 0.5) schedule.
    m_ref = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    opt_ref = torch.optim.SGD(m_ref.parameters(), lr=1e-2)
    sched_ref = torch.optim.lr_scheduler.StepLR(opt_ref, step_size=2, gamma=0.5)
    h_ref = Trainer(
        m_ref,
        loss_fn=functional.cox_nll,
        optimizer=opt_ref,
        schedule=sched_ref,
    ).fit(_data(), epochs=6, verbose=False)

    # Split: 3 epochs, save, then resume for 3 more.
    torch.manual_seed(0)
    m_a = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    opt_a = torch.optim.SGD(m_a.parameters(), lr=1e-2)
    sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=2, gamma=0.5)
    Trainer(
        m_a,
        loss_fn=functional.cox_nll,
        optimizer=opt_a,
        schedule=sched_a,
    ).fit(_data(), epochs=3, out_dir=tmp_path / "a", verbose=False)

    # Fresh scheduler — resume must restore its state from the checkpoint.
    m_b = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    opt_b = torch.optim.SGD(m_b.parameters(), lr=1e-2)
    sched_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=2, gamma=0.5)
    h_resume = Trainer(
        m_b,
        loss_fn=functional.cox_nll,
        optimizer=opt_b,
        schedule=sched_b,
    ).fit(
        _data(),
        epochs=6,
        out_dir=tmp_path / "b",
        resume_from=tmp_path / "a" / "final",
        verbose=False,
    )

    np.testing.assert_allclose(h_resume["lr"], h_ref["lr"], rtol=1e-6)


def test_trainer_rejects_unknown_schedule_type():
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    with pytest.raises(TypeError, match="schedule"):
        Trainer(model, loss_fn=functional.cox_nll, schedule=42)
