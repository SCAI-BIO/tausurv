from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from tausurv.nn.training.checkpoint import load_state, prune_checkpoints, save_state
from tausurv.nn.training.logger import Logger
from tausurv.nn.training.optimizer import build_optimizer
from tausurv.nn.training.schedule import get_lr


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_device(data: tuple[Any, ...], device: torch.device) -> tuple[Tensor, ...]:
    return tuple(
        t.to(device) if isinstance(t, Tensor) else torch.as_tensor(t).to(device)
        for t in data
    )


def _pick_tracked(
    val_metrics: dict[str, float] | None,
    best_from: str | None,
    val_loss: float | None,
) -> tuple[float | None, str | None]:
    """Pick the (value, key) used for best-checkpoint and early-stopping."""
    if val_metrics is None:
        return val_loss, ("val_loss" if val_loss is not None else None)
    if best_from is not None:
        if best_from not in val_metrics:
            raise ValueError(
                f"best_from={best_from!r} not in val_metric keys {list(val_metrics)}"
            )
        return val_metrics[best_from], best_from
    key = "value" if "value" in val_metrics else next(iter(val_metrics))
    return val_metrics[key], key


class Trainer:
    r"""Survival training orchestrator. Four override points; one main loop.

    Mental model:

    - :meth:`train_step` — the per-batch training math (forward → loss →
      backward → step). Override for custom losses, regularizers, adversarial
      updates. Most subclasses override only this.
    - :meth:`eval_step` — the per-batch inference math. Override to return
      richer prediction objects (e.g., CIFs per cause, counterfactuals).
    - :meth:`train_epoch` — orchestrates :meth:`train_step` over batches.
      Override for non-standard epoch structure (two-stage training,
      alternating updates).
    - :meth:`evaluate` — orchestrates :meth:`eval_step` and applies
      ``val_metric``. Override for custom metric pipelines (e.g., metrics
      that need cross-fitted predictions).

    :meth:`fit` is the main loop and is *not* an override point. It owns
    scheduling, logging, checkpointing, best-tracking, early-stopping,
    interrupt handling. Subclasses customize behavior through the four
    methods above.

    No callbacks. No hooks. No config dataclass. No third-party dependency
    beyond torch + numpy.

    Parameters
    ----------
    model : nn.Module
    loss_fn : Callable, optional
        Called as ``loss_fn(predictions, *targets) -> Tensor``. Required by
        default :meth:`train_step` / :meth:`eval_step`; subclasses that fully
        override both may pass ``None``.
    optimizer : str, torch.optim.Optimizer, or callable, default "adamw"
        ``"adamw"`` / ``"adam"`` / ``"sgd"`` — built with ``lr`` / ``weight_decay``.
        A pre-constructed :class:`torch.optim.Optimizer` is used as-is. A
        callable factory is invoked with ``model.parameters()`` and must return
        an :class:`~torch.optim.Optimizer`. When an instance or factory is
        given, ``lr`` and ``weight_decay`` are ignored.
    lr : float, default 1e-3
    weight_decay : float, default 0.0
    schedule : str or torch.optim.lr_scheduler.LRScheduler, default "constant"
        ``"constant"`` / ``"cosine"`` / ``"linear_warmup_cosine"`` use a built-in
        epoch-indexed formula. Alternatively pass an :class:`LRScheduler` whose
        ``.optimizer`` is the same optimizer the Trainer uses; the Trainer will
        call ``scheduler.step()`` once per epoch and persist its state on
        resume.
    grad_clip : float, default 0.0
        Gradient-norm clip; ``0`` disables.
    nan_tolerance : int, default 5
        Consecutive non-finite training losses tolerated before :meth:`fit` raises.
    device : str or torch.device, optional
        Defaults to CUDA if available, else CPU.
    seed : int, default 0
        Seeds ``random``, ``numpy``, ``torch``.

    Examples
    --------
    Default use:

    >>> trainer = Trainer(model, loss_fn=cox_nll, lr=1e-2)
    >>> history = trainer.fit(train_data, val_data, epochs=100)

    Custom training math (add a regularizer):

    >>> class MyTrainer(Trainer):
    ...     def __init__(self, model, *, l2=0.01, **kw):
    ...         super().__init__(model, **kw)
    ...         self.l2 = l2
    ...     def train_step(self, batch):
    ...         self.model.train()
    ...         X, *targets = batch
    ...         self.optimizer.zero_grad(set_to_none=True)
    ...         pred = self.model(X)
    ...         loss = self.loss_fn(pred, *targets) + self.l2 * (pred ** 2).mean()
    ...         if not torch.isfinite(loss):
    ...             return loss
    ...         loss.backward()
    ...         if self.grad_clip > 0:
    ...             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
    ...         self.optimizer.step()
    ...         return loss
    """

    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Callable | None
    device: torch.device

    def __init__(
        self,
        model: nn.Module,
        *,
        loss_fn: Callable | None = None,
        optimizer: "str | torch.optim.Optimizer | Callable[..., torch.optim.Optimizer]" = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        schedule: "str | torch.optim.lr_scheduler.LRScheduler" = "constant",
        grad_clip: float = 0.0,
        nan_tolerance: int = 5,
        device: str | torch.device | None = None,
        seed: int = 0,
    ) -> None:
        device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        _set_seed(seed)

        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = build_optimizer(self.model, optimizer, lr, weight_decay)

        # base_lr drives the built-in string-keyed schedules. Read from the
        # optimizer so that pre-constructed instances/factories carry their
        # own lr through transparently.
        self.base_lr = float(self.optimizer.param_groups[0]["lr"])

        if isinstance(schedule, str):
            self.schedule = schedule
            self._lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        elif isinstance(schedule, torch.optim.lr_scheduler.LRScheduler):
            if schedule.optimizer is not self.optimizer:
                raise ValueError(
                    "LRScheduler.optimizer must be the same optimizer used by the "
                    "Trainer. Build the scheduler against the optimizer you pass in."
                )
            self.schedule = type(schedule).__name__
            self._lr_scheduler = schedule
        else:
            raise TypeError(
                "schedule must be a string or torch.optim.lr_scheduler.LRScheduler, "
                f"got {type(schedule).__name__}"
            )

        self.grad_clip = grad_clip
        self.nan_tolerance = nan_tolerance
        self.device = device
        self.seed = seed

    def train_step(self, batch: tuple[Tensor, ...]) -> Tensor:
        """Run one optimizer step on ``batch``. Returns the loss tensor.

        Default behavior: forward → loss → backward → grad-clip → step.
        Non-finite losses are returned as-is for :meth:`fit` to count toward
        ``nan_tolerance``. Override for custom training math.
        """
        if self.loss_fn is None:
            raise NotImplementedError(
                "Default train_step requires loss_fn. Pass loss_fn or override train_step."
            )
        self.model.train()
        X = batch[0]
        targets = batch[1:]
        self.optimizer.zero_grad(set_to_none=True)
        predictions = self.model(X)
        loss = self.loss_fn(predictions, *targets)
        if not torch.isfinite(loss):
            return loss
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss

    def eval_step(self, batch: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
        """Run one no-grad evaluation step. Returns ``(predictions, loss)``.

        Symmetric with :meth:`train_step`: ``predictions = model(X)``,
        ``loss = loss_fn(predictions, *targets)``. Override for richer
        eval semantics — e.g., models whose forward needs more than ``X``
        (see :class:`CopulaSurvTrainer`).
        """
        if self.loss_fn is None:
            raise NotImplementedError(
                "Default eval_step requires loss_fn. Pass loss_fn or override eval_step."
            )
        self.model.eval()
        X = batch[0]
        targets = batch[1:]
        with torch.no_grad():
            predictions = self.model(X)
            loss = self.loss_fn(predictions, *targets)
        return predictions, loss

    def train_epoch(
        self,
        train_tensors: tuple[Tensor, ...],
        *,
        batch_size: int | None = None,
    ) -> float:
        """Run one training epoch over ``train_tensors``. Returns mean loss.

        Default: full batch if ``batch_size is None``, else random-shuffled
        mini-batches. Override for non-standard epoch structure (two-stage,
        alternating updates).
        """
        X = train_tensors[0]
        targets = train_tensors[1:]
        n = X.size(0)

        if batch_size is None or batch_size >= n:
            loss = self.train_step((X, *targets))
            return float(loss.item()) if torch.isfinite(loss) else float("nan")

        perm = torch.randperm(n, device=X.device)
        total = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch = (X[idx], *(t[idx] for t in targets))
            loss = self.train_step(batch)
            if not torch.isfinite(loss):
                return float("nan")
            total += float(loss.item())
            n_batches += 1
        return total / max(n_batches, 1)

    def evaluate(
        self,
        val_tensors: tuple[Tensor, ...],
        val_metric: Callable | None = None,
    ) -> tuple[float, dict[str, float] | None]:
        """Run evaluation. Returns ``(loss, metrics_dict)``.

        Default: single full-batch :meth:`eval_step`, then apply
        ``val_metric`` to ``(predictions, *targets)``. Scalar returns are
        wrapped as ``{"value": x}``; dict returns flow through unchanged.
        Override for custom metric pipelines (e.g., metric needs
        predictions evaluated on multiple held-out folds).
        """
        targets = val_tensors[1:]
        predictions, loss = self.eval_step(val_tensors)
        val_loss = float(loss.item())
        if val_metric is None:
            return val_loss, None
        result = val_metric(predictions, *targets)
        if isinstance(result, dict):
            return val_loss, {k: float(v) for k, v in result.items()}
        return val_loss, {"value": float(result)}

    def predict(self, X: Tensor) -> Tensor:
        """Inference on ``X``: eval mode + ``no_grad`` + model forward."""
        X = (
            X.to(self.device)
            if isinstance(X, Tensor)
            else torch.as_tensor(X).to(self.device)
        )
        self.model.eval()
        with torch.no_grad():
            return self.model(X)

    def fit(
        self,
        train_data: tuple,
        val_data: tuple | None = None,
        *,
        epochs: int = 100,
        batch_size: int | None = None,
        val_metric: Callable[..., "float | dict[str, float]"] | None = None,
        val_every: int = 1,
        patience: int | None = None,
        val_direction: str = "min",
        best_from: str | None = None,
        out_dir: str | Path | None = None,
        save_best: bool = True,
        save_every: int | None = None,
        keep_last: int = 3,
        resume_from: str | Path | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run the training loop. Returns a history dict; writes to ``out_dir`` if set.

        See class docstring for design philosophy; this method is not
        intended for override.
        """
        if val_direction not in ("min", "max"):
            raise ValueError(
                f"val_direction must be 'min' or 'max', got {val_direction!r}"
            )
        if patience is not None and val_data is None:
            raise ValueError("patience requires val_data")
        out_dir_path = Path(out_dir) if out_dir is not None else None

        # Setup
        start_epoch = 0
        saved_extra: dict[str, Any] = {}
        if resume_from is not None:
            start_epoch, saved_extra = load_state(
                resume_from, self.model, self.optimizer
            )
            sched_state = saved_extra.get("scheduler_state")
            if sched_state is not None and self._lr_scheduler is not None:
                self._lr_scheduler.load_state_dict(sched_state)

        train_tensors = _to_device(train_data, self.device)
        val_tensors = (
            _to_device(val_data, self.device) if val_data is not None else None
        )

        logger = Logger(out_dir_path, verbose=verbose, total_epochs=epochs)
        self._log_summary(
            logger,
            n_samples=train_tensors[0].size(0),
            epochs=epochs,
            batch_size=batch_size,
            has_val=val_tensors is not None,
            out_dir=out_dir_path,
            resume_from=resume_from,
            start_epoch=start_epoch,
        )

        # Restore training state from checkpoint if resuming, else fresh defaults.
        history: dict[str, Any] = saved_extra.get("history") or {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": {},
            "lr": [],
            "epoch_ms": [],
        }
        for k in ("train_loss", "val_loss", "lr", "epoch_ms"):
            history.setdefault(k, [])
        history.setdefault("val_metrics", {})

        best_value = saved_extra.get(
            "best_value", math.inf if val_direction == "min" else -math.inf
        )
        best_epoch = saved_extra.get("best_epoch", -1)
        best_key: str | None = saved_extra.get("best_metric")

        rng = saved_extra.get("rng_state")
        if rng is not None:
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy_state"])
            random.setstate(rng["python_state"])
            logger.message(
                f"resumed from epoch {start_epoch} "
                f"(best {best_key}={best_value:.4g} @ epoch {best_epoch + 1})"
                if best_epoch >= 0
                else f"resumed from epoch {start_epoch}"
            )
        n_nan = 0
        n_bad_val = 0
        interrupted = False
        epoch = start_epoch - 1

        # Loop
        try:
            for epoch in range(start_epoch, epochs):
                t0 = time.time()
                if self._lr_scheduler is None:
                    cur_lr = get_lr(self.schedule, epoch, epochs, self.base_lr)
                    for g in self.optimizer.param_groups:
                        g["lr"] = cur_lr
                else:
                    cur_lr = float(self.optimizer.param_groups[0]["lr"])

                train_loss = self.train_epoch(train_tensors, batch_size=batch_size)
                if not math.isfinite(train_loss):
                    n_nan += 1
                    logger.message(
                        f"non-finite loss at epoch {epoch + 1} ({n_nan} consecutive)"
                    )
                    if n_nan > self.nan_tolerance:
                        raise RuntimeError(
                            f"loss diverged: {n_nan} consecutive non-finite epochs"
                        )
                    continue
                n_nan = 0

                val_loss: float | None = None
                val_metrics: dict[str, float] | None = None
                if val_tensors is not None and (
                    epoch % val_every == 0 or epoch == epochs - 1
                ):
                    val_loss, val_metrics = self.evaluate(val_tensors, val_metric)

                epoch_ms = 1000.0 * (time.time() - t0)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["lr"].append(cur_lr)
                history["epoch_ms"].append(epoch_ms)
                if val_metrics is not None:
                    for name, value in val_metrics.items():
                        history["val_metrics"].setdefault(name, []).append(value)

                logger.log_train(epoch=epoch, loss=train_loss, lr=cur_lr, ms=epoch_ms)
                if val_loss is not None:
                    val_fields: dict[str, Any] = {"loss": val_loss}
                    if val_metrics is not None:
                        val_fields.update(val_metrics)
                    logger.log_val(epoch=epoch, **val_fields)

                # Advance the instance scheduler so optimizer.lr is set for the
                # NEXT epoch. Subsequent checkpoint saves capture this state so
                # resume continues seamlessly.
                if self._lr_scheduler is not None:
                    self._lr_scheduler.step()

                tracked, tracked_key = _pick_tracked(val_metrics, best_from, val_loss)
                if tracked is not None:
                    improved = (val_direction == "min" and tracked < best_value) or (
                        val_direction == "max" and tracked > best_value
                    )
                    if improved:
                        best_value = tracked
                        best_epoch = epoch
                        best_key = tracked_key
                        n_bad_val = 0
                        if save_best and out_dir_path is not None:
                            self._save(
                                out_dir_path / "best",
                                epoch_idx=epoch,
                                logger=logger,
                                kind="best",
                                extra=self._capture_resume_state(
                                    best_value=best_value,
                                    best_epoch=best_epoch,
                                    best_key=best_key,
                                    history=history,
                                ),
                            )
                    else:
                        n_bad_val += 1

                if (
                    save_every
                    and (epoch + 1) % save_every == 0
                    and out_dir_path is not None
                ):
                    step_path = out_dir_path / f"step-{epoch + 1:06d}"
                    self._save(
                        step_path,
                        epoch_idx=epoch,
                        logger=logger,
                        kind="step",
                        extra=self._capture_resume_state(
                            best_value=best_value,
                            best_epoch=best_epoch,
                            best_key=best_key,
                            history=history,
                        ),
                    )
                    prune_checkpoints(out_dir_path, keep_last)

                if patience is not None and n_bad_val >= patience:
                    logger.message(
                        f"early stopping at epoch {epoch + 1} (patience {patience})"
                    )
                    break

        except KeyboardInterrupt:
            interrupted = True
            logger.message("interrupted; saving state…")
            if out_dir_path is not None:
                self._save(
                    out_dir_path / "interrupt",
                    epoch_idx=epoch,
                    logger=logger,
                    kind="interrupt",
                    extra=self._capture_resume_state(
                        best_value=best_value,
                        best_epoch=best_epoch,
                        best_key=best_key,
                        history=history,
                    ),
                )

        if out_dir_path is not None and not interrupted:
            self._save(
                out_dir_path / "final",
                epoch_idx=epoch,
                logger=logger,
                kind="final",
                extra=self._capture_resume_state(
                    best_value=best_value,
                    best_epoch=best_epoch,
                    best_key=best_key,
                    history=history,
                ),
            )

        logger.close()

        history["best_epoch"] = best_epoch
        history["best_value"] = best_value if best_epoch >= 0 else None
        history["best_metric"] = best_key if best_epoch >= 0 else None
        return history

    def _save(
        self,
        path: Path,
        *,
        epoch_idx: int,
        logger: Logger,
        kind: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        save_state(path, self.model, self.optimizer, epoch=epoch_idx + 1, extra=extra)
        # If the model supports PretrainedMixin, also write config.json + model.pt
        # so the directory is loadable via Model.from_pretrained(path).
        save_pretrained = getattr(self.model, "save_pretrained", None)
        if callable(save_pretrained):
            save_pretrained(path)
        logger.log_checkpoint(epoch=epoch_idx, path=path, kind=kind)

    def _capture_resume_state(
        self,
        *,
        best_value: float,
        best_epoch: int,
        best_key: str | None,
        history: dict[str, Any],
    ) -> dict[str, Any]:
        """Snapshot of fit-loop state needed for transparent resume."""
        state: dict[str, Any] = {
            "best_value": best_value,
            "best_epoch": best_epoch,
            "best_metric": best_key,
            "history": history,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy_state": np.random.get_state(),
                "python_state": random.getstate(),
            },
        }
        if self._lr_scheduler is not None:
            state["scheduler_state"] = self._lr_scheduler.state_dict()
        return state

    def _log_summary(
        self,
        logger: Logger,
        *,
        n_samples: int,
        epochs: int,
        batch_size: int | None,
        has_val: bool,
        out_dir: Path | None,
        resume_from: str | Path | None,
        start_epoch: int,
    ) -> None:
        n_params = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        batch_str = "full" if batch_size is None else str(batch_size)
        logger.message("─── tausurv training ────────────────────")
        logger.message(
            f"  model   {type(self.model).__name__}  "
            f"{n_params / 1e3:.1f}K params ({n_train / 1e3:.1f}K trainable)"
        )
        logger.message(
            f"  data    n={n_samples}  batch_size={batch_str}  val={'yes' if has_val else 'no'}"
        )
        logger.message(
            f"  optim   {type(self.optimizer).__name__}  "
            f"lr={self.base_lr:g}  schedule={self.schedule}  grad_clip={self.grad_clip:g}"
        )
        logger.message(
            f"  loop    {epochs - start_epoch} epochs (from {start_epoch})  device={self.device}"
        )
        if out_dir is not None:
            logger.message(f"  out     {out_dir}")
        if resume_from is not None:
            logger.message(f"  resume  ← {resume_from}")
        logger.message("─────────────────────────────────────────")
