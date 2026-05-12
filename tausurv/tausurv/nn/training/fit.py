from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tausurv.nn.training.trainer import Trainer


def fit(
    model: nn.Module,
    loss_fn: Callable,
    train_data: tuple,
    val_data: tuple | None = None,
    *,
    # core
    epochs: int = 100,
    batch_size: int | None = None,
    seed: int = 0,
    device: str | torch.device | None = None,
    # optimizer
    optimizer: "str | torch.optim.Optimizer | Callable[..., torch.optim.Optimizer]" = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    schedule: "str | torch.optim.lr_scheduler.LRScheduler" = "constant",
    grad_clip: float = 0.0,
    nan_tolerance: int = 5,
    # validation
    val_metric: Callable[..., "float | dict[str, float]"] | None = None,
    val_every: int = 1,
    patience: int | None = None,
    val_direction: str = "min",
    best_from: str | None = None,
    # io
    out_dir: str | Path | None = None,
    save_best: bool = True,
    save_every: int | None = None,
    keep_last: int = 3,
    resume_from: str | Path | None = None,
    # misc
    verbose: bool = True,
) -> dict[str, Any]:
    """One-shot training convenience. Identical behavior to constructing a
    default :class:`Trainer` and calling :meth:`Trainer.fit`.

    Use :class:`Trainer` directly when you want to override ``train_step`` or
    ``eval_step`` — e.g., causal-survival methods (SurvITE's IPM regularizer,
    T/S-learner two-stage training, counterfactual losses).

    See :class:`Trainer` and :meth:`Trainer.fit` for parameter docs.
    """
    trainer = Trainer(
        model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        schedule=schedule,
        grad_clip=grad_clip,
        nan_tolerance=nan_tolerance,
        device=device,
        seed=seed,
    )
    return trainer.fit(
        train_data,
        val_data,
        epochs=epochs,
        batch_size=batch_size,
        val_metric=val_metric,
        val_every=val_every,
        patience=patience,
        val_direction=val_direction,
        best_from=best_from,
        out_dir=out_dir,
        save_best=save_best,
        save_every=save_every,
        keep_last=keep_last,
        resume_from=resume_from,
        verbose=verbose,
    )
