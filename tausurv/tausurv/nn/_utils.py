"""Shared helpers for NN models. Private to ``tausurv.nn``."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor, nn

from tausurv.discretization import time_grid


def as_model_tensor(X: ArrayLike, model: nn.Module) -> Tensor:
    """Convert ``X`` to a tensor on the model's device with float32 dtype.

    Pass-through for tensors (only the device is matched); ndarrays /
    array-likes are coerced via :func:`torch.as_tensor`.
    """
    if isinstance(X, Tensor):
        t = X
    else:
        t = torch.as_tensor(np.asarray(X, dtype=np.float32))
    params = list(model.parameters())
    return t.to(params[0].device) if params else t


def as_training_tensors(
    model: nn.Module,
    X: ArrayLike,
    event_time: ArrayLike,
    event_indicator: ArrayLike,
) -> tuple[Tensor, Tensor, Tensor]:
    """``(X, event_time, event_indicator)`` as float32 tensors on the model's device.

    The losses cast the indicator to whatever they need, so one dtype for
    all three keeps polars frames, integer arrays and tensors interchangeable.
    """
    return (
        as_model_tensor(X, model),
        as_model_tensor(event_time, model),
        as_model_tensor(event_indicator, model),
    )


class DiscreteTimeModel(Protocol):
    """A bin-indexed model whose grid is set after construction."""

    n_bins: int

    def set_time_grid(self, times: ArrayLike) -> object: ...


def fit_time_grid(
    model: DiscreteTimeModel, event_time: ArrayLike, event_indicator: ArrayLike
) -> None:
    """Set a discrete-time model's grid from the training data unless already set.

    Tied quantiles can collapse the grid below ``n_bins``, which is an
    architecture mismatch the caller has to resolve.
    """
    if hasattr(model, "_times"):
        return
    grid = time_grid(event_time, event_indicator, n_bins=model.n_bins)
    if len(grid) != model.n_bins:
        raise ValueError(
            f"time_grid produced {len(grid)} edges for n_bins={model.n_bins}; "
            f"construct the model with n_bins={len(grid)} or call "
            f"set_time_grid() with a grid of length {model.n_bins}"
        )
    model.set_time_grid(grid)


def reset_parameters(model: nn.Module, seed: int) -> None:
    """Re-initialise every submodule that defines ``reset_parameters``.

    Seeds torch first, so a model fitted twice with the same seed trains from
    identical weights. Called at the start of the models' ``fit``.
    """
    torch.manual_seed(seed)
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if module is not model and callable(reset):
            reset()
