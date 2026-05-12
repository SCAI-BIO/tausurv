"""Shared helpers for NN models. Private to ``tausurv.nn``."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor, nn


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
