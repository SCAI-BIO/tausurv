from __future__ import annotations

from collections.abc import Callable
from typing import Iterable

import torch
from torch import nn
from torch.nn import Parameter


def build_optimizer(
    model: nn.Module,
    spec: str
    | torch.optim.Optimizer
    | Callable[[Iterable[Parameter]], torch.optim.Optimizer],
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Resolve ``spec`` into a ``torch.optim.Optimizer`` bound to ``model``.

    Parameters
    ----------
    model : nn.Module
        Source of parameters when ``spec`` is a string or factory callable.
    spec
        One of:

        - ``str`` — one of ``{"adamw", "adam", "sgd"}``. Built using ``lr`` and
          ``weight_decay``.
        - ``torch.optim.Optimizer`` — used as-is (the caller already attached
          parameters; ``lr`` and ``weight_decay`` are ignored).
        - ``Callable[[Iterable[Parameter]], Optimizer]`` — factory invoked with
          ``model.parameters()``; ``lr`` and ``weight_decay`` are ignored.
    lr, weight_decay : float
        Only consulted for the string form.
    """
    if isinstance(spec, str):
        params = model.parameters()
        if spec == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if spec == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if spec == "sgd":
            return torch.optim.SGD(
                params, lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        raise ValueError(
            f"optimizer string must be 'adamw', 'adam', or 'sgd', got {spec!r}"
        )
    if isinstance(spec, torch.optim.Optimizer):
        return spec
    if callable(spec):
        opt = spec(model.parameters())
        if not isinstance(opt, torch.optim.Optimizer):
            raise TypeError(
                f"optimizer factory must return torch.optim.Optimizer, got {type(opt).__name__}"
            )
        return opt
    raise TypeError(
        "optimizer must be a string ('adamw'|'adam'|'sgd'), a torch.optim.Optimizer "
        f"instance, or a callable factory; got {type(spec).__name__}"
    )
