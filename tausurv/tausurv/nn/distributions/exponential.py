from __future__ import annotations

import torch
from torch import Tensor

from tausurv.nn.distributions.base import SurvivalDistribution


class Exponential(SurvivalDistribution):
    r"""Differentiable Exponential(rate).

    See :class:`tausurv.distributions.Exponential` for the math.
    """

    def __init__(self, rate: Tensor) -> None:
        self.rate = torch.as_tensor(rate)

    @property
    def _params(self) -> tuple[Tensor, ...]:
        return (self.rate,)

    def log_pdf(self, t: Tensor) -> Tensor:
        t = torch.as_tensor(t, dtype=self.rate.dtype)
        return torch.where(
            t >= 0,
            torch.log(self.rate) - self.rate * t,
            torch.full_like(t, float("-inf")),
        )

    def log_survival(self, t: Tensor) -> Tensor:
        t = torch.as_tensor(t, dtype=self.rate.dtype)
        return torch.where(t >= 0, -self.rate * t, torch.zeros_like(t))

    def quantile(self, p: Tensor) -> Tensor:
        p = torch.as_tensor(p, dtype=self.rate.dtype)
        return -torch.log1p(-p) / self.rate

    def mean(self) -> Tensor:
        return 1.0 / self.rate
