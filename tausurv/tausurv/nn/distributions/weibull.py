from __future__ import annotations

import torch
from torch import Tensor

from tausurv.nn.distributions.base import SurvivalDistribution


class Weibull(SurvivalDistribution):
    r"""Differentiable Weibull(shape, scale).

    See :class:`tausurv.distributions.Weibull` for the math. Parameters
    must be broadcastable :class:`torch.Tensor`\\s; gradients flow through
    every method.
    """

    def __init__(self, shape: Tensor, scale: Tensor) -> None:
        self.shape = torch.as_tensor(shape)
        self.scale = torch.as_tensor(scale)

    @property
    def _params(self) -> tuple[Tensor, ...]:
        return (self.shape, self.scale)

    def log_pdf(self, t: Tensor) -> Tensor:
        t = torch.as_tensor(t, dtype=self.scale.dtype)
        k, lam = self.shape, self.scale
        log_t = torch.log(t.clamp(min=torch.finfo(t.dtype).tiny))
        log_p = (
            torch.log(k)
            - torch.log(lam)
            + (k - 1.0) * (log_t - torch.log(lam))
            - (t / lam) ** k
        )
        return torch.where(t > 0, log_p, torch.full_like(log_p, float("-inf")))

    def log_survival(self, t: Tensor) -> Tensor:
        t = torch.as_tensor(t, dtype=self.scale.dtype)
        return torch.where(
            t > 0,
            -((t / self.scale) ** self.shape),
            torch.zeros_like(t),
        )

    def quantile(self, p: Tensor) -> Tensor:
        p = torch.as_tensor(p, dtype=self.scale.dtype)
        return self.scale * (-torch.log1p(-p)) ** (1.0 / self.shape)

    def mean(self) -> Tensor:
        return self.scale * torch.exp(torch.lgamma(1.0 + 1.0 / self.shape))
