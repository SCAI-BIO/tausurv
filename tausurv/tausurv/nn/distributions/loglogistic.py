from __future__ import annotations

import math

import torch
from torch import Tensor

from tausurv.nn.distributions.base import SurvivalDistribution


class LogLogistic(SurvivalDistribution):
    r"""Differentiable LogLogistic(shape, scale) (Fisk distribution).

    Closed-form everything; ``log_survival`` uses ``log1p(exp(...))`` form
    for stability in the lower-tail.
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
        z_log = log_t - torch.log(lam)
        log_p = (
            torch.log(k)
            - torch.log(lam)
            + (k - 1.0) * z_log
            - 2.0 * torch.nn.functional.softplus(k * z_log)
        )
        return torch.where(t > 0, log_p, torch.full_like(log_p, float("-inf")))

    def log_survival(self, t: Tensor) -> Tensor:
        t = torch.as_tensor(t, dtype=self.scale.dtype)
        k, lam = self.shape, self.scale
        log_t = torch.log(t.clamp(min=torch.finfo(t.dtype).tiny))
        log_s = -torch.nn.functional.softplus(k * (log_t - torch.log(lam)))
        return torch.where(t > 0, log_s, torch.zeros_like(t))

    def quantile(self, p: Tensor) -> Tensor:
        p = torch.as_tensor(p, dtype=self.scale.dtype)
        return self.scale * (p / (1.0 - p)) ** (1.0 / self.shape)

    def mean(self) -> Tensor:
        # E[T] = lambda * pi / (k * sin(pi/k)), undefined for k <= 1.
        k, lam = self.shape, self.scale
        return lam * math.pi / (k * torch.sin(math.pi / k))
