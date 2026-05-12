from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.special import log_ndtr, ndtri

from tausurv.nn.distributions.base import SurvivalDistribution


class LogNormal(SurvivalDistribution):
    r"""Differentiable LogNormal(mu, sigma) with $\log T \sim \mathcal{N}(\mu, \sigma^2)$.

    Uses :func:`torch.special.log_ndtr` for a stable upper-tail log-survival
    and :func:`torch.special.ndtri` for the quantile. Both are
    differentiable in their inputs.
    """

    def __init__(self, mu: Tensor, sigma: Tensor) -> None:
        self.mu = torch.as_tensor(mu)
        self.sigma = torch.as_tensor(sigma)

    @property
    def _params(self) -> tuple[Tensor, ...]:
        return (self.mu, self.sigma)

    def log_pdf(self, t: Tensor) -> Tensor:
        t = torch.as_tensor(t, dtype=self.sigma.dtype)
        log_t = torch.log(t.clamp(min=torch.finfo(t.dtype).tiny))
        z = (log_t - self.mu) / self.sigma
        log_p = (
            -log_t - torch.log(self.sigma) - 0.5 * math.log(2 * math.pi) - 0.5 * z**2
        )
        return torch.where(t > 0, log_p, torch.full_like(log_p, float("-inf")))

    def log_survival(self, t: Tensor) -> Tensor:
        t = torch.as_tensor(t, dtype=self.sigma.dtype)
        log_t = torch.log(t.clamp(min=torch.finfo(t.dtype).tiny))
        z = (log_t - self.mu) / self.sigma
        return torch.where(t > 0, log_ndtr(-z), torch.zeros_like(t))

    def quantile(self, p: Tensor) -> Tensor:
        p = torch.as_tensor(p, dtype=self.sigma.dtype)
        return torch.exp(self.mu + self.sigma * ndtri(p))

    def mean(self) -> Tensor:
        return torch.exp(self.mu + 0.5 * self.sigma**2)
