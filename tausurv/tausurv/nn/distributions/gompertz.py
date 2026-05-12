from __future__ import annotations

import torch
from torch import Tensor

from tausurv.nn.distributions.base import SurvivalDistribution


class Gompertz(SurvivalDistribution):
    r"""Differentiable Gompertz(shape, rate).

    Hazard $h(t) = b\,e^{a t}$ — exponentially increasing.

    :meth:`mean` is not implemented for torch (the closed form involves
    the exponential integral $E_1$, which has no built-in differentiable
    torch op). For point estimates of the mean, use the numpy
    :class:`tausurv.distributions.Gompertz` with the detached parameter
    values.
    """

    def __init__(self, shape: Tensor, rate: Tensor) -> None:
        self.shape = torch.as_tensor(shape)
        self.rate = torch.as_tensor(rate)

    @property
    def _params(self) -> tuple[Tensor, ...]:
        return (self.shape, self.rate)

    def log_pdf(self, t: Tensor) -> Tensor:
        t = torch.as_tensor(t, dtype=self.rate.dtype)
        a, b = self.shape, self.rate
        log_p = torch.log(b) + a * t - (b / a) * torch.expm1(a * t)
        return torch.where(t >= 0, log_p, torch.full_like(log_p, float("-inf")))

    def log_survival(self, t: Tensor) -> Tensor:
        t = torch.as_tensor(t, dtype=self.rate.dtype)
        a, b = self.shape, self.rate
        return torch.where(t >= 0, -(b / a) * torch.expm1(a * t), torch.zeros_like(t))

    def quantile(self, p: Tensor) -> Tensor:
        p = torch.as_tensor(p, dtype=self.rate.dtype)
        a, b = self.shape, self.rate
        return torch.log1p(-(a / b) * torch.log1p(-p)) / a

    def mean(self) -> Tensor:
        raise NotImplementedError(
            "Gompertz.mean (torch) requires the exponential integral E_1, "
            "which has no differentiable torch primitive. Use "
            "tausurv.distributions.Gompertz with detached params for a "
            "numeric value."
        )
