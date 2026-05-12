from __future__ import annotations

import torch
from torch import Tensor

from tausurv.nn.copulas.base import ArchimedeanCopula


class Clayton(ArchimedeanCopula):
    r"""Differentiable Clayton copula with parameter $\theta > 0$.

    See :class:`tausurv.copulas.Clayton` for math. All operations are
    closed-form vectorized torch; gradients flow through ``theta``.

    Supports arbitrary $d$-variate ``log_pdf`` in closed form — the
    $k$-th derivative of $\varphi^{-1}$ has a clean factorial expression.
    """

    theta_range = (1e-8, 100.0)

    def phi(self, t):
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t, dtype=self.theta.dtype)
        return (torch.pow(t, -self.theta) - 1.0) / self.theta

    def phi_inv(self, s):
        if not isinstance(s, Tensor):
            s = torch.as_tensor(s, dtype=self.theta.dtype)
        return torch.pow(1.0 + self.theta * s, -1.0 / self.theta)

    def log_phi_deriv_abs(self, t, k=1):
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t, dtype=self.theta.dtype)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        # log |phi^(k)(t)| = sum_{j=1}^{k-1} log(theta + j) - (theta + k) log(t)
        if k == 1:
            log_factor = torch.zeros((), dtype=t.dtype, device=t.device)
        else:
            js = torch.arange(1, k, dtype=t.dtype, device=t.device)
            log_factor = torch.log(self.theta + js).sum()
        return log_factor - (self.theta + k) * torch.log(t)

    def log_phi_inv_deriv_abs(self, s, k=1):
        if not isinstance(s, Tensor):
            s = torch.as_tensor(s, dtype=self.theta.dtype)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        # log |(phi_inv)^(k)(s)| = sum_{j=0}^{k-1} log(1 + j*theta) - (1/theta + k) log(1 + theta s)
        if k == 1:
            log_factor = torch.zeros((), dtype=s.dtype, device=s.device)
        else:
            js = torch.arange(k, dtype=s.dtype, device=s.device)
            log_factor = torch.log1p(self.theta * js).sum()
        return log_factor - (1.0 / self.theta + k) * torch.log1p(self.theta * s)

    def kendalls_tau(self) -> Tensor:
        return self.theta / (self.theta + 2.0)

    def tail_dependence(self) -> tuple[Tensor, Tensor]:
        return (
            torch.pow(torch.tensor(2.0, dtype=self.theta.dtype), -1.0 / self.theta),
            torch.zeros((), dtype=self.theta.dtype),
        )
