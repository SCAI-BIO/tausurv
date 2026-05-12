from __future__ import annotations

import torch
from torch import Tensor

from tausurv.nn.copulas.base import ArchimedeanCopula


class Joe(ArchimedeanCopula):
    r"""Differentiable Joe copula with parameter $\theta \ge 1$.

    See :class:`tausurv.copulas.Joe` for math.
    """

    theta_range = (1.0, 50.0)

    def phi(self, t):
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t, dtype=self.theta.dtype)
        log_1m_t = torch.log1p(-t)
        return -torch.log1p(-torch.exp(self.theta * log_1m_t))

    def phi_inv(self, s):
        if not isinstance(s, Tensor):
            s = torch.as_tensor(s, dtype=self.theta.dtype)
        # Use -expm1(-s) for accurate (1 - e^{-s}) and clamp away from 0
        # so log is finite at the s=0 boundary.
        s_safe = s.clamp(min=1e-300)
        return 1.0 - torch.exp(torch.log(-torch.expm1(-s_safe)) / self.theta)

    def log_phi_deriv_abs(self, t, k=1):
        if k != 1:
            raise NotImplementedError("Joe.log_phi_deriv_abs supports k=1 only.")
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t, dtype=self.theta.dtype)
        log_1m_t = torch.log1p(-t)
        return (
            torch.log(self.theta)
            + (self.theta - 1.0) * log_1m_t
            - torch.log1p(-torch.exp(self.theta * log_1m_t))
        )

    def log_phi_inv_deriv_abs(self, s, k=1):
        if not isinstance(s, Tensor):
            s = torch.as_tensor(s, dtype=self.theta.dtype)
        log_q = torch.log1p(-torch.exp(-s))
        log_1mq = -s
        a = 1.0 / self.theta
        if k == 1:
            return (a - 1.0) * log_q + log_1mq - torch.log(self.theta)
        if k == 2:
            return (
                -torch.log(self.theta)
                + log_1mq
                + (a - 2.0) * log_q
                + torch.log1p(-a * (1.0 - torch.exp(log_q)))
            )
        raise NotImplementedError(
            f"Joe.log_phi_inv_deriv_abs supports k in {{1, 2}}; got k={k}."
        )

    def kendalls_tau(self) -> Tensor:
        if float(self.theta.detach()) == 1.0:
            return torch.zeros((), dtype=self.theta.dtype)
        theta_val = float(self.theta.detach())
        total = 0.0
        for k in range(1, 1000):
            term = 1.0 / (k * (theta_val * k + 2.0) * (theta_val * (k - 1) + 2.0))
            total += term
            if term < 1e-15:
                break
        return torch.tensor(1.0 - 4.0 * total, dtype=self.theta.dtype)

    def tail_dependence(self) -> tuple[Tensor, Tensor]:
        return (
            torch.zeros((), dtype=self.theta.dtype),
            2.0
            - torch.pow(torch.tensor(2.0, dtype=self.theta.dtype), 1.0 / self.theta),
        )
