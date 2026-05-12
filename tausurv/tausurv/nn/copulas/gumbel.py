from __future__ import annotations

import torch
from torch import Tensor

from tausurv.nn.copulas.base import ArchimedeanCopula


class Gumbel(ArchimedeanCopula):
    r"""Differentiable Gumbel copula with parameter $\theta \ge 1$.

    See :class:`tausurv.copulas.Gumbel` for math. Closed-form bivariate
    ``log_pdf`` (k=2). For d > 2 raise (Stirling-number expansion not
    implemented in v0.1).
    """

    theta_range = (1.0, 100.0)

    def phi(self, t):
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t, dtype=self.theta.dtype)
        return torch.pow(-torch.log(t), self.theta)

    def phi_inv(self, s):
        if not isinstance(s, Tensor):
            s = torch.as_tensor(s, dtype=self.theta.dtype)
        return torch.exp(-torch.pow(s, 1.0 / self.theta))

    def log_phi_deriv_abs(self, t, k=1):
        if k != 1:
            raise NotImplementedError(
                "Gumbel.log_phi_deriv_abs is implemented for k=1 only."
            )
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t, dtype=self.theta.dtype)
        return (
            torch.log(self.theta)
            + (self.theta - 1.0) * torch.log(-torch.log(t))
            - torch.log(t)
        )

    def log_phi_inv_deriv_abs(self, s, k=1):
        if not isinstance(s, Tensor):
            s = torch.as_tensor(s, dtype=self.theta.dtype)
        log_s = torch.log(s)
        w = torch.pow(s, 1.0 / self.theta)
        if k == 1:
            return -w - torch.log(self.theta) + (1.0 / self.theta - 1.0) * log_s
        if k == 2:
            return (
                -w
                + (1.0 / self.theta - 2.0) * log_s
                - 2.0 * torch.log(self.theta)
                + torch.log(w + self.theta - 1.0)
            )
        raise NotImplementedError(
            f"Gumbel.log_phi_inv_deriv_abs supports k in {{1, 2}}; got k={k}."
        )

    def kendalls_tau(self) -> Tensor:
        return 1.0 - 1.0 / self.theta

    def tail_dependence(self) -> tuple[Tensor, Tensor]:
        return (
            torch.zeros((), dtype=self.theta.dtype),
            2.0
            - torch.pow(torch.tensor(2.0, dtype=self.theta.dtype), 1.0 / self.theta),
        )
