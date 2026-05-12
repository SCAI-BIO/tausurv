from __future__ import annotations

import torch
from torch import Tensor

from tausurv.nn.copulas.base import ArchimedeanCopula


class Independence(ArchimedeanCopula):
    r"""Independence copula $C(u_1, \dots, u_d) = \prod_i u_i$."""

    theta_range = (0.0, 0.0)

    def __init__(self) -> None:
        self.theta = torch.tensor(0.0)

    def phi(self, t):
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t, dtype=torch.float64)
        return -torch.log(t)

    def phi_inv(self, s):
        if not isinstance(s, Tensor):
            s = torch.as_tensor(s, dtype=torch.float64)
        return torch.exp(-s)

    def log_phi_deriv_abs(self, t, k=1):
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t, dtype=torch.float64)
        return torch.lgamma(torch.tensor(float(k), dtype=t.dtype)) - k * torch.log(t)

    def log_phi_inv_deriv_abs(self, s, k=1):
        if not isinstance(s, Tensor):
            s = torch.as_tensor(s, dtype=torch.float64)
        return -s

    def cdf(self, u):
        if not isinstance(u, Tensor):
            u = torch.as_tensor(u, dtype=torch.float64)
        return torch.prod(u, dim=-1)

    def log_pdf(self, u):
        if not isinstance(u, Tensor):
            u = torch.as_tensor(u, dtype=torch.float64)
        return torch.zeros(u.shape[:-1], dtype=u.dtype, device=u.device)

    def kendalls_tau(self) -> Tensor:
        return torch.tensor(0.0)

    def tail_dependence(self) -> tuple[Tensor, Tensor]:
        return torch.tensor(0.0), torch.tensor(0.0)
