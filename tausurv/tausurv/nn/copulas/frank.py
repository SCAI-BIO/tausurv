from __future__ import annotations

import torch
from torch import Tensor

from tausurv.nn.copulas.base import ArchimedeanCopula


class Frank(ArchimedeanCopula):
    r"""Differentiable Frank copula with $\theta \in \mathbb{R} \setminus \{0\}$.

    See :class:`tausurv.copulas.Frank` for math.
    """

    theta_range = (-50.0, 50.0)

    def __init__(self, theta) -> None:
        super().__init__(theta)
        if float(self.theta.detach()) == 0.0:
            raise ValueError("Frank theta cannot be 0 (use Independence)")

    def phi(self, t):
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t, dtype=self.theta.dtype)
        return -torch.log(torch.expm1(-self.theta * t) / torch.expm1(-self.theta))

    def phi_inv(self, s):
        if not isinstance(s, Tensor):
            s = torch.as_tensor(s, dtype=self.theta.dtype)
        return -torch.log1p(torch.exp(-s) * torch.expm1(-self.theta)) / self.theta

    def log_phi_deriv_abs(self, t, k=1):
        if k != 1:
            raise NotImplementedError("Frank.log_phi_deriv_abs supports k=1 only.")
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t, dtype=self.theta.dtype)
        return torch.log(self.theta.abs()) - torch.log(
            torch.expm1(self.theta * t).abs()
        )

    def log_phi_inv_deriv_abs(self, s, k=1):
        if not isinstance(s, Tensor):
            s = torch.as_tensor(s, dtype=self.theta.dtype)
        q = -torch.expm1(-self.theta)
        log_abs_q = torch.log(q.abs())
        log_r = log_abs_q - s
        log_one_minus_r = torch.log1p(-q * torch.exp(-s))
        if k == 1:
            return log_r - torch.log(self.theta.abs()) - log_one_minus_r
        if k == 2:
            return log_r - torch.log(self.theta.abs()) - 2.0 * log_one_minus_r
        raise NotImplementedError(
            f"Frank.log_phi_inv_deriv_abs supports k in {{1, 2}}; got k={k}."
        )

    def kendalls_tau(self) -> Tensor:
        # Numerical Debye function; not differentiable through theta.
        from scipy.integrate import quad

        def integrand(x):
            import math as _m

            return x / (_m.expm1(x) if x != 0 else 1.0)

        theta_val = float(self.theta.detach())
        if theta_val == 0:
            return torch.tensor(0.0, dtype=self.theta.dtype)
        debye_1, _ = quad(integrand, 0.0, abs(theta_val), limit=200)
        debye_1 /= abs(theta_val)
        return torch.tensor(
            1.0 - 4.0 / theta_val * (1.0 - debye_1), dtype=self.theta.dtype
        )

    def tail_dependence(self) -> tuple[Tensor, Tensor]:
        return torch.zeros((), dtype=self.theta.dtype), torch.zeros(
            (), dtype=self.theta.dtype
        )
