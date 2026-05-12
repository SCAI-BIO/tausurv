r"""Differentiable Archimedean copula base (torch, Nelsen convention).

Mirror of :mod:`tausurv.copulas` with :class:`torch.Tensor` parameters
and outputs. All operations are closed-form vectorized torch ops — no
Python loops over batch, no autograd-via-fallback for quantities that
have a closed form. Gradients flow through ``theta``.
"""

from __future__ import annotations

import torch
from torch import Tensor


class ArchimedeanCopula:
    """Differentiable Archimedean copula base class."""

    #: Allowed range for the dependence parameter ``theta``.
    theta_range: tuple[float, float] = (-float("inf"), float("inf"))

    #: Dependence parameter (a torch scalar tensor).
    theta: Tensor

    def __init__(self, theta: Tensor | float) -> None:
        if not isinstance(theta, Tensor):
            theta = torch.tensor(float(theta))
        lo, hi = self.theta_range
        if not (lo <= float(theta.detach()) <= hi):
            raise ValueError(
                f"{type(self).__name__}: theta must be in [{lo}, {hi}], "
                f"got {float(theta.detach())}"
            )
        self.theta = theta

    def phi(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def phi_inv(self, s: Tensor) -> Tensor:
        raise NotImplementedError

    def log_phi_deriv_abs(self, t: Tensor, k: int = 1) -> Tensor:
        raise NotImplementedError

    def log_phi_inv_deriv_abs(self, s: Tensor, k: int = 1) -> Tensor:
        raise NotImplementedError

    def kendalls_tau(self) -> Tensor:
        raise NotImplementedError

    def tail_dependence(self) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def cdf(self, u: Tensor) -> Tensor:
        r"""$C(u_1, \dots, u_d) = \varphi^{-1}(\sum_i \varphi(u_i))$.

        ``u`` of shape ``(..., d)`` — last axis indexes the components.
        For competing risks, pass per-cause survival values ``S_k(t|x)``
        to obtain joint survival.
        """
        if not isinstance(u, Tensor):
            u = torch.as_tensor(u, dtype=self.theta.dtype)
        return self.phi_inv(self.phi(u).sum(dim=-1))

    def log_pdf(self, u: Tensor) -> Tensor:
        r"""$\log c(u) = \log |(\varphi^{-1})^{(d)}(s)|
            + \sum_i \log |\varphi'(u_i)|$, $s = \sum_i \varphi(u_i)$.

        Bivariate ($d = 2$) closed form is always supported. For $d > 2$,
        only subclasses with a closed-form $k$-th derivative of
        $\varphi^{-1}$ (Clayton, Independence in v0.1) work; others
        raise :class:`NotImplementedError`. Single partial derivatives
        of :meth:`cdf` w.r.t. any input are available via autograd.
        """
        if not isinstance(u, Tensor):
            u = torch.as_tensor(u, dtype=self.theta.dtype)
        d = u.shape[-1]
        s = self.phi(u).sum(dim=-1)
        return self.log_phi_inv_deriv_abs(s, k=d) + self.log_phi_deriv_abs(u, k=1).sum(
            dim=-1
        )

    def pdf(self, u: Tensor) -> Tensor:
        return torch.exp(self.log_pdf(u))
