r"""Survival-flavored distribution base class (differentiable torch).

Mirrors :mod:`tausurv.distributions` with the same uniform survival API.
Parameters are :class:`torch.Tensor`\\s; gradients flow through every
method.

Concrete subclasses implement :meth:`log_pdf`, :meth:`log_survival`, and
:meth:`quantile`; everything else is derived. Sampling uses inverse-CDF
sampling on ``torch.rand``, which is reparameterized — gradients flow
back to the distribution parameters through :meth:`rsample`.
"""

from __future__ import annotations

import torch
from torch import Tensor


class SurvivalDistribution:
    """Base class for differentiable survival distributions."""

    def log_pdf(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def log_survival(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def quantile(self, p: Tensor) -> Tensor:
        raise NotImplementedError

    def mean(self) -> Tensor:
        raise NotImplementedError

    @property
    def _params(self) -> tuple[Tensor, ...]:
        raise NotImplementedError(
            f"{type(self).__name__} must expose a `_params` property"
        )

    def pdf(self, t: Tensor) -> Tensor:
        return torch.exp(self.log_pdf(t))

    def survival(self, t: Tensor) -> Tensor:
        return torch.exp(self.log_survival(t))

    def cdf(self, t: Tensor) -> Tensor:
        return -torch.expm1(self.log_survival(t))

    def hazard(self, t: Tensor) -> Tensor:
        return torch.exp(self.log_hazard(t))

    def log_hazard(self, t: Tensor) -> Tensor:
        return self.log_pdf(t) - self.log_survival(t)

    def median(self) -> Tensor:
        ref = self._params[0]
        return self.quantile(torch.full_like(ref, 0.5))

    def rsample(self, sample_shape: tuple[int, ...] | int = ()) -> Tensor:
        r"""Reparameterized sample via inverse-CDF on ``torch.rand``.

        Returns a tensor of shape ``sample_shape + param_shape`` — sampling
        is broadcast over the (possibly batched) distribution parameters.
        Gradients flow from the samples back into the params.
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        ref = self._params[0]
        u = torch.rand(
            tuple(sample_shape) + ref.shape, dtype=ref.dtype, device=ref.device
        )
        return self.quantile(u)

    def sample(self, sample_shape: tuple[int, ...] | int = ()) -> Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)
