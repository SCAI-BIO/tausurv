r"""Monotone MLP — neural network whose output is monotone non-decreasing
in every input.

The two pieces:

- :class:`PositiveLinear` — a ``nn.Linear`` whose weight matrix is
  constrained to be non-negative entry-wise by the reparameterization
  $W_{ij} = w_{ij}^2$. Free parameters $w$ are unconstrained, gradients
  flow normally, but the *effective* weights are always $\\ge 0$.
- :class:`MonotoneMLP` — a stack of :class:`PositiveLinear` layers
  separated by a monotone activation (default ``tanh``). The composition
  of non-negative-weight linear maps with a monotone non-decreasing
  activation is itself monotone non-decreasing in each input
  (Daniels & Velikova, 2010; Wehenkel & Louppe, 2019).

A common use is to feed ``(features, t)`` in and obtain a real-valued
output that is monotone non-decreasing in ``t``; wrapping the output as
$\\sigma(\\cdot)$ or $1 - \\sigma(\\cdot)$ gives bounded CDF- or
survival-flavored outputs.

Trainability note: the gradient through a :class:`PositiveLinear` layer
scales with the raw parameter $w$, so stacking many wide layers makes
training extremely slow. Keep the monotone part narrow (e.g.,
``hidden_features = [64, 64, 64]``) and use an unconstrained encoder for
high-dimensional features.

References
----------
Daniels, H., Velikova, M. (2010). Monotone and partially monotone neural
networks. IEEE Transactions on Neural Networks, 21(6).
Wehenkel, A., Louppe, G. (2019). Unconstrained monotonic neural networks.
NeurIPS.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PositiveLinear(nn.Module):
    r"""Linear layer with non-negative weights via $W = w^2$.

    Mathematically equivalent to ``nn.Linear`` plus a constraint $W \\ge 0$,
    enforced implicitly by squaring the free parameter. Bias is
    unconstrained (signs of bias do not affect monotonicity of the
    output w.r.t. inputs).
    """

    in_features: int
    out_features: int
    weight_param: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_param = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_param)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_param)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        # Transform so that ``param ** 2`` is roughly the Xavier-initialized
        # magnitude — otherwise effective weights would start near zero.
        with torch.no_grad():
            self.weight_param.abs_().sqrt_()

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight_param**2, self.bias)


class MonotoneMLP(nn.Module):
    r"""MLP whose output is monotone non-decreasing in every input.

    Architecture: ``PositiveLinear → activation → PositiveLinear → activation
    → ... → PositiveLinear``. Each :class:`PositiveLinear` enforces
    non-negative weights, and the activation must itself be monotone
    non-decreasing (``tanh`` and ``sigmoid`` both qualify); the composition
    is monotone non-decreasing in each input.

    Parameters
    ----------
    in_features : int
    out_features : int, default 1
    hidden_features : sequence of int, default (64, 64, 64)
        Width of each hidden layer. Keep these narrow — the
        squared-weight reparameterization makes wide layers very slow
        to train.
    activation : {"tanh", "sigmoid"}, default "tanh"
        Monotone non-decreasing activation between hidden layers.
    bias : bool, default True

    Examples
    --------
    >>> net = MonotoneMLP(in_features=5, out_features=1)
    >>> y = net(torch.randn(8, 5))
    >>> y.shape
    torch.Size([8, 1])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        *,
        hidden_features: tuple[int, ...] = (64, 64, 64),
        activation: Literal["tanh", "sigmoid"] = "tanh",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if activation == "tanh":
            act_cls: type[nn.Module] = nn.Tanh
        elif activation == "sigmoid":
            act_cls = nn.Sigmoid
        else:
            raise ValueError(
                f"activation must be 'tanh' or 'sigmoid', got {activation!r}"
            )

        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_features:
            layers.append(PositiveLinear(prev, h, bias=bias))
            layers.append(act_cls())
            prev = h
        layers.append(PositiveLinear(prev, out_features, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
