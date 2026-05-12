from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


def _make_activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"activation must be one of {list(_ACTIVATIONS)}, got {name!r}"
        )
    return _ACTIVATIONS[name]()


def _make_norm(name: str, dim: int) -> nn.Module:
    if name == "layer":
        return nn.LayerNorm(dim)
    if name == "batch":
        return nn.BatchNorm1d(dim)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"norm must be 'layer', 'batch', or 'none', got {name!r}")


class _Block(nn.Module):
    """Pre-norm residual block: norm -> linear -> activation -> dropout (+ residual)."""

    def __init__(
        self,
        dim: int,
        activation: str,
        norm: str,
        dropout: float,
        residual: bool,
    ) -> None:
        super().__init__()
        self.norm = _make_norm(norm, dim)
        self.linear = nn.Linear(dim, dim)
        self.act = _make_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.linear(h)
        h = self.act(h)
        h = self.dropout(h)
        return x + h if self.residual else h


class MLP(nn.Module):
    r"""Configurable feed-forward backbone for tabular survival models.

    Input projection -> ``n_blocks`` of (LayerNorm/BatchNorm -> Linear ->
    activation -> Dropout, optionally residual) -> output projection.
    Pre-norm + residual is the modern default. For shallow tabular survival
    nets (2-4 blocks), this is empirically near-optimal (Gorishniy et al.,
    2021).

    Parameters
    ----------
    in_features : int
    out_features : int
    hidden_dim : int, default 64
    n_blocks : int, default 2
    activation : "gelu" | "relu", default "gelu"
    norm : "layer" | "batch" | "none", default "layer"
        Use "layer" with Cox-style losses; "batch" couples unrelated samples
        across the partial likelihood and fails on small batches.
    dropout : float in [0, 1), default 0.1
    residual : bool, default True

    References
    ----------
    Gorishniy, Y. et al. (2021). Revisiting Deep Learning Models for Tabular
    Data. NeurIPS.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_dim: int = 64,
        n_blocks: int = 2,
        activation: Literal["gelu", "relu"] = "gelu",
        norm: Literal["layer", "batch", "none"] = "layer",
        dropout: float = 0.1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj: nn.Module = (
            nn.Identity()
            if in_features == hidden_dim
            else nn.Linear(in_features, hidden_dim)
        )
        self.blocks = nn.ModuleList(
            [
                _Block(hidden_dim, activation, norm, dropout, residual)
                for _ in range(n_blocks)
            ]
        )
        self.output_proj: nn.Module = (
            nn.Identity()
            if hidden_dim == out_features
            else nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)
