r"""Treatment-specific output heads for HTE neural models.

Given a shared latent representation $\Phi(x) \in \mathbb{R}^d$, K
independent heads $h_a : \mathbb{R}^d \to \mathbb{R}^o$ produce per-arm
outputs. For discrete-time hazard estimation, ``o = n_bins`` and the
output is interpreted as hazard logits.

The split between *encoder* (which sees all arms) and *heads* (each of
which sees only its arm's data) is the SurvITE/CFR design: a balanced
representation feeds arm-specific heads, with an IPM penalty applied at
the encoder output.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tausurv.nn.modules.mlp import MLP


class TreatmentSpecificHeads(nn.Module):
    r"""K independent MLP heads sharing a common input dimension.

    Parameters
    ----------
    n_arms : int
        Number of treatment arms (``2`` for binary).
    in_features : int
        Latent representation dimension (encoder output).
    out_features : int
        Per-arm output dimension (``n_bins`` for hazard parametrisation).
    hidden_dim, n_blocks, activation, norm, dropout, residual
        Forwarded to :class:`tausurv.nn.modules.MLP` for each head.

    Examples
    --------
    Hazard logits over 30 time bins for binary treatment::

        heads = TreatmentSpecificHeads(n_arms=2, in_features=64, out_features=30)
        phi = encoder(X)                            # (n, 64)
        logits = heads(phi)                         # list of len 2, each (n, 30)
    """

    def __init__(
        self,
        *,
        n_arms: int,
        in_features: int,
        out_features: int,
        hidden_dim: int = 100,
        n_blocks: int = 1,
        activation: str = "relu",
        norm: str = "layer",
        dropout: float = 0.1,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if n_arms < 2:
            raise ValueError(f"n_arms must be >= 2; got {n_arms}")
        self.n_arms = n_arms
        self.heads = nn.ModuleList(
            [
                MLP(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_dim=hidden_dim,
                    n_blocks=n_blocks,
                    activation=activation,  # type: ignore[arg-type]
                    norm=norm,  # type: ignore[arg-type]
                    dropout=dropout,
                    residual=residual,
                )
                for _ in range(n_arms)
            ]
        )

    def forward(self, phi: Tensor) -> list[Tensor]:
        """Return a list of K per-arm logit tensors, each ``(n, out_features)``."""
        return [head(phi) for head in self.heads]
