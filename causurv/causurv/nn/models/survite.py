r"""SurvITE neural-network module: shared encoder + treatment-specific heads.

This file defines the *architecture* (a :class:`torch.nn.Module`) — the
glue that makes it a fittable learner satisfying :class:`HTEPredictor`
lives in :mod:`causurv.learners.survite`. The split mirrors the
tausurv pattern (:mod:`tausurv.nn.models` for nn.Modules, paired with
fit/predict wrappers elsewhere).

Architecture (Curth et al., 2021, "SurvITE"):

- An :class:`MLP` encoder maps $X \in \mathbb{R}^d$ to a balanced
  representation $\Phi(X) \in \mathbb{R}^r$.
- :class:`TreatmentSpecificHeads` — one MLP head per arm — maps $\Phi$
  to per-bin hazard logits $\lambda_a(t_k \mid x)$ for $k = 1, \ldots,
  n_{\text{bins}}$.
- The training loss combines a discrete-time hazard NLL on the
  observed arm with an IPM penalty on $\Phi$; see
  :class:`causurv.nn.losses.SurvITELoss`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from causurv.nn.modules.treatment_heads import TreatmentSpecificHeads
from tausurv.nn.modules.mlp import MLP
from tausurv.nn.training.trainer import Trainer


@dataclass
class SurvITEConfig:
    """Architectural configuration for :class:`SurvITEModule`."""

    in_features: int
    n_bins: int
    n_arms: int = 2
    repr_dim: int = 100
    encoder_hidden: int = 100
    encoder_blocks: int = 2
    heads_hidden: int = 100
    heads_blocks: int = 1
    activation: Literal["gelu", "relu"] = "relu"
    norm: Literal["layer", "batch", "none"] = "layer"
    dropout: float = 0.1


class SurvITEModule(nn.Module):
    r"""SurvITE network: shared encoder feeding K treatment-specific heads.

    Parameters
    ----------
    config : :class:`SurvITEConfig`

    Returns
    -------
    ``forward(X)`` returns ``(phi, logits_list)`` where

    - ``phi`` is the encoder output, ``(n, repr_dim)``.
    - ``logits_list`` is a list of ``n_arms`` tensors, each ``(n, n_bins)``,
      interpreted as discrete-hazard logits for that arm.

    The encoder output is returned so that the loss can apply the IPM
    penalty without re-running the encoder.
    """

    config_class = SurvITEConfig
    config: SurvITEConfig

    def __init__(self, config: SurvITEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = MLP(
            in_features=config.in_features,
            out_features=config.repr_dim,
            hidden_dim=config.encoder_hidden,
            n_blocks=config.encoder_blocks,
            activation=config.activation,
            norm=config.norm,
            dropout=config.dropout,
        )
        self.heads = TreatmentSpecificHeads(
            n_arms=config.n_arms,
            in_features=config.repr_dim,
            out_features=config.n_bins,
            hidden_dim=config.heads_hidden,
            n_blocks=config.heads_blocks,
            activation=config.activation,
            norm=config.norm,
            dropout=config.dropout,
        )

    def forward(self, X: Tensor) -> tuple[Tensor, list[Tensor]]:
        phi = self.encoder(X)
        logits_list = self.heads(phi)
        return phi, logits_list

    @torch.no_grad()
    def hazard(self, X: Tensor) -> list[Tensor]:
        """Per-arm discrete hazards on the natural bin grid, shape ``(n, n_bins)``."""
        was_training = self.training
        self.eval()
        _, logits = self.forward(X)
        if was_training:
            self.train()
        return [torch.sigmoid(z) for z in logits]


class SurvITETrainer(Trainer):
    r"""Trainer for :class:`SurvITEModule`.

    Overrides :meth:`train_step` because SurvITE's forward returns
    ``(phi, logits_list)`` (representation + per-arm logits) and its
    loss consumes both alongside ``(event_time, event_indicator,
    treatment)``. The default :meth:`Trainer.train_step` assumes a
    single tensor prediction and a ``loss_fn(predictions, *targets)``
    signature, which doesn't fit SurvITE's multi-output forward.
    """

    def train_step(self, batch: tuple[Tensor, ...]) -> Tensor:
        self.model.train()
        X, T, E, A = batch
        self.optimizer.zero_grad(set_to_none=True)
        phi, logits = self.model(X)
        out = self.loss_fn(phi, logits, T, E, A)
        if not torch.isfinite(out.total):
            return out.total
        out.total.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return out.total
