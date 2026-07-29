from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from tausurv.nn._utils import as_model_tensor
from tausurv.nn.modules.mlp import MLP
from tausurv.nn.pretrained import PretrainedMixin
from tausurv.predictor import CompetingRisksPredictor


@dataclass
class DeepHitConfig:
    """Architectural configuration for :class:`DeepHit`.

    Only architectural choices live here. The real-time interpretation
    of the bins is data-derived state and is set post-construction via
    :meth:`DeepHit.set_time_grid`.
    """

    in_features: int
    n_bins: int
    n_causes: int = 1
    hidden_dim: int = 64
    n_blocks: int = 2
    activation: Literal["gelu", "relu"] = "gelu"
    norm: Literal["layer", "batch", "none"] = "layer"
    dropout: float = 0.1
    residual: bool = True


class DeepHit(CompetingRisksPredictor, PretrainedMixin, nn.Module):
    r"""Discrete-time PMF survival model (Lee et al., 2018).

    Predicts logits over $K$ time bins for each of $K_{\text{causes}}$ event
    causes. The output shape is always ``(n, n_causes, n_bins)`` — default
    ``n_causes=1`` gives single-event survival; ``n_causes >= 2`` gives the
    competing-risks head from the original DeepHit-CR paper.

    A joint softmax over the flattened ``(n_causes * n_bins)`` dimension
    converts logits to a PMF over (cause, bin) pairs that sums to 1 per
    subject. Cause-specific CIFs are recovered as a cumulative sum along
    the bin axis; survival is one minus the marginalized CIF.

    The model's output is bin-indexed (1..``n_bins``). Set the real-time
    interpretation of those bins post-construction with
    :meth:`set_time_grid`; if unset, prediction methods raise. The grid
    round-trips through :meth:`save_pretrained` /
    :meth:`from_pretrained`.

    Pair with :func:`tausurv.nn.functional.pmf_nll` and optionally
    :func:`tausurv.nn.functional.deephit_ranking` for training.

    References
    ----------
    Lee, C., Zame, W. R., Yoon, J., van der Schaar, M. (2018). DeepHit: A
    deep learning approach to survival analysis with competing risks. AAAI.
    """

    config_class = DeepHitConfig
    config: DeepHitConfig
    n_causes: int

    def __init__(
        self,
        config: DeepHitConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = DeepHitConfig(**kwargs)
        elif kwargs:
            raise TypeError(f"{type(self).__name__}: pass `config` OR kwargs, not both")
        if config.n_causes < 1:
            raise ValueError(f"n_causes must be >= 1, got {config.n_causes}")

        self.config = config
        self.n_bins = config.n_bins
        self.n_causes = config.n_causes
        self.backbone = MLP(
            config.in_features,
            config.n_causes * config.n_bins,
            hidden_dim=config.hidden_dim,
            n_blocks=config.n_blocks,
            activation=config.activation,
            norm=config.norm,
            dropout=config.dropout,
            residual=config.residual,
        )

    def forward(self, x: Tensor) -> Tensor:
        logits_flat = self.backbone(x)
        return logits_flat.reshape(-1, self.n_causes, self.n_bins)

    def set_time_grid(self, times: ArrayLike) -> "DeepHit":
        r"""Set the real-time interpretation of the model's bins.

        ``times`` must have length ``n_bins`` and gives the real-time
        upper-bound of each discrete bin — the same array passed as
        ``time_bins`` to the training loss, typically built with
        :func:`tausurv.discretization.time_grid`. Stored as ``self.times_`` and
        used by :class:`SurvivalPredictor` predict methods. Round-trips
        through :meth:`save_pretrained` / :meth:`from_pretrained`.
        """
        times = np.asarray(times, dtype=np.float64)
        if times.shape != (self.n_bins,):
            raise ValueError(
                f"times must have shape ({self.n_bins},), got {times.shape}"
            )
        self._times = times
        return self

    @property
    def times_(self) -> NDArray[np.float64]:  # type: ignore[override]
        if not hasattr(self, "_times"):
            raise RuntimeError(
                f"{type(self).__name__}.times_ is not set; call "
                f"`model.set_time_grid(times)` first or pass `times=` "
                f"explicitly to predict methods."
            )
        return self._times

    def _cif(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if not hasattr(self, "_times"):
            raise RuntimeError(
                f"{type(self).__name__}._cif requires a time grid; "
                f"call `model.set_time_grid(times)` first."
            )
        X_t = as_model_tensor(X, self)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self.forward(X_t)
            n = logits.size(0)
            pmf = F.softmax(logits.reshape(n, -1), dim=-1).reshape(
                n, self.n_causes, self.n_bins
            )
            cif_full = pmf.cumsum(dim=-1).cpu().numpy().astype(np.float64)
        if was_training:
            self.train()
        # Right-continuous step lookup onto the requested grid. Before the
        # first bin upper-bound CIF is 0.
        idx = np.searchsorted(self._times, times, side="right") - 1
        clipped = np.clip(idx, 0, self.n_bins - 1)
        cif_at = cif_full[..., clipped]
        zero = idx < 0
        if zero.any():
            cif_at[..., zero] = 0.0
        return cif_at

    def save_pretrained(self, path: str | Path) -> None:
        """Persist config + weights, plus the time grid set by
        :meth:`set_time_grid` (if any)."""
        super().save_pretrained(path)
        if hasattr(self, "_times"):
            np.savez(Path(path) / "time_grid.npz", times=self._times)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "DeepHit":
        model: DeepHit = super().from_pretrained(path)  # type: ignore[assignment]
        time_grid_path = Path(path) / "time_grid.npz"
        if time_grid_path.exists():
            model._times = np.load(time_grid_path)["times"]
        return model
