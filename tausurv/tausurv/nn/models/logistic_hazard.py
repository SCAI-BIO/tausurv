from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from tausurv.nn._utils import as_model_tensor
from tausurv.nn.modules.mlp import MLP
from tausurv.nn.pretrained import PretrainedMixin
from tausurv.predictor import SurvivalPredictor


@dataclass
class LogisticHazardConfig:
    """Architectural configuration for :class:`LogisticHazard`.

    Only architectural choices live here. The real-time interpretation
    of the bins is data-derived state, set post-construction via
    :meth:`LogisticHazard.set_time_grid`.
    """

    in_features: int
    n_bins: int
    hidden_dim: int = 64
    n_blocks: int = 2
    activation: Literal["gelu", "relu"] = "gelu"
    norm: Literal["layer", "batch", "none"] = "layer"
    dropout: float = 0.1
    residual: bool = True


class LogisticHazard(SurvivalPredictor, PretrainedMixin, nn.Module):
    r"""Discrete-time hazard-parameterized survival model.

    Independently introduced by Gensheimer & Narasimhan (2019) as
    "Nnet-Survival" and by Kvamme & Borgan (2021) as "LogisticHazard".
    Outputs logits over $K$ time bins; ``sigmoid(logits)`` yields per-bin
    conditional hazards
    $h_k = P(T \in \text{bin}_k \mid T \ge \text{start of bin}_k)$,
    from which survival is recovered as
    $\hat S(t_k) = \prod_{j \le k} (1 - h_j)$.

    Compared with DeepHit's PMF parameterization, the hazard form is more
    stable on small datasets and avoids the softmax-normalization
    competition between bins.

    The model's output is bin-indexed (1..``n_bins``). Set the real-time
    interpretation of those bins post-construction with
    :meth:`set_time_grid`; if unset, prediction methods raise. The grid
    round-trips through :meth:`save_pretrained` /
    :meth:`from_pretrained`.

    Pair with :func:`tausurv.nn.functional.logistic_hazard_nll` for training.

    References
    ----------
    Gensheimer, M. F., Narasimhan, B. (2019). A scalable discrete-time
    survival model for neural networks. PeerJ, 7.
    Kvamme, H., Borgan, Ø. (2021). Continuous and discrete-time survival
    prediction with neural networks. Lifetime Data Analysis, 27(4).
    """

    config_class = LogisticHazardConfig
    config: LogisticHazardConfig

    def __init__(
        self,
        config: LogisticHazardConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = LogisticHazardConfig(**kwargs)
        elif kwargs:
            raise TypeError(f"{type(self).__name__}: pass `config` OR kwargs, not both")
        self.config = config
        self.n_bins = config.n_bins
        self.backbone = MLP(
            config.in_features,
            config.n_bins,
            hidden_dim=config.hidden_dim,
            n_blocks=config.n_blocks,
            activation=config.activation,
            norm=config.norm,
            dropout=config.dropout,
            residual=config.residual,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def set_time_grid(self, times: ArrayLike) -> "LogisticHazard":
        r"""Set the real-time interpretation of the model's bins. See
        :meth:`DeepHit.set_time_grid`."""
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

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if not hasattr(self, "_times"):
            raise RuntimeError(
                f"{type(self).__name__}._survival_function requires a time "
                f"grid; call `model.set_time_grid(times)` first."
            )
        X_t = as_model_tensor(X, self)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self.forward(X_t)
            hazard = torch.sigmoid(logits)
            log_surv = torch.log1p(-hazard.clamp(max=1.0 - 1e-7)).cumsum(dim=-1)
            S_full = torch.exp(log_surv).cpu().numpy().astype(np.float64)
        if was_training:
            self.train()
        idx = np.searchsorted(self._times, times, side="right") - 1
        clipped = np.clip(idx, 0, self.n_bins - 1)
        S_at = S_full[..., clipped]
        before = idx < 0
        if before.any():
            S_at[..., before] = 1.0
        return S_at

    def save_pretrained(self, path: str | Path) -> None:
        """Persist config + weights, plus the time grid set by
        :meth:`set_time_grid` (if any)."""
        super().save_pretrained(path)
        if hasattr(self, "_times"):
            np.savez(Path(path) / "time_grid.npz", times=self._times)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "LogisticHazard":
        model: LogisticHazard = super().from_pretrained(path)  # type: ignore[assignment]
        time_grid_path = Path(path) / "time_grid.npz"
        if time_grid_path.exists():
            model._times = np.load(time_grid_path)["times"]
        return model
