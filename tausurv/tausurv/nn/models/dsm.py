r"""Deep Survival Machines (Nagpal et al., 2021).

A neural net predicts a per-subject mixture of $K$ parametric
distributions (Weibull or LogNormal). The marginal survival is a
weighted sum of the per-component survivals:

$$
\hat S(t \mid x) = \sum_{k=1}^K w_k(x)\, S_k\big(t;\, \theta_k(x)\big).
$$

DSM is the parametric counterpart to DeepHit: DeepHit puts a PMF over
discrete bins, DSM uses a mixture of continuous parametric families.
Everything (density, survival, prediction) is closed-form — no
autograd-through-$t$ or Newton inversion needed.

The model outputs a flat ``(n, 3K)`` raw tensor: the first $K$ columns
are raw shape (Weibull) / location (LogNormal), next $K$ are raw scale
/ standard deviation, last $K$ are mixture-weight logits. Constraints
(softplus for positivity, softmax for weights) are applied by the loss
and the predict methods, so the model's ``forward`` stays a clean
``nn.Linear`` output.

Pair with :func:`tausurv.nn.functional.dsm_nll` or
:class:`tausurv.nn.losses.DSMLoss` for training. Standard
:class:`tausurv.nn.Trainer` works — DSM doesn't need a Trainer subclass.

References
----------
Nagpal, C., Li, X., Dubrawski, A. (2021). Deep Survival Machines:
Fully Parametric Survival Regression and Representation Learning for
Censored Data with Competing Risks. IEEE JBHI 25(8).
"""

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
from tausurv.nn.distributions import LogNormal, Weibull
from tausurv.nn.modules import MLP
from tausurv.nn.pretrained import PretrainedMixin
from tausurv.predictor import SurvivalPredictor


@dataclass
class DSMConfig:
    """Architectural configuration for :class:`DSM`."""

    in_features: int
    n_components: int = 4
    distribution_family: Literal["weibull", "lognormal"] = "weibull"
    hidden_features: tuple[int, ...] = (64, 64)
    activation: Literal["gelu", "relu"] = "gelu"
    norm: Literal["layer", "batch", "none"] = "layer"
    dropout: float = 0.1
    residual: bool = True

    def __post_init__(self):
        if isinstance(self.hidden_features, list):
            self.hidden_features = tuple(self.hidden_features)


class DSM(SurvivalPredictor, PretrainedMixin, nn.Module):
    r"""Deep Survival Machines model.

    Construct with :class:`DSMConfig` or kwargs (HF-style). After
    training (via the standard :class:`tausurv.nn.Trainer` with
    :func:`tausurv.nn.functional.dsm_nll`), inherits the unified
    prediction API from :class:`SurvivalPredictor`.

    Real-time interpretation of the predict grid is data-derived state;
    set via :meth:`set_time_grid` post-fit or pass ``times=`` explicitly
    to predict methods.
    """

    config_class = DSMConfig
    config: DSMConfig

    def __init__(
        self,
        config: DSMConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = DSMConfig(**kwargs)
        elif kwargs:
            raise TypeError(
                f"{type(self).__name__}: pass `config` OR kwargs, not both"
            )
        if config.n_components < 1:
            raise ValueError(
                f"n_components must be >= 1, got {config.n_components}"
            )
        if config.distribution_family not in {"weibull", "lognormal"}:
            raise ValueError(
                "distribution_family must be 'weibull' or 'lognormal', "
                f"got {config.distribution_family!r}"
            )
        self.config = config

        h_dim = config.hidden_features[-1]
        self.encoder = MLP(
            in_features=config.in_features,
            out_features=h_dim,
            hidden_dim=h_dim,
            n_blocks=len(config.hidden_features),
            activation=config.activation,
            norm=config.norm,
            dropout=config.dropout,
            residual=config.residual,
        )
        # 3 heads: shape/location, scale/sigma, weight logits — each of size K.
        self.head = nn.Linear(h_dim, 3 * config.n_components)

    def forward(self, x: Tensor) -> Tensor:
        """Raw ``(n, 3K)`` output. Constraints (softplus/softmax) are
        applied by the loss and predict methods."""
        return self.head(self.encoder(x))

    def _constrained_params(
        self, raw: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply per-distribution constraints to the raw forward output.

        Returns ``(p1, p2, weights)``. For Weibull: ``p1`` = shape > 0,
        ``p2`` = scale > 0. For LogNormal: ``p1`` = mu (unconstrained),
        ``p2`` = sigma > 0. ``weights`` sum to 1 over the last axis.
        """
        K = self.config.n_components
        eps = 1e-4
        raw_p1 = raw[..., :K]
        raw_p2 = raw[..., K : 2 * K]
        raw_w = raw[..., 2 * K :]
        if self.config.distribution_family == "weibull":
            p1 = F.softplus(raw_p1) + eps
            p2 = F.softplus(raw_p2) + eps
        else:  # lognormal
            p1 = raw_p1
            p2 = F.softplus(raw_p2) + eps
        weights = F.softmax(raw_w, dim=-1)
        return p1, p2, weights

    def _make_distribution(self, p1: Tensor, p2: Tensor):
        if self.config.distribution_family == "weibull":
            return Weibull(p1, p2)
        return LogNormal(p1, p2)

    def set_time_grid(self, times: ArrayLike) -> "DSM":
        """Set the default time grid for the predict API when ``times=None``."""
        self._times = np.asarray(times, dtype=np.float64)
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
        X_t = as_model_tensor(X, self)
        times_t = torch.as_tensor(times, dtype=X_t.dtype, device=X_t.device)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            raw = self.forward(X_t)
            p1, p2, w = self._constrained_params(raw)         # (n, K) each
            # Broadcast component params over the time grid: (n, K, 1).
            dist = self._make_distribution(p1.unsqueeze(-1), p2.unsqueeze(-1))
            S_k = dist.survival(times_t)                       # (n, K, T)
            S = (w.unsqueeze(-1) * S_k).sum(dim=1)             # (n, T)
        if was_training:
            self.train()
        return S.cpu().numpy().astype(np.float64)

    def save_pretrained(self, path: str | Path) -> None:
        super().save_pretrained(path)
        if hasattr(self, "_times"):
            np.savez(Path(path) / "time_grid.npz", times=self._times)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "DSM":
        model: DSM = super().from_pretrained(path)  # type: ignore[assignment]
        time_grid_path = Path(path) / "time_grid.npz"
        if time_grid_path.exists():
            model._times = np.load(time_grid_path)["times"]
        return model
