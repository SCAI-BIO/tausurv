from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from tausurv.nn._utils import (
    as_model_tensor,
    as_training_tensors,
    fit_time_grid,
    hazard_survival,
    reset_parameters,
    step_lookup,
)
from tausurv.nn.checkpoint import CheckpointMixin
from tausurv.nn.losses.hazard import LogisticHazardLoss
from tausurv.nn.modules.mlp import MLP
from tausurv.nn.training.fit import fit
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


class LogisticHazard(SurvivalPredictor, CheckpointMixin, nn.Module):
    r"""Discrete-time hazard-parameterized survival model.

    The "Nnet-Survival" model of Gensheimer & Narasimhan (2019) and the
    "Logistic-Hazard" method of Kvamme & Borgan (2021); the discrete-time
    logistic hazard goes back to Brown (1975).
    Outputs logits over $K$ time bins; ``sigmoid(logits)`` yields per-bin
    conditional hazards
    $h_k = P(T \in \text{bin}_k \mid T \ge \text{start of bin}_k)$,
    from which survival is recovered as
    $\hat S(t_k) = \prod_{j \le k} (1 - h_j)$.

    Compared with DeepHit's PMF parameterization, the hazard form is more
    stable on small datasets and avoids the softmax-normalization
    competition between bins.

    The model's output is bin-indexed (1..``n_bins``). :meth:`fit` derives
    the real-time interpretation of those bins from the training data with
    :func:`tausurv.discretization.time_grid`; when training through
    :class:`~tausurv.nn.Trainer` instead, set it with :meth:`set_time_grid`.
    If unset, prediction methods raise. The grid round-trips through
    :meth:`save` / :meth:`load`.

    Pair with :func:`tausurv.nn.functional.logistic_hazard_nll` for training.

    References
    ----------
    Brown, C. C. (1975). On the use of indicator variables for studying
    the time-dependence of parameters in a response-time model.
    Biometrics, 31(4), 863-872.
    Gensheimer, M. F., Narasimhan, B. (2019). A scalable discrete-time
    survival model for neural networks. PeerJ, 7.
    Kvamme, H., Borgan, Ø. (2021). Continuous and discrete-time survival
    prediction with neural networks. Lifetime Data Analysis, 27(4).
    """

    config_class = LogisticHazardConfig
    config: LogisticHazardConfig
    history_: dict[str, Any]

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

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
        *,
        val_data: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
        loss: Callable[..., Tensor] | None = None,
        epochs: int = 100,
        batch_size: int | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        seed: int = 0,
        device: str | torch.device | None = None,
        verbose: bool = False,
    ) -> "LogisticHazard":
        r"""Train on the discrete-time hazard likelihood.

        A thin wrapper around :func:`tausurv.nn.fit` with
        :class:`~tausurv.nn.losses.LogisticHazardLoss` as the default loss.
        The time grid comes from :func:`tausurv.discretization.time_grid` on
        the training data unless :meth:`set_time_grid` was called first. Use
        :class:`~tausurv.nn.Trainer` directly for schedules, early stopping,
        checkpointing or a custom training step.

        Parameters
        ----------
        X : (n, d) array
        event_time : (n,) array
        event_indicator : (n,) array
        val_data : (X, event_time, event_indicator), optional
            Held-out data; its loss is recorded in ``history_``.
        loss : callable, optional
            Replaces the default loss. Called as
            ``loss(logits, event_time, event_indicator, time_bins)``.
        epochs, batch_size, lr, weight_decay, seed, device, verbose
            Forwarded to :func:`tausurv.nn.fit`.

        Weights are re-initialised under ``seed`` first, so a second call
        retrains from scratch rather than continuing.

        Returns
        -------
        self
            With ``history_`` and the time grid set.
        """
        reset_parameters(self, seed)
        train = as_training_tensors(self, X, event_time, event_indicator)
        val = as_training_tensors(self, *val_data) if val_data is not None else None
        fit_time_grid(self, event_time, event_indicator)
        time_bins = as_model_tensor(self.times_, self)
        criterion = LogisticHazardLoss() if loss is None else loss

        def loss_fn(logits: Tensor, time: Tensor, indicator: Tensor) -> Tensor:
            return criterion(logits, time, indicator, time_bins)

        self.history_ = fit(
            self,
            loss_fn,
            train,
            val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        return self

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
        if was_training:
            self.train()
        return step_lookup(self._times, hazard_survival(logits), times, before=1.0)

    def save(self, path: str | Path) -> None:
        """Persist config + weights, plus the time grid set by
        :meth:`set_time_grid` (if any)."""
        super().save(path)
        if hasattr(self, "_times"):
            np.savez(Path(path) / "time_grid.npz", times=self._times)

    @classmethod
    def load(cls, path: str | Path) -> "LogisticHazard":
        model: LogisticHazard = super().load(path)  # type: ignore[assignment]
        time_grid_path = Path(path) / "time_grid.npz"
        if time_grid_path.exists():
            model._times = np.load(time_grid_path)["times"]
        return model
