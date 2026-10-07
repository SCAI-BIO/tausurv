r"""Copula-based deep survival model with dependent censoring.

After Gharari Foomani et al., "Copula-Based Deep Survival Models for
Dependent Censoring" (2023, arXiv:2306.11912). Two neural networks predict the
marginal survival functions for the event time $T$ and the censoring
time $C$, and an Archimedean copula joins them on the survival scale:

$$
\Pr(T > t, C > t \mid x) = C\big(S_T(t \mid x),\, S_C(t \mid x);\,\theta\big).
$$

The likelihood under this construction drops the Kaplan–Meier–style
independent-censoring assumption — censored rows actually inform the
copula parameter and the event-time marginal. This is the precursor
model to HACSurv (which generalises to $K$ competing causes with a
hierarchical copula).

The marginal survival nets use :class:`MonotoneMLP` to guarantee
$S_T(\cdot \mid x)$ and $S_C(\cdot \mid x)$ are monotone non-increasing
in $t$ by construction. The copula parameter is wrapped in the
family's valid domain via softplus.

``forward(X, t)`` returns the full dict of quantities the loss needs
(``S_T``, ``S_C``, ``f_T``, ``f_C``, ``joint``, ``dC_dST``, ``dC_dSC``)
with autograd-derived densities and copula partials computed inside
forward. Pair with :func:`tausurv.nn.functional.copula_survival_nll`
(functional) or :class:`tausurv.nn.losses.CopulaSurvLoss` (class form).

The :class:`CopulaSurvTrainer` subclass handles the ``(X, t, e)`` batch
routing and enables grad context during eval (the density needs it).
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

from tausurv.nn import copulas as nn_copulas
from tausurv.nn._utils import as_model_tensor
from tausurv.nn.checkpoint import CheckpointMixin
from tausurv.nn.copulas.base import ArchimedeanCopula
from tausurv.nn.modules import MLP, MonotoneMLP
from tausurv.nn.training.trainer import Trainer
from tausurv.predictor import SurvivalPredictor

_COPULA_CLASSES: dict[str, type[ArchimedeanCopula]] = {
    "independence": nn_copulas.Independence,
    "clayton": nn_copulas.Clayton,
    "gumbel": nn_copulas.Gumbel,
    "frank": nn_copulas.Frank,
    "joe": nn_copulas.Joe,
}


@dataclass
class CopulaSurvConfig:
    """Configuration for :class:`CopulaSurv`."""

    in_features: int
    t_max: float = 1.0
    # Encoder (unconstrained features)
    encoder_hidden_features: tuple[int, ...] = (64, 64)
    encoder_activation: Literal["gelu", "relu"] = "gelu"
    encoder_norm: Literal["layer", "batch", "none"] = "layer"
    encoder_dropout: float = 0.1
    encoder_residual: bool = True
    # Bottleneck — narrow projection into the monotone marginal nets
    monotone_dim: int = 32
    # Monotone marginal nets
    monotone_hidden: tuple[int, ...] = (32, 32, 32)
    monotone_activation: Literal["tanh", "sigmoid"] = "tanh"
    # Copula
    copula_family: Literal["independence", "clayton", "gumbel", "frank", "joe"] = (
        "clayton"
    )
    copula_theta_init: float = 2.0

    def __post_init__(self):
        # JSON round-trip converts tuples to lists; normalize so equality
        # checks and isinstance() are stable across save/load.
        if isinstance(self.encoder_hidden_features, list):
            self.encoder_hidden_features = tuple(self.encoder_hidden_features)
        if isinstance(self.monotone_hidden, list):
            self.monotone_hidden = tuple(self.monotone_hidden)


def _constrain_theta(family: str, raw: Tensor) -> Tensor:
    """Map an unconstrained raw parameter to the family's valid θ range."""
    if family == "gumbel" or family == "joe":
        return 1.0 + F.softplus(raw)
    if family == "clayton":
        return F.softplus(raw) + 1e-4
    if family == "frank":
        # Any nonzero real; nudge away from zero so the copula is defined.
        return torch.where(raw.abs() < 1e-4, torch.full_like(raw, 1e-4), raw)
    if family == "independence":
        return torch.zeros_like(raw)
    raise ValueError(f"unknown copula family {family!r}")


def _inverse_constrain_theta(family: str, theta: float) -> float:
    """Compute a raw value that yields ``theta`` under the constraint."""
    if family == "gumbel" or family == "joe":
        if theta <= 1.0:
            raise ValueError(f"{family} theta_init must be > 1, got {theta}")
        return float(np.log(np.expm1(theta - 1.0)))
    if family == "clayton":
        if theta <= 0:
            raise ValueError(f"clayton theta_init must be > 0, got {theta}")
        return float(np.log(np.expm1(theta - 1e-4)))
    if family == "frank":
        return float(theta)
    if family == "independence":
        return 0.0
    raise ValueError(f"unknown copula family {family!r}")


class CopulaSurv(SurvivalPredictor, CheckpointMixin, nn.Module):
    r"""Copula-based deep survival model with dependent censoring.

    Construct with :class:`CopulaSurvConfig` or kwargs (HF-style). After
    training (via :class:`CopulaSurvTrainer` +
    :func:`tausurv.nn.functional.copula_survival_nll`), inherits the
    unified prediction API from :class:`SurvivalPredictor`.

    Set ``copula_family="independence"`` to recover a standard
    independent-censoring deep survival model (the marginal $S_C$ is
    fitted but doesn't influence $S_T$); useful as a baseline.

    References
    ----------
    Gharari Foomani, A. H., Cooper, M., Greiner, R., Krishnan, R. G.
    (2023). Copula-Based Deep Survival Models for Dependent Censoring.
    Proceedings of UAI 2023, PMLR 216, 669-680. arXiv:2306.11912.
    """

    config_class = CopulaSurvConfig
    config: CopulaSurvConfig

    def __init__(
        self,
        config: CopulaSurvConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = CopulaSurvConfig(**kwargs)
        elif kwargs:
            raise TypeError(f"{type(self).__name__}: pass `config` OR kwargs, not both")
        if config.copula_family not in _COPULA_CLASSES:
            raise ValueError(
                f"copula_family must be one of {list(_COPULA_CLASSES)}, "
                f"got {config.copula_family!r}"
            )
        self.config = config
        h_dim = config.encoder_hidden_features[-1]
        self.encoder = MLP(
            in_features=config.in_features,
            out_features=h_dim,
            hidden_dim=h_dim,
            n_blocks=len(config.encoder_hidden_features),
            activation=config.encoder_activation,
            norm=config.encoder_norm,
            dropout=config.encoder_dropout,
            residual=config.encoder_residual,
        )
        self.bottleneck = nn.Linear(h_dim, config.monotone_dim)
        self.monotone_T = MonotoneMLP(
            in_features=config.monotone_dim + 1,
            out_features=1,
            hidden_features=config.monotone_hidden,
            activation=config.monotone_activation,
        )
        self.monotone_C = MonotoneMLP(
            in_features=config.monotone_dim + 1,
            out_features=1,
            hidden_features=config.monotone_hidden,
            activation=config.monotone_activation,
        )
        self._copula_family = config.copula_family
        raw_init = _inverse_constrain_theta(
            config.copula_family, config.copula_theta_init
        )
        self.raw_theta = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    @property
    def theta(self) -> Tensor:
        """Constrained copula parameter (read-only)."""
        return _constrain_theta(self._copula_family, self.raw_theta)

    def _make_copula(self) -> ArchimedeanCopula:
        if self._copula_family == "independence":
            return _COPULA_CLASSES["independence"]()
        return _COPULA_CLASSES[self._copula_family](self.theta)

    def _features(self, X: Tensor) -> Tensor:
        h = self.encoder(X)
        return self.bottleneck(h)

    def _marginal_S(self, monotone_net: nn.Module, z: Tensor, t: Tensor) -> Tensor:
        t_norm = (
            t.unsqueeze(-1) / self.config.t_max
            if t.dim() == 1
            else t / self.config.t_max
        )
        zt = torch.cat([z, t_norm], dim=-1)
        return torch.sigmoid(-monotone_net(zt).squeeze(-1))

    def forward(self, X: Tensor, t: Tensor) -> dict[str, Tensor]:
        r"""Compute marginal survivals, densities, joint, and copula partials.

        Returns a dict of tensors all with gradients flowing back to the
        encoder, monotone marginal nets, and copula $\\theta$ — enough
        for :func:`tausurv.nn.functional.copula_survival_nll` to compute
        the dependent-censoring NLL.

        Densities $f_T, f_C$ are obtained as $-\partial S/\partial t$ via
        autograd; copula partials as
        ``torch.autograd.grad(joint, [S_T, S_C], create_graph=True)``.

        Parameters
        ----------
        X : (n, in_features) tensor
        t : (n,) tensor
            Observed times. Detached + ``requires_grad_(True)`` is set
            internally so the density-via-autograd path works regardless
            of the caller's choice.
        """
        # t must participate in the graph with requires_grad for the density.
        t_grad = t.detach().clone().requires_grad_(True)
        z = self._features(X)
        S_T = self._marginal_S(self.monotone_T, z, t_grad)
        S_C = self._marginal_S(self.monotone_C, z, t_grad)

        (dS_T_dt,) = torch.autograd.grad(
            S_T.sum(), t_grad, create_graph=True, retain_graph=True
        )
        (dS_C_dt,) = torch.autograd.grad(
            S_C.sum(), t_grad, create_graph=True, retain_graph=True
        )
        f_T = -dS_T_dt
        f_C = -dS_C_dt

        copula = self._make_copula()
        joint = copula.cdf(torch.stack([S_T, S_C], dim=-1))
        dC_dST, dC_dSC = torch.autograd.grad(joint.sum(), [S_T, S_C], create_graph=True)

        return {
            "S_T": S_T,
            "S_C": S_C,
            "f_T": f_T,
            "f_C": f_C,
            "joint": joint,
            "dC_dST": dC_dST,
            "dC_dSC": dC_dSC,
            # Cox-style ranking score: higher = more event-time risk.
            "predictions": -S_T,
        }

    def set_time_grid(self, times: ArrayLike) -> "CopulaSurv":
        r"""Set the default time grid used by :meth:`predict_survival_function`
        (and the rest of the :class:`SurvivalPredictor` API) when ``times=None``.

        Continuous-time copula models have no natural grid baked into the
        architecture; this lets the caller pin one once after fit instead
        of passing ``times=...`` to every predict call. Round-trips through
        :meth:`save` / :meth:`load`.
        """
        self._times = np.asarray(times, dtype=np.float64)
        return self

    @property
    def times_(self) -> NDArray[np.float64]:
        if not hasattr(self, "_times"):
            raise RuntimeError(
                f"{type(self).__name__}.times_ is not set. Pass `times` "
                f"explicitly to predict_* methods, or call "
                f"`model.set_time_grid(times)` first."
            )
        return self._times

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        X_t = as_model_tensor(X, self)
        times_t = torch.as_tensor(times, dtype=X_t.dtype, device=X_t.device)
        n = X_t.shape[0]
        T_n = times_t.shape[0]
        was_training = self.training
        self.eval()
        with torch.no_grad():
            z = self._features(X_t)  # (n, monotone_dim)
            z_exp = z.unsqueeze(1).expand(-1, T_n, -1)  # (n, T, monotone_dim)
            t_exp = times_t.unsqueeze(0).expand(n, -1).unsqueeze(-1) / self.config.t_max
            zt = torch.cat([z_exp, t_exp], dim=-1)  # (n, T, monotone_dim+1)
            zt_flat = zt.reshape(-1, zt.shape[-1])
            raw_T = self.monotone_T(zt_flat).reshape(n, T_n)
            S_T = torch.sigmoid(-raw_T)
        if was_training:
            self.train()
        return S_T.cpu().numpy().astype(np.float64)

    def save(self, path: str | Path) -> None:
        """Persist config + weights, plus the time grid set by
        :meth:`set_time_grid` (if any)."""
        super().save(path)
        if hasattr(self, "_times"):
            np.savez(Path(path) / "time_grid.npz", times=self._times)

    @classmethod
    def load(cls, path: str | Path) -> "CopulaSurv":
        model: CopulaSurv = super().load(path)  # type: ignore[assignment]
        time_grid_path = Path(path) / "time_grid.npz"
        if time_grid_path.exists():
            model._times = np.load(time_grid_path)["times"]
        return model


class CopulaSurvTrainer(Trainer):
    r"""Trainer for :class:`CopulaSurv`-style models.

    The standard :class:`Trainer` calls ``model(X)`` and
    ``loss_fn(predictions, *targets)``. Copula-based models need both
    ``X`` and ``t`` in the forward pass (the loss density is the
    derivative of $S$ w.r.t. $t$), so this subclass routes the
    ``(X, t, e)`` batch as

    .. code-block:: python

        outputs = model(X, t)         # dict with f_T, f_C, dC_dST, dC_dSC, ...
        loss = loss_fn(outputs, e)    # e.g. copula_survival_nll

    The eval path enables grad locally so the density-via-autograd in
    :meth:`CopulaSurv.forward` still works (no backprop, just grad
    through the forward).
    """

    def train_step(self, batch: tuple[Tensor, ...]) -> Tensor:
        if self.loss_fn is None:
            raise NotImplementedError(
                "CopulaSurvTrainer.train_step requires loss_fn "
                "(e.g., copula_survival_nll)."
            )
        self.model.train()
        X, t, e = batch
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(X, t)
        loss = self.loss_fn(outputs, e)
        if not torch.isfinite(loss):
            return loss
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss

    def eval_step(self, batch: tuple[Tensor, ...]) -> tuple[dict[str, Tensor], Tensor]:
        if self.loss_fn is None:
            raise NotImplementedError("CopulaSurvTrainer.eval_step requires loss_fn.")
        self.model.eval()
        X, t, e = batch
        # Density-via-autograd needs grad enabled (we just don't backprop).
        with torch.enable_grad():
            outputs = self.model(X, t)
            loss = self.loss_fn(outputs, e)
        # Detach for downstream val metrics.
        outputs = {
            k: v.detach() if torch.is_tensor(v) else v for k, v in outputs.items()
        }
        return outputs, loss
