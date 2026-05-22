r"""HACSurv: copula-based deep survival for dependent competing risks.

After Liu, Zhang, Zhang, "HACSurv: A Hierarchical Copula-Based Approach
for Survival Analysis with Dependent Competing Risks" (AISTATS 2025,
arXiv:2410.15180).

$K$ neural networks predict marginal survival functions $S_k(t \mid x)$
for each competing cause, and an Archimedean copula joins them on the
survival scale:

$$
\Pr(T_1 > t, \dots, T_K > t \mid x) = C\big(S_1(t \mid x), \dots, S_K(t \mid x);\, \theta\big).
$$

The copula can be:

- **Analytical** — Clayton, Gumbel, Frank, Joe, or Independence from
  :mod:`tausurv.nn.copulas`, with a single trainable $\theta$.
- **Learned** — a Bernstein-mixture Laplace transform
  $\varphi^{-1}(s) = \mathbb{E}_M[\exp(-s M)]$ with $M = \exp(g_\theta(U))$
  parameterised by a small neural network. By Bernstein's theorem any
  mixture of decaying exponentials is completely monotone, so the result
  is a valid Archimedean generator without assuming a parametric family.

Generalises :class:`CopulaSurv` from $K = 2$ (event + censoring) to
arbitrary $K$ competing causes. Same architectural pattern: encoder +
bottleneck + per-cause :class:`MonotoneMLP` marginals + copula joint +
autograd-derived densities and partials.

The "hierarchical" aspect of the paper (composing multiple copulas in a
tree) is not yet implemented; this is the flat-copula version.
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
from tausurv.nn.copulas.base import ArchimedeanCopula
from tausurv.nn.modules import MLP, MonotoneMLP
from tausurv.nn.pretrained import PretrainedMixin
from tausurv.nn.training.trainer import Trainer
from tausurv.predictor import CompetingRisksPredictor


_ANALYTICAL_COPULAS: dict[str, type[ArchimedeanCopula]] = {
    "independence": nn_copulas.Independence,
    "clayton": nn_copulas.Clayton,
    "gumbel": nn_copulas.Gumbel,
    "frank": nn_copulas.Frank,
    "joe": nn_copulas.Joe,
}


@dataclass
class HACSurvConfig:
    """Architectural configuration for :class:`HACSurv`."""

    in_features: int
    n_causes: int = 2
    t_max: float = 1.0
    encoder_hidden_features: tuple[int, ...] = (64, 64)
    encoder_activation: Literal["gelu", "relu"] = "gelu"
    encoder_norm: Literal["layer", "batch", "none"] = "layer"
    encoder_dropout: float = 0.1
    encoder_residual: bool = True
    monotone_dim: int = 32
    monotone_hidden: tuple[int, ...] = (32, 32, 32)
    monotone_activation: Literal["tanh", "sigmoid"] = "tanh"
    # Copula
    copula_family: Literal[
        "learned", "independence", "clayton", "gumbel", "frank", "joe"
    ] = "learned"
    copula_theta_init: float = 2.0
    # Only used when copula_family == "learned"
    learned_hidden_dim: int = 10
    learned_n_samples: int = 200

    def __post_init__(self):
        # JSON round-trip converts tuples to lists; normalize so equality
        # checks and isinstance() are stable across save/load.
        if isinstance(self.encoder_hidden_features, list):
            self.encoder_hidden_features = tuple(self.encoder_hidden_features)
        if isinstance(self.monotone_hidden, list):
            self.monotone_hidden = tuple(self.monotone_hidden)


def _constrain_theta(family: str, raw: Tensor) -> Tensor:
    if family in ("gumbel", "joe"):
        return 1.0 + F.softplus(raw)
    if family == "clayton":
        return F.softplus(raw) + 1e-4
    if family == "frank":
        return torch.where(raw.abs() < 1e-4, torch.full_like(raw, 1e-4), raw)
    if family == "independence":
        return torch.zeros_like(raw)
    raise ValueError(f"unknown copula family {family!r}")


def _inverse_constrain_theta(family: str, theta: float) -> float:
    if family in ("gumbel", "joe"):
        if theta <= 1.0:
            raise ValueError(f"{family} theta_init must be > 1, got {theta}")
        return float(np.log(np.expm1(theta - 1.0)))
    if family == "clayton":
        if theta <= 0:
            raise ValueError(f"clayton theta_init must be > 0, got {theta}")
        return float(np.log(np.expm1(theta - 1e-4)))
    if family == "frank":
        return float(theta)
    if family in ("independence", "learned"):
        return 0.0
    raise ValueError(f"unknown copula family {family!r}")


def _newton_root(
    phi_callable,
    y: Tensor,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Tensor:
    """Newton's method to find ``t`` such that ``phi_callable(t) == y``.

    Used to invert the learned-generator Laplace transform. Returns the
    best iterate (no autograd through the iterations — wrap in the
    implicit-function-theorem :class:`_ImplicitInverse` to attach grads).
    """
    t = torch.zeros_like(y)
    # Newton needs grad through phi_callable even if the outer caller is in
    # no_grad — re-enable locally.
    with torch.enable_grad():
        for _ in range(max_iter):
            t_req = t.detach().requires_grad_(True)
            f_t = phi_callable(t_req)
            (fp_t,) = torch.autograd.grad(f_t.sum(), t_req)
            residual = f_t - y
            if residual.abs().max().item() < tol:
                return t_req.detach()
            t = (t_req - residual / fp_t).detach()
    return t


class _ImplicitInverse(torch.autograd.Function):
    r"""Differentiable functional inverse via the implicit function theorem.

    Forward pass returns the pre-computed root $t^* = \varphi^{-1}_{\text{Newton}}(y)$.
    Backward provides exact gradients without differentiating through the
    Newton iterations: $d\varphi^{-1}/dy = 1 / \varphi'(\varphi^{-1}(y))$.
    Gradients also flow through the parameters of ``phi`` via the chain
    rule on the residual.
    """

    @staticmethod
    def forward(ctx, y, t_star, phi_at_t_star, phi_module):
        ctx.save_for_backward(y, t_star, phi_at_t_star)
        ctx.phi_module = phi_module
        return t_star

    @staticmethod
    def backward(ctx, grad):
        y, t_star, phi_at_t_star = ctx.saved_tensors
        phi_module = ctx.phi_module
        with torch.enable_grad():
            z = _ImplicitInverse.apply(y, t_star, phi_at_t_star, phi_module)
            phi_z = phi_module(z)
            (dphi_dz,) = torch.autograd.grad(phi_z.sum(), z, create_graph=True)
            grad_y = (grad / dphi_dz).clamp(-1e6, 1e6)
            grad_phi = (-grad / dphi_dz).clamp(-1e6, 1e6)
        return grad_y, None, grad_phi, None


class LearnedGenerator(nn.Module):
    r"""Completely-monotone Archimedean generator parameterised by a small net.

    By Bernstein's theorem, $\varphi^{-1}(s) = \mathbb{E}_M[\exp(-sM)]$ is
    completely monotone for any positive random variable $M$ — exactly
    the requirement for an Archimedean generator. We approximate the
    expectation by Monte-Carlo with $M_l = \exp(g_\theta(U_l))$,
    $U_l \sim \mathrm{Uniform}(0, 1)$. The mixing-network parameters
    $\theta$ are learned end-to-end through the HACSurv loss.

    Following the Nelsen convention:

    - :meth:`phi_inv` is the Laplace transform $\varphi^{-1}(s) \in [0, 1]$.
    - :meth:`phi` is its functional inverse on $[0, 1] \to [0, \infty)$,
      computed by Newton's method with implicit-function-theorem grads.

    Call :meth:`resample` before each forward pass during training so
    the mixing sample is fresh and gradients flow through the weight
    network.
    """

    def __init__(self, *, hidden_dim: int = 10, n_samples: int = 200) -> None:
        super().__init__()
        self.n_samples = n_samples
        self._weight_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        for m in self._weight_net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.bias, 0.0)
        self._M: Tensor | None = None

    @property
    def _device(self) -> torch.device:
        return next(self._weight_net.parameters()).device

    def resample(self, n_samples: int | None = None) -> None:
        r"""Draw a fresh mixing sample $M_l = \exp(g_\theta(U_l))$.

        Must be called before any forward pass during training so the
        sample is fresh and gradients flow through the weight network.
        """
        n = n_samples or self.n_samples
        u = torch.rand(n, 1, device=self._device)
        self._M = torch.exp(self._weight_net(u).squeeze(-1))

    def phi_inv(self, s: Tensor) -> Tensor:
        r"""$\varphi^{-1}(s) = \mathbb{E}_M[\exp(-sM)]$ — the Laplace transform."""
        if self._M is None:
            raise RuntimeError(
                "LearnedGenerator.phi_inv requires a mixing sample; "
                "call `resample()` first."
            )
        M = self._M
        # log-sum-exp for numerical stability.
        log_exps = -s.unsqueeze(-1) * M
        max_log = log_exps.max(dim=-1, keepdim=True).values
        stable = (log_exps - max_log).exp().mean(dim=-1)
        return stable * max_log.squeeze(-1).exp()

    def phi(self, t: Tensor, *, max_iter: int = 200, tol: float = 1e-10) -> Tensor:
        r"""$\varphi(t)$ via Newton's method on $\varphi^{-1}(t) = y$.

        Forward pass uses Newton iterations (no autograd through them);
        backward uses the implicit function theorem for clean gradients.
        """
        with torch.no_grad():
            t_star = _newton_root(self.phi_inv, t, max_iter=max_iter, tol=tol)
        t_opt = t_star.detach().requires_grad_(True)
        phi_at = self.phi_inv(t_opt)
        return _ImplicitInverse.apply(t, t_opt, phi_at, self.phi_inv)

    def cdf(self, u: Tensor) -> Tensor:
        r"""$C(u_1, \dots, u_d) = \varphi^{-1}\big(\sum_i \varphi(u_i)\big)$."""
        return self.phi_inv(self.phi(u).sum(dim=-1))

    def kendalls_tau(self, *, n_mc: int = 10_000) -> float:
        r"""Monte-Carlo Kendall's $\tau = 4\,\mathbb{E}[C(U_1, U_2)] - 1$
        for $U_1, U_2 \sim \mathrm{Uniform}(0, 1)$ iid."""
        with torch.no_grad():
            if self._M is None:
                self.resample()
            u = torch.rand(n_mc, 2, device=self._device)
            c = self.cdf(u)
            return float(4.0 * c.mean() - 1.0)


class HACSurv(CompetingRisksPredictor, PretrainedMixin, nn.Module):
    r"""Copula-based deep survival for dependent competing risks.

    See module docstring. After training (via :class:`HACSurvTrainer` +
    :func:`tausurv.nn.functional.hacsurv_nll`), inherits the unified
    prediction API from :class:`CompetingRisksPredictor`.
    """

    config_class = HACSurvConfig
    config: HACSurvConfig
    n_causes: int

    def __init__(
        self,
        config: HACSurvConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = HACSurvConfig(**kwargs)
        elif kwargs:
            raise TypeError(
                f"{type(self).__name__}: pass `config` OR kwargs, not both"
            )
        if config.n_causes < 2:
            raise ValueError(
                f"HACSurv requires n_causes >= 2 (use CopulaSurv for K=1 "
                f"with dependent censoring); got n_causes={config.n_causes}"
            )
        if config.copula_family not in {"learned", *_ANALYTICAL_COPULAS}:
            raise ValueError(
                f"copula_family must be 'learned' or one of "
                f"{list(_ANALYTICAL_COPULAS)}, got {config.copula_family!r}"
            )
        self.config = config
        self.n_causes = config.n_causes

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
        self.marginal_nets = nn.ModuleList(
            [
                MonotoneMLP(
                    in_features=config.monotone_dim + 1,
                    out_features=1,
                    hidden_features=config.monotone_hidden,
                    activation=config.monotone_activation,
                )
                for _ in range(config.n_causes)
            ]
        )

        # Copula
        self._copula_family = config.copula_family
        if config.copula_family == "learned":
            self.learned_generator: LearnedGenerator | None = LearnedGenerator(
                hidden_dim=config.learned_hidden_dim,
                n_samples=config.learned_n_samples,
            )
            self.raw_theta = None
        else:
            self.learned_generator = None
            raw_init = _inverse_constrain_theta(
                config.copula_family, config.copula_theta_init
            )
            self.raw_theta = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    @property
    def theta(self) -> Tensor | None:
        """Constrained copula parameter (None for learned generator)."""
        if self.raw_theta is None:
            return None
        return _constrain_theta(self._copula_family, self.raw_theta)

    def _get_copula(self):
        if self.learned_generator is not None:
            self.learned_generator.resample()
            return self.learned_generator
        if self._copula_family == "independence":
            return _ANALYTICAL_COPULAS["independence"]()
        return _ANALYTICAL_COPULAS[self._copula_family](self.theta)

    def _features(self, X: Tensor) -> Tensor:
        return self.bottleneck(self.encoder(X))

    def _marginal_S(self, net: nn.Module, z: Tensor, t: Tensor) -> Tensor:
        t_norm = t.unsqueeze(-1) / self.config.t_max if t.dim() == 1 else t / self.config.t_max
        zt = torch.cat([z, t_norm], dim=-1)
        return 1.0 - torch.sigmoid(net(zt).squeeze(-1))

    def forward(self, X: Tensor, t: Tensor) -> dict[str, Tensor]:
        r"""Forward at the observed times. Returns a dict the loss consumes.

        Densities $f_k = -\partial S_k / \partial t$ are obtained as
        autograd derivatives w.r.t. ``t``; copula partials
        $\partial C / \partial S_k$ via autograd w.r.t. each marginal.

        Returns
        -------
        dict with keys
            ``S`` ``(n, K)`` — marginal survivals per cause.
            ``f`` ``(n, K)`` — marginal densities at the observed times.
            ``joint`` ``(n,)`` — joint survival under the copula.
            ``dC_dS`` ``(n, K)`` — partial derivatives w.r.t. each marginal.
        """
        t_grad = t.detach().clone().requires_grad_(True)
        z = self._features(X)
        S_list = [self._marginal_S(net, z, t_grad) for net in self.marginal_nets]

        f_list = []
        for S_k in S_list:
            (dS_dt,) = torch.autograd.grad(
                S_k.sum(), t_grad, create_graph=True, retain_graph=True
            )
            f_list.append(-dS_dt)

        copula = self._get_copula()
        S_stacked = torch.stack(S_list, dim=-1)
        joint = copula.cdf(S_stacked)

        dC_dS = list(
            torch.autograd.grad(joint.sum(), S_list, create_graph=True)
        )

        return {
            "S": torch.stack(S_list, dim=-1),
            "f": torch.stack(f_list, dim=-1),
            "joint": joint,
            "dC_dS": torch.stack(dC_dS, dim=-1),
        }

    def kendalls_tau(self) -> float:
        r"""Kendall's $\tau$ from the fitted copula (closed-form for
        analytical families; Monte-Carlo for the learned generator)."""
        copula = self._get_copula()
        if hasattr(copula, "kendalls_tau"):
            tau = copula.kendalls_tau()
            return float(tau) if not torch.is_tensor(tau) else float(tau.item())
        raise NotImplementedError

    def set_time_grid(self, times: ArrayLike) -> "HACSurv":
        """Set the default time grid used by predict methods when ``times=None``."""
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

    def _cif(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""$\hat F_k(t \mid x)$ for each cause, computed as the discrete
        cumulative sum $\sum_{s \le t} (\partial C/\partial S_k)\,(-\Delta S_k)$
        across the requested time grid."""
        X_t = as_model_tensor(X, self)
        times_t = torch.as_tensor(times, dtype=X_t.dtype, device=X_t.device)
        n = X_t.shape[0]
        T_n = times_t.shape[0]

        was_training = self.training
        self.eval()
        try:
            with torch.enable_grad():
                z = self._features(X_t).detach()
                # Evaluate each marginal at every (x_i, t_k).
                marginals = []
                for net in self.marginal_nets:
                    z_exp = z.unsqueeze(1).expand(-1, T_n, -1)
                    t_exp = times_t.unsqueeze(0).expand(n, -1).unsqueeze(-1) / self.config.t_max
                    zt = torch.cat([z_exp, t_exp], dim=-1)
                    zt_flat = zt.reshape(-1, zt.shape[-1])
                    raw = net(zt_flat).reshape(n, T_n)
                    marginals.append(1.0 - torch.sigmoid(raw))
                # Detach + require grad so we can take partials w.r.t. each marginal.
                marginals_grad = [s.detach().requires_grad_(True) for s in marginals]
                copula = self._get_copula()
                S_stacked = torch.stack(marginals_grad, dim=-1)
                joint = copula.cdf(S_stacked)

                cifs = []
                for k in range(self.n_causes):
                    (dC_dSk,) = torch.autograd.grad(
                        joint.sum(), marginals_grad[k], retain_graph=True
                    )
                    S_k = marginals[k]
                    ones = torch.ones_like(S_k[:, :1])
                    dS_k = torch.diff(S_k, dim=1, prepend=ones)
                    cif_k = torch.cumsum(dC_dSk.detach() * (-dS_k), dim=1)
                    cifs.append(cif_k)
        finally:
            if was_training:
                self.train()

        # (n, K, T)
        cif_array = torch.stack(cifs, dim=1).detach().cpu().numpy().astype(np.float64)
        return cif_array

    def save_pretrained(self, path: str | Path) -> None:
        super().save_pretrained(path)
        if hasattr(self, "_times"):
            np.savez(Path(path) / "time_grid.npz", times=self._times)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "HACSurv":
        model: HACSurv = super().from_pretrained(path)  # type: ignore[assignment]
        time_grid_path = Path(path) / "time_grid.npz"
        if time_grid_path.exists():
            model._times = np.load(time_grid_path)["times"]
        return model


class HACSurvTrainer(Trainer):
    r"""Trainer for :class:`HACSurv` — handles ``(X, t, e)`` batches and
    routes through ``outputs = model(X, t); loss = loss_fn(outputs, e)``.
    Same pattern as :class:`CopulaSurvTrainer`; eval enables grad locally
    for the autograd-derived density in :meth:`HACSurv.forward`.
    """

    def train_step(self, batch: tuple[Tensor, ...]) -> Tensor:
        if self.loss_fn is None:
            raise NotImplementedError(
                "HACSurvTrainer.train_step requires loss_fn (e.g., hacsurv_nll)."
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

    def eval_step(
        self, batch: tuple[Tensor, ...]
    ) -> tuple[dict[str, Tensor], Tensor]:
        if self.loss_fn is None:
            raise NotImplementedError("HACSurvTrainer.eval_step requires loss_fn.")
        self.model.eval()
        X, t, e = batch
        with torch.enable_grad():
            outputs = self.model(X, t)
            loss = self.loss_fn(outputs, e)
        outputs = {k: v.detach() if torch.is_tensor(v) else v for k, v in outputs.items()}
        return outputs, loss
