from __future__ import annotations

import torch
from torch import Tensor


def copula_survival_nll(
    outputs: dict[str, Tensor],
    event_indicator: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    r"""Negative log-likelihood for copula-based survival with dependent censoring.

    The per-subject contribution under the copula
    $C(S_T(t \mid x), S_C(t \mid x);\, \theta)$ is

    - Event ($\delta_i = 1$):
      $\ell_i = \log f_T(t_i \mid x_i) + \log \!\big(\partial C/\partial S_T\big)\big|_{(S_T, S_C)}$
    - Censored ($\delta_i = 0$):
      $\ell_i = \log f_C(t_i \mid x_i) + \log \!\big(\partial C/\partial S_C\big)\big|_{(S_T, S_C)}$

    where $f_T = -\partial S_T / \partial t$ and $f_C = -\partial S_C / \partial t$
    are the marginal densities, and the copula partials are evaluated at the
    predicted marginal survival values.

    Operates on the ``outputs`` dict produced by ``CopulaSurv.forward(X, t)``
    so that the density / partial-derivative autograd lives in the model
    forward, not in the loss. Loss configuration (``reduction``,
    regularizers, etc.) stays on the loss object — exactly the layering
    used by every other ``tausurv.nn.functional`` loss.

    Parameters
    ----------
    outputs : dict
        Must contain ``f_T``, ``f_C``, ``dC_dST``, ``dC_dSC`` — all
        ``(n,)`` tensors with gradients flowing back to the model parameters
        and to the copula $\\theta$.
    event_indicator : (n,) tensor
        $\delta = 1$ for events, $0$ for right-censored.
    reduction : {"mean", "sum", "none"}, default "mean"

    References
    ----------
    Zhang, W., Yi, Y., Pal, A., Lyu, J., Tarique, M., Krishnan, R. G.
    (2023). Copula-Based Deep Survival Models for Dependent Censoring.
    arXiv:2306.11912.
    """
    eps = 1e-8
    f_T = outputs["f_T"].clamp(min=eps)
    f_C = outputs["f_C"].clamp(min=eps)
    dC_dST = outputs["dC_dST"].abs().clamp(min=eps)
    dC_dSC = outputs["dC_dSC"].abs().clamp(min=eps)

    is_event = (event_indicator > 0).to(f_T.dtype)
    log_lik = is_event * (torch.log(f_T) + torch.log(dC_dST)) + (1.0 - is_event) * (
        torch.log(f_C) + torch.log(dC_dSC)
    )

    if reduction == "none":
        return -log_lik
    if reduction == "sum":
        return -log_lik.sum()
    if reduction == "mean":
        return -log_lik.mean()
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")
