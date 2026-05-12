from __future__ import annotations

import torch
from torch import Tensor


def hacsurv_nll(
    outputs: dict[str, Tensor],
    event_indicator: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    r"""Negative log-likelihood for copula-based competing-risks survival.

    Per the joint-survival construction
    $\Pr(T_1 > t, \dots, T_K > t \mid x) = C\big(S_1(t \mid x), \dots, S_K(t \mid x);\, \theta\big)$,
    the per-subject contribution is

    - Event of cause $k$ ($\delta_i = k \ge 1$):
      $\ell_i = \log f_k(t_i \mid x_i) + \log \!\big(\partial C/\partial S_k\big)$.
    - Censored ($\delta_i = 0$):
      $\ell_i = \log C\big(S_1(t_i \mid x_i), \dots, S_K(t_i \mid x_i)\big)$.

    where $f_k = -\partial S_k / \partial t$ are the marginal densities and
    the copula partials are evaluated at the predicted marginal survivals.

    Operates on the dict produced by ``HACSurv.forward(X, t)`` so the
    density and copula partial autograd lives in the model forward, not
    the loss.

    Parameters
    ----------
    outputs : dict
        Must contain ``S`` (shape ``(n, K)``), ``f`` ``(n, K)``,
        ``joint`` ``(n,)``, ``dC_dS`` ``(n, K)``.
    event_indicator : (n,) tensor
        $\delta = 0$ censored, $\delta = k \ge 1$ event of cause $k$.
    reduction : {"mean", "sum", "none"}, default "mean"

    References
    ----------
    Liu, X., Zhang, X., Zhang, Y. (2025). HACSurv: A Hierarchical
    Copula-Based Approach for Survival Analysis with Dependent Competing
    Risks. AISTATS. arXiv:2410.15180.
    """
    eps = 1e-8
    f = outputs["f"].clamp(min=eps)
    joint = outputs["joint"].clamp(min=eps)
    dC_dS = outputs["dC_dS"].abs().clamp(min=eps)
    n, K = f.shape

    is_censored = event_indicator == 0
    log_lik = torch.zeros(n, dtype=f.dtype, device=f.device)

    if is_censored.any():
        log_lik = torch.where(is_censored, torch.log(joint), log_lik)

    for k in range(K):
        mask = event_indicator == (k + 1)
        if not mask.any():
            continue
        contribution = torch.log(f[:, k]) + torch.log(dC_dS[:, k])
        log_lik = torch.where(mask, contribution, log_lik)

    if reduction == "none":
        return -log_lik
    if reduction == "sum":
        return -log_lik.sum()
    if reduction == "mean":
        return -log_lik.mean()
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")
