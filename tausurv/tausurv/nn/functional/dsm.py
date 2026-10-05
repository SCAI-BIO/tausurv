from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from tausurv.nn.distributions import LogNormal, Weibull


def dsm_nll(
    predictions: Tensor,
    event_time: Tensor,
    event_indicator: Tensor,
    *,
    distribution: Literal["weibull", "lognormal"] = "weibull",
    reduction: str = "mean",
) -> Tensor:
    r"""Deep Survival Machines mixture NLL (Nagpal et al., 2021).

    The model predicts $3K$ raw values per subject — interpreted as
    $K$ shape (or location), $K$ scale, and $K$ mixture-weight logits.
    After constraining shape/scale to be positive and softmaxing the
    weights, the marginal survival and density are mixtures:

    $$
    \hat S(t \mid x) = \sum_{k=1}^K w_k(x)\, S_k\big(t;\, \theta_k(x)\big),
    \qquad
    \hat f(t \mid x) = \sum_{k=1}^K w_k(x)\, f_k\big(t;\, \theta_k(x)\big).
    $$

    The log-likelihood is computed with ``logsumexp`` for stability:

    - Event ($\delta_i = 1$):
      $\log \hat f(t_i \mid x_i) = \operatorname{logsumexp}_k\big(\log w_k + \log f_k(t_i)\big)$.
    - Censored ($\delta_i = 0$):
      $\log \hat S(t_i \mid x_i) = \operatorname{logsumexp}_k\big(\log w_k + \log S_k(t_i)\big)$.

    Parameters
    ----------
    predictions : (n, 3K) tensor
        Raw output of :meth:`DSM.forward`. Columns ``[:K]``, ``[K:2K]``,
        ``[2K:3K]`` are the raw shape (or location), raw scale, and
        raw mixture logits respectively.
    event_time : (n,) tensor
    event_indicator : (n,) tensor
        $\delta = 1$ for events, $0$ for right-censored.
    distribution : {"weibull", "lognormal"}, default "weibull"
        Per-component distribution family. Must match the model's
        ``config.distribution_family``.
    reduction : {"mean", "sum", "none"}, default "mean"

    References
    ----------
    Nagpal, C., Li, X., Dubrawski, A. (2021). Deep Survival Machines:
    Fully Parametric Survival Regression and Representation Learning for
    Censored Data with Competing Risks. IEEE JBHI 25(8).
    """
    n, three_K = predictions.shape
    if three_K % 3 != 0:
        raise ValueError(f"predictions last dim must be divisible by 3, got {three_K}")
    K = three_K // 3
    raw_p1 = predictions[:, :K]
    raw_p2 = predictions[:, K : 2 * K]
    raw_w = predictions[:, 2 * K :]

    eps = 1e-4
    if distribution == "weibull":
        p1 = F.softplus(raw_p1) + eps  # shape
        p2 = F.softplus(raw_p2) + eps  # scale
        dist = Weibull(p1, p2)
    elif distribution == "lognormal":
        p1 = raw_p1  # mu (unconstrained)
        p2 = F.softplus(raw_p2) + eps  # sigma
        dist = LogNormal(p1, p2)
    else:
        raise ValueError(
            f"distribution must be 'weibull' or 'lognormal', got {distribution!r}"
        )

    log_w = F.log_softmax(raw_w, dim=-1)  # (n, K)
    t_b = event_time.unsqueeze(-1)  # (n, 1) → broadcasts to (n, K)
    log_f_k = dist.log_pdf(t_b)  # (n, K)
    log_S_k = dist.log_survival(t_b)  # (n, K)

    log_f = torch.logsumexp(log_w + log_f_k, dim=-1)  # (n,)
    log_S = torch.logsumexp(log_w + log_S_k, dim=-1)  # (n,)

    is_event = (event_indicator > 0).to(log_f.dtype)
    log_lik = is_event * log_f + (1.0 - is_event) * log_S

    if reduction == "none":
        return -log_lik
    if reduction == "sum":
        return -log_lik.sum()
    if reduction == "mean":
        return -log_lik.mean()
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")
