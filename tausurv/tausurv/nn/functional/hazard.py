from __future__ import annotations

import torch
from torch import Tensor


def logistic_hazard_nll(
    logits: Tensor,
    event_time: Tensor,
    event_indicator: Tensor,
    time_bins: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    r"""Negative log-likelihood for discrete-time logistic-hazard models.

    The model outputs per-bin logits; sigmoid converts to discrete hazards
    $h_k = P(T \in \text{bin}_k \mid T \ge \text{start of bin}_k) \in (0, 1)$.
    Survival is a product: $\hat S(t_k) = \prod_{j \le k} (1 - h_j)$.
    Per-subject log-likelihood, with bin $k_i$ containing each observed time:

    - Event ($\delta_i = 1$):
      $\ell_i = \log h_{k_i} + \sum_{j < k_i} \log(1 - h_j)$.
    - Censored ($\delta_i = 0$):
      $\ell_i = \sum_{j \le k_i} \log(1 - h_j)$.

    Implemented as a sum of per-bin Bernoulli contributions for stability.

    Parameters
    ----------
    logits : (n, K) tensor
    event_time : (n,) tensor
    event_indicator : (n,) tensor
        $\delta = 1$ for events, $0$ for censored.
    time_bins : (K,) tensor
        Right edges of the $K$ bins (sorted ascending), typically built
        with :func:`tausurv.discretization.time_grid`. Observed times
        beyond the last edge fall into the final bin.
    reduction : {"mean", "sum", "none"}, default "mean"

    References
    ----------
    Gensheimer, M. F., Narasimhan, B. (2019). A scalable discrete-time
    survival model for neural networks. PeerJ, 7.
    Kvamme, H., Borgan, Ø. (2021). Continuous and discrete-time survival
    prediction with neural networks. Lifetime Data Analysis, 27(4).
    """
    if not isinstance(time_bins, Tensor):
        time_bins = torch.as_tensor(time_bins, dtype=logits.dtype, device=logits.device)

    n, K = logits.shape
    bin_idx = torch.searchsorted(time_bins, event_time, right=False).clamp(max=K - 1)

    h = torch.sigmoid(logits)
    log_h = torch.log(h.clamp(min=1e-7))
    log_1_minus_h = torch.log((1.0 - h).clamp(min=1e-7))

    bins = torch.arange(K, device=logits.device)
    before = (bins[None, :] < bin_idx[:, None]).to(logits.dtype)
    at = (bins[None, :] == bin_idx[:, None]).to(logits.dtype)
    delta = event_indicator.to(logits.dtype)[:, None]

    # Bins before observation contribute log(1 - h) (survived).
    # The observation bin contributes log h for events, log(1 - h) for censored.
    contributions = before * log_1_minus_h + at * (
        delta * log_h + (1.0 - delta) * log_1_minus_h
    )
    per_subject = contributions.sum(dim=1)

    if reduction == "none":
        return -per_subject
    if reduction == "sum":
        return -per_subject.sum()
    if reduction == "mean":
        return -per_subject.mean()
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")
