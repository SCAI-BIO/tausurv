from __future__ import annotations

import torch
from torch import Tensor


def pmf_nll(
    logits: Tensor,
    event_time: Tensor,
    event_indicator: Tensor,
    time_bins: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    r"""Negative log-likelihood for discrete-time PMF survival models.

    Handles single-event and competing-risks via the same formula. ``logits``
    may be ``(n, K)`` (single-event shortcut) or ``(n, n_causes, K)``
    (general case). For competing risks, softmax is over the joint
    ``(cause, bin)`` space so the PMF sums to 1 across all pairs per subject.

    With $\hat p_{c, k}$ the predicted (cause, bin) PMF and bin $k_i$
    containing each observed time:

    - Event ($\delta_i = c \ge 1$): $\ell_i = \log \hat p_{c, k_i}(x_i)$.
    - Censored ($\delta_i = 0$):
      $\ell_i = \log \big(1 - \sum_c \sum_{j \le k_i} \hat p_{c, j}(x_i)\big)$.

    Event coding follows the package convention: $\delta = 0$ censored,
    $\delta = c \ge 1$ event of cause $c$. For single-event, $c = 1$ for
    events.

    Parameters
    ----------
    logits : (n, K) or (n, n_causes, K) tensor
    event_time : (n,) tensor
    event_indicator : (n,) tensor — integer-valued.
    time_bins : (K,) tensor
        Right edges of the $K$ bins (sorted ascending), typically built
        with :func:`tausurv.discretization.time_grid`. Observed times
        beyond the last edge fall into the final bin.
    reduction : {"mean", "sum", "none"}, default "mean"
    """
    if not isinstance(time_bins, Tensor):
        time_bins = torch.as_tensor(time_bins, dtype=logits.dtype, device=logits.device)

    if logits.dim() == 2:
        logits = logits.unsqueeze(1)

    n, n_causes, K = logits.shape
    bin_idx = torch.searchsorted(time_bins, event_time, right=False).clamp(max=K - 1)

    pmf = torch.softmax(logits.reshape(n, -1), dim=1).reshape(n, n_causes, K)
    cdf_per_cause = torch.cumsum(pmf, dim=2)
    cdf_marginal = cdf_per_cause.sum(dim=1)

    rows = torch.arange(n, device=logits.device)
    cause_idx = (event_indicator.long() - 1).clamp(min=0, max=n_causes - 1)
    pmf_at_event = pmf[rows, cause_idx, bin_idx].clamp(min=1e-12)
    surv_at_event = (1.0 - cdf_marginal[rows, bin_idx]).clamp(min=1e-12)

    is_event = (event_indicator > 0).to(logits.dtype)
    per_subject = is_event * torch.log(pmf_at_event) + (1.0 - is_event) * torch.log(
        surv_at_event
    )

    if reduction == "none":
        return -per_subject
    if reduction == "sum":
        return -per_subject.sum()
    if reduction == "mean":
        return -per_subject.mean()
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")
