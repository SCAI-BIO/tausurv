from __future__ import annotations

import torch
from torch import Tensor


def cox_nll(
    log_risk: Tensor,
    event_time: Tensor,
    event_indicator: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    r"""Negative Cox partial log-likelihood.

    $$
    \ell(\theta) = \sum_{i: \delta_i = 1}
        \left[\theta_i - \log \sum_{j \in R(Y_i)} \exp(\theta_j)\right]
    $$

    Returns the negative of this. Computed in one stable pass via
    ``torch.logcumsumexp`` on subjects sorted by descending event time.
    Breslow tie handling is implicit and approximate; not an issue for
    random continuous-time data.

    Parameters
    ----------
    log_risk : (n,) or (n, 1) tensor
    event_time : (n,) tensor
    event_indicator : (n,) tensor
        $\delta = 1$ for events, $0$ for censored.
    reduction : {"mean", "sum", "none"}, default "mean"
    """
    if log_risk.dim() == 2:
        log_risk = log_risk.squeeze(-1)

    order = torch.argsort(event_time, descending=True, stable=True)
    log_risk_s = log_risk[order]
    delta_s = event_indicator[order].to(log_risk.dtype)

    log_S0 = torch.logcumsumexp(log_risk_s, dim=0)
    per_subject = delta_s * (log_risk_s - log_S0)

    if reduction == "none":
        return -per_subject[delta_s > 0]
    total = -per_subject.sum()
    if reduction == "sum":
        return total
    if reduction == "mean":
        n_events = delta_s.sum()
        return total / n_events.clamp(min=1)
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")
