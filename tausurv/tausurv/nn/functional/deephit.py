from __future__ import annotations

import torch
from torch import Tensor

from tausurv.nn.functional.pmf import pmf_nll


def deephit_ranking(
    logits: Tensor,
    event_time: Tensor,
    event_indicator: Tensor,
    time_bins: Tensor,
    *,
    sigma: float = 0.1,
    reduction: str = "mean",
) -> Tensor:
    r"""DeepHit pair-ranking regularizer (Lee et al. 2018, equation 3).

    For each cause $c$ and each comparable pair $(i, j)$ with
    $\delta_i = c$ and $T_i < T_j$:

    $$
    L_{\text{rank}}^{(c)} = \frac{1}{n_{\text{pairs}}^{(c)}}
        \sum_{(i, j) \in \mathcal{P}_c}
        \exp\!\left(-\frac{\hat F_{i, c}(T_i) - \hat F_{j, c}(T_i)}{\sigma}\right)
    $$

    Returns the sum of per-cause losses divided by the total comparable pairs
    across all causes (a single scalar). Larger
    $\hat F_{i, c}(T_i) - \hat F_{j, c}(T_i)$ → smaller penalty (concordant).
    For single-event data ($n_{\text{causes}} = 1$), reduces to the original
    DeepHit ranking.

    Memory is $O(n^2)$ — fine for typical training batch sizes.

    Parameters
    ----------
    logits : (n, K) or (n, n_causes, K) tensor
    event_time, event_indicator, time_bins : as in :func:`pmf_nll`.
    sigma : float, default 0.1
    reduction : {"mean", "sum"}, default "mean"
    """
    if not isinstance(time_bins, Tensor):
        time_bins = torch.as_tensor(time_bins, dtype=logits.dtype, device=logits.device)

    if logits.dim() == 2:
        logits = logits.unsqueeze(1)

    n, n_causes, K = logits.shape
    bin_idx = torch.searchsorted(time_bins, event_time, right=False).clamp(max=K - 1)

    pmf = torch.softmax(logits.reshape(n, -1), dim=1).reshape(n, n_causes, K)
    cdf_per_cause = torch.cumsum(pmf, dim=2)

    Y = event_time
    rows = torch.arange(n, device=logits.device)

    total_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
    total_pairs = torch.zeros((), dtype=logits.dtype, device=logits.device)

    later_mask = (Y[:, None] < Y[None, :]).to(logits.dtype)
    for c_idx in range(n_causes):
        is_case = (event_indicator == c_idx + 1).to(logits.dtype)
        if is_case.sum() == 0:
            continue
        F_own = cdf_per_cause[rows, c_idx, bin_idx]
        F_partner = cdf_per_cause[:, c_idx, bin_idx].T
        diff = F_own[:, None] - F_partner
        eta = torch.exp(-diff / sigma)
        comparable = later_mask * is_case[:, None]
        total_loss = total_loss + (comparable * eta).sum()
        total_pairs = total_pairs + comparable.sum()

    if reduction == "sum":
        return total_loss
    if reduction == "mean":
        return total_loss / total_pairs.clamp(min=1)
    raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")


def deephit_loss(
    logits: Tensor,
    event_time: Tensor,
    event_indicator: Tensor,
    time_bins: Tensor,
    *,
    alpha: float = 0.5,
    sigma: float = 0.1,
    reduction: str = "mean",
) -> Tensor:
    r"""Combined DeepHit loss: $\alpha \cdot \text{pmf\_nll} + (1 - \alpha) \cdot L_{\text{rank}}$.

    Works on both single-event and competing-risks logits. See :func:`pmf_nll`
    and :func:`deephit_ranking` for input shape conventions.

    Parameters
    ----------
    alpha : float in [0, 1], default 0.5
        Weight on the NLL term. ``alpha=1`` recovers pure :func:`pmf_nll`.
    sigma : float, default 0.1
    reduction : passed through.
    """
    nll = pmf_nll(logits, event_time, event_indicator, time_bins, reduction=reduction)
    rank = deephit_ranking(
        logits, event_time, event_indicator, time_bins, sigma=sigma, reduction=reduction
    )
    return alpha * nll + (1.0 - alpha) * rank
