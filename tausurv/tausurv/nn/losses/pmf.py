from __future__ import annotations

from torch import Tensor, nn

from tausurv.nn.functional.pmf import pmf_nll


class PMFLoss(nn.Module):
    r"""Discrete-time PMF negative log-likelihood as an ``nn.Module``.

    Class form of :func:`tausurv.nn.functional.pmf_nll`.

    Parameters
    ----------
    reduction : {"mean", "sum", "none"}, default "mean"
    """

    def __init__(self, *, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: Tensor,
        event_time: Tensor,
        event_indicator: Tensor,
        time_bins: Tensor,
    ) -> Tensor:
        return pmf_nll(
            logits, event_time, event_indicator, time_bins, reduction=self.reduction
        )
