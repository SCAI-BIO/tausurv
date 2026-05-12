from __future__ import annotations

from torch import Tensor, nn

from tausurv.nn.functional.hazard import logistic_hazard_nll


class LogisticHazardLoss(nn.Module):
    r"""Discrete-time logistic-hazard NLL as an ``nn.Module``.

    Class form of :func:`tausurv.nn.functional.logistic_hazard_nll`.

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
        return logistic_hazard_nll(
            logits, event_time, event_indicator, time_bins, reduction=self.reduction
        )
