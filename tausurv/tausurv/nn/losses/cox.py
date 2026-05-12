from __future__ import annotations

from torch import Tensor, nn

from tausurv.nn.functional.cox import cox_nll


class CoxPHLoss(nn.Module):
    r"""Negative Cox partial log-likelihood as an ``nn.Module``.

    Class form of :func:`tausurv.nn.functional.cox_nll`. Use this when you
    want to bundle reduction config at construction; reach for the functional
    form for compositional / one-shot calls.

    Parameters
    ----------
    reduction : {"mean", "sum", "none"}, default "mean"
    """

    def __init__(self, *, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        log_risk: Tensor,
        event_time: Tensor,
        event_indicator: Tensor,
    ) -> Tensor:
        return cox_nll(log_risk, event_time, event_indicator, reduction=self.reduction)
