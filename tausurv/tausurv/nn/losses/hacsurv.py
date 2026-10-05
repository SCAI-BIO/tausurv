from __future__ import annotations

from torch import Tensor, nn

from tausurv.nn.functional.hacsurv import hacsurv_nll


class HACSurvLoss(nn.Module):
    r"""Copula-based competing-risks NLL as an ``nn.Module``.

    Class form of :func:`tausurv.nn.functional.hacsurv_nll`. Use this
    when bundling ``reduction`` at construction; reach for the
    functional form for compositional / one-shot calls.

    Parameters
    ----------
    reduction : {"mean", "sum", "none"}, default "mean"
    """

    def __init__(self, *, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs: dict[str, Tensor], event_indicator: Tensor) -> Tensor:
        return hacsurv_nll(outputs, event_indicator, reduction=self.reduction)
