from __future__ import annotations

from torch import Tensor, nn

from tausurv.nn.functional.copula_survival import copula_survival_nll


class CopulaSurvLoss(nn.Module):
    r"""Copula-based dependent-censoring NLL as an ``nn.Module``.

    Class form of :func:`tausurv.nn.functional.copula_survival_nll`. Use
    this when you want to bundle ``reduction`` at construction; reach
    for the functional form for compositional / one-shot calls.

    Parameters
    ----------
    reduction : {"mean", "sum", "none"}, default "mean"
    """

    def __init__(self, *, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs: dict[str, Tensor], event_indicator: Tensor) -> Tensor:
        return copula_survival_nll(outputs, event_indicator, reduction=self.reduction)
