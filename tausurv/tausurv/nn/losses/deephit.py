from __future__ import annotations

from torch import Tensor, nn

from tausurv.nn.functional.deephit import deephit_loss, deephit_ranking


class DeepHitRankingLoss(nn.Module):
    r"""DeepHit pair-ranking regularizer as an ``nn.Module``.

    Class form of :func:`tausurv.nn.functional.deephit_ranking`.

    Parameters
    ----------
    sigma : float, default 0.1
        Soft-step bandwidth.
    reduction : {"mean", "sum", "none"}, default "mean"
    """

    def __init__(self, *, sigma: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.sigma = sigma
        self.reduction = reduction

    def forward(
        self,
        logits: Tensor,
        event_time: Tensor,
        event_indicator: Tensor,
        time_bins: Tensor,
    ) -> Tensor:
        return deephit_ranking(
            logits,
            event_time,
            event_indicator,
            time_bins,
            sigma=self.sigma,
            reduction=self.reduction,
        )


class DeepHitLoss(nn.Module):
    r"""Combined DeepHit loss (PMF NLL + ranking) as an ``nn.Module``.

    Class form of :func:`tausurv.nn.functional.deephit_loss`. ``alpha=1``
    recovers pure :class:`PMFLoss`.

    Parameters
    ----------
    alpha : float in [0, 1], default 0.5
        Weight on the NLL term.
    sigma : float, default 0.1
        Ranking-loss bandwidth.
    reduction : {"mean", "sum", "none"}, default "mean"
    """

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        sigma: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.reduction = reduction

    def forward(
        self,
        logits: Tensor,
        event_time: Tensor,
        event_indicator: Tensor,
        time_bins: Tensor,
    ) -> Tensor:
        return deephit_loss(
            logits,
            event_time,
            event_indicator,
            time_bins,
            alpha=self.alpha,
            sigma=self.sigma,
            reduction=self.reduction,
        )
