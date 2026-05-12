from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from tausurv.nn.functional.dsm import dsm_nll


class DSMLoss(nn.Module):
    r"""Deep Survival Machines mixture NLL as an ``nn.Module``.

    Class form of :func:`tausurv.nn.functional.dsm_nll`. Use this when
    bundling ``distribution`` and ``reduction`` at construction; reach
    for the functional form for compositional / one-shot calls.

    Parameters
    ----------
    distribution : {"weibull", "lognormal"}, default "weibull"
        Must match the DSM model's ``config.distribution_family``.
    reduction : {"mean", "sum", "none"}, default "mean"
    """

    def __init__(
        self,
        *,
        distribution: Literal["weibull", "lognormal"] = "weibull",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.distribution = distribution
        self.reduction = reduction

    def forward(
        self,
        predictions: Tensor,
        event_time: Tensor,
        event_indicator: Tensor,
    ) -> Tensor:
        return dsm_nll(
            predictions,
            event_time,
            event_indicator,
            distribution=self.distribution,
            reduction=self.reduction,
        )
