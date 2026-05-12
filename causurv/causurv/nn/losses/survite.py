r"""SurvITE training loss as an ``nn.Module``.

Thin wrapper around :func:`causurv.nn.functional.survite_nll` that
bundles the time grid and hyperparameters at construction time.
"""

from __future__ import annotations

from torch import Tensor, nn

from causurv.nn.functional.survite import SurvITELossOutput, survite_nll


class SurvITELoss(nn.Module):
    r"""SurvITE training criterion (class form).

    Parameters
    ----------
    time_bins : ``(K,)`` tensor
        Right-edges of the discrete-time bins (sorted ascending).
        Registered as a buffer so the loss moves with the model on
        ``.to(device)``.
    beta : float, default ``1e-3``
        IPM weight $\beta$.
    ipm : ``{"wasserstein", "mmd"}``, default ``"wasserstein"``
    """

    def __init__(
        self,
        time_bins: Tensor,
        *,
        beta: float = 1e-3,
        ipm: str = "wasserstein",
    ) -> None:
        super().__init__()
        self.register_buffer("time_bins", time_bins)
        self.beta = float(beta)
        self.ipm_name = ipm

    def forward(
        self,
        phi: Tensor,
        logits_list: list[Tensor],
        event_time: Tensor,
        event_indicator: Tensor,
        treatment: Tensor,
    ) -> SurvITELossOutput:
        return survite_nll(
            phi,
            logits_list,
            event_time,
            event_indicator,
            treatment,
            self.time_bins,
            beta=self.beta,
            ipm=self.ipm_name,
        )
