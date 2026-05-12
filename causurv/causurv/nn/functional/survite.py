r"""SurvITE training objective in functional form.

Pairs with :class:`causurv.nn.losses.SurvITELoss` — the class form
wraps this function and bundles hyperparameters at construction.
Mirrors the tausurv convention of providing both a functional and a
:class:`nn.Module` shape for every loss
(see :func:`tausurv.nn.functional.logistic_hazard_nll` /
:class:`tausurv.nn.losses.LogisticHazardLoss`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from causurv.nn.functional.ipm import get_ipm
from tausurv.nn.functional.hazard import logistic_hazard_nll


@dataclass
class SurvITELossOutput:
    """Decomposed SurvITE loss for logging.

    Attributes
    ----------
    total : scalar tensor
        ``hazard_nll + beta * ipm`` — the value to backprop.
    hazard_nll : scalar tensor
        Factual discrete-hazard NLL summed across arms.
    ipm : scalar tensor
        Representation-balance penalty (un-weighted).
    """

    total: Tensor
    hazard_nll: Tensor
    ipm: Tensor


def survite_nll(
    phi: Tensor,
    logits_list: list[Tensor],
    event_time: Tensor,
    event_indicator: Tensor,
    treatment: Tensor,
    time_bins: Tensor,
    *,
    beta: float = 1e-3,
    ipm: str = "wasserstein",
) -> SurvITELossOutput:
    r"""SurvITE training objective.

    Combines a per-arm discrete-time hazard NLL (the factual outcome
    loss) with an integral-probability-metric penalty on the encoder
    representation $\phi$ (the balancing term, Shalit et al. 2017).

    Parameters
    ----------
    phi : ``(n, d)`` tensor
        Encoder output to penalise across arms.
    logits_list : list of ``(n, K)`` tensors
        One hazard-logit tensor per arm, in arm-label order.
    event_time, event_indicator : ``(n,)`` tensors
    treatment : ``(n,)`` integer tensor
        Arm labels in ``{0, 1, ..., n_arms - 1}``.
    time_bins : ``(K,)`` tensor
        Right-edges of the discrete-time bins (sorted ascending).
    beta : float, default ``1e-3``
        IPM weight.
    ipm : ``{"wasserstein", "mmd"}``, default ``"wasserstein"``

    Returns
    -------
    :class:`SurvITELossOutput`

    Notes
    -----
    Multi-arm: for $K > 2$ arms the IPM penalty is the average of all
    $\binom{K}{2}$ pairwise distances. Binary degenerates to the single
    SurvITE term.
    """
    ipm_fn = get_ipm(ipm)
    n_arms = len(logits_list)
    device = phi.device

    hazard_nll = torch.zeros((), device=device, dtype=phi.dtype)
    for a, logits_a in enumerate(logits_list):
        mask = treatment == a
        if not mask.any():
            continue
        hazard_nll = hazard_nll + logistic_hazard_nll(
            logits_a[mask],
            event_time[mask],
            event_indicator[mask],
            time_bins,
            reduction="mean",
        )

    ipm_total = torch.zeros((), device=device, dtype=phi.dtype)
    n_pairs = 0
    for a in range(n_arms):
        for b in range(a + 1, n_arms):
            phi_a = phi[treatment == a]
            phi_b = phi[treatment == b]
            if phi_a.shape[0] == 0 or phi_b.shape[0] == 0:
                continue
            ipm_total = ipm_total + ipm_fn(phi_a, phi_b)
            n_pairs += 1
    if n_pairs > 0:
        ipm_total = ipm_total / n_pairs

    total = hazard_nll + beta * ipm_total
    return SurvITELossOutput(total=total, hazard_nll=hazard_nll, ipm=ipm_total)
