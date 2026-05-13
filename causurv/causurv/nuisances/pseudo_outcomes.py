r"""Pseudo-outcomes for orthogonal survival learners.

Implements the DR-survival and R-survival pseudo-outcomes from
Frauen, Schröder, Hess, Feuerriegel (2025) — *Orthogonal Survival
Learners for Estimating Heterogeneous Treatment Effects from
Time-to-Event Data*, arXiv:2505.13072.

Notation follows the paper's discrete-time setting:

- $\pi(X) = P(A=1 \mid X)$ — propensity
- $S_t(X, a) = P(T > t \mid X, A = a)$ — per-arm event-free survival
- $G_t(X, a) = P(C > t \mid X, A = a)$ — per-arm censoring survival
- $\lambda_t^S(X, a) = 1 - S_t(X, a) / S_{t-1}(X, a)$ — discrete event hazard
- $\xi_S(Z, \eta_t)$ — IPCW-weighted martingale correction (paper Eq 10)

The target estimand is the per-subject survival difference
$\tau_t(x) = S_t(x, 1) - S_t(x, 0)$ at fixed time $t$. Each function
returns ``(pseudo, weight)`` for the second-stage weighted regression
``model.fit(X, pseudo, sample_weight=weight)``.

References
----------
Frauen, D., Schröder, M., Hess, K., & Feuerriegel, S. (2025).
*Orthogonal Survival Learners for Estimating Heterogeneous Treatment
Effects from Time-to-Event Data.* arXiv:2505.13072.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from causurv.nuisances.types import CrossFitNuisances


def dr_survival(
    cf: CrossFitNuisances,
    *,
    event_time: NDArray[np.float64],
    event_indicator: NDArray,
    treatment: NDArray[np.int_],
    target_time: float,
    propensity_clip: float = 1e-2,
    eps: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""DR-survival pseudo-outcome and weight (Frauen 2025 ∅-learner).

    Weighting function $f \equiv 1$, so the second-stage weight
    $\rho \equiv 1$ and

    $$
    \varphi(Z, \eta_t) =
        S_t(X, 1) - S_t(X, 0)
        - \frac{(A - \pi(X))\,\xi_S(Z, \eta_t)\,S_t(X, A)}
               {\pi(X)\,(1 - \pi(X))}.
    $$

    Sensitive to all three overlap types (treatment / censoring /
    survival); the R / C / S / TCS variants weight specific overlap
    regimes — see :func:`r_survival`.
    """
    c = _components(
        cf,
        event_time=event_time,
        event_indicator=event_indicator,
        treatment=treatment,
        target_time=target_time,
        propensity_clip=propensity_clip,
        eps=eps,
    )
    pi_one_minus_pi = np.clip(c.pi * (1.0 - c.pi), eps, None)
    phi = (
        c.S_1_at_t
        - c.S_0_at_t
        - c.A_tilde * c.xi_S * c.S_a_at_t / pi_one_minus_pi
    )
    rho = np.ones_like(phi)
    return phi.astype(np.float64), rho.astype(np.float64)


def r_survival(
    cf: CrossFitNuisances,
    *,
    event_time: NDArray[np.float64],
    event_indicator: NDArray,
    treatment: NDArray[np.int_],
    target_time: float,
    propensity_clip: float = 1e-2,
    eps: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""R-survival pseudo-outcome and weight (Frauen 2025 T-weighting).

    Weighting function $f(\tilde\eta_t) = \pi(X)(1 - \pi(X))$ —
    down-weights samples with low treatment overlap. Computed via the
    numerically-stable residualised form (paper Eq 14):

    $$
    \mathcal{L}_R(g, \eta_t)
        = \mathbb{E}\big[(\tilde Y - \tilde A\,g(X))^2\big],
    $$

    with $\tilde A = A - \pi(X)$ and
    $\tilde Y = S_t(X, A)(1 - \xi_S) - \bar S_t(X)$ where
    $\bar S_t = \pi\,S_t(X,1) + (1-\pi)\,S_t(X,0)$.

    Cast to weighted MSE for sklearn-style regressors: ``weight``
    $= \tilde A^2$, ``pseudo`` $= \tilde Y / \tilde A$. The division
    is bounded because $\tilde A$ is clipped away from zero by
    ``propensity_clip``.
    """
    c = _components(
        cf,
        event_time=event_time,
        event_indicator=event_indicator,
        treatment=treatment,
        target_time=target_time,
        propensity_clip=propensity_clip,
        eps=eps,
    )
    # Y(eta_t) = S_t(X, A) * (1 - xi_S)  (paper, derivable from Eq 9/13).
    Y = c.S_a_at_t * (1.0 - c.xi_S)
    S_bar = c.pi * c.S_1_at_t + (1.0 - c.pi) * c.S_0_at_t
    # A_tilde is bounded by [propensity_clip, 1 - propensity_clip] in
    # magnitude thanks to clipping, so division is safe.
    A_tilde_safe = np.where(
        c.A_tilde >= 0,
        np.maximum(c.A_tilde, eps),
        np.minimum(c.A_tilde, -eps),
    )
    pseudo = (Y - S_bar) / A_tilde_safe
    weight = c.A_tilde ** 2
    return pseudo.astype(np.float64), weight.astype(np.float64)


PSEUDO_OUTCOME_FOR = {
    "DR": dr_survival,
    "R": r_survival,
}


@dataclass
class _Components:
    """Per-target-time quantities shared across orthogonal weightings."""

    pi: NDArray[np.float64]      # clipped propensity P(A=1|x), shape (n,)
    S_0_at_t: NDArray[np.float64]  # S_t(x, 0)
    S_1_at_t: NDArray[np.float64]  # S_t(x, 1)
    S_a_at_t: NDArray[np.float64]  # S_t(x, A_i) — factual-arm survival
    xi_S: NDArray[np.float64]     # paper Eq 10
    A_tilde: NDArray[np.float64]  # A - pi


def _components(
    cf: CrossFitNuisances,
    *,
    event_time: NDArray[np.float64],
    event_indicator: NDArray,
    treatment: NDArray[np.int_],
    target_time: float,
    propensity_clip: float,
    eps: float,
) -> _Components:
    if cf.oof_censoring is None:
        raise ValueError(
            "orthogonal survival pseudo-outcomes require per-arm "
            "censoring; cross-fit with censoring_factory=... first."
        )
    times = np.asarray(cf.times, dtype=np.float64)
    if target_time not in times:
        raise ValueError(
            f"target_time={target_time} not in cross-fit times grid; "
            f"grid runs {times[0]}..{times[-1]} ({len(times)} bins)"
        )
    t_idx = int(np.where(times == target_time)[0][0])

    A = treatment.astype(np.int_)
    T = event_time.astype(np.float64)
    E = event_indicator.astype(np.int_)

    pi = np.clip(
        cf.oof_propensity[:, 1], propensity_clip, 1.0 - propensity_clip
    )

    S_0 = cf.oof_outcome[0]
    S_1 = cf.oof_outcome[1]
    G_0 = cf.oof_censoring[0]
    G_1 = cf.oof_censoring[1]

    A_col = A[:, None]
    S_a = np.where(A_col == 1, S_1, S_0)
    G_a = np.where(A_col == 1, G_1, G_0)

    # Discrete event hazard λ_i^S = 1 - S_i / S_{i-1}, with S_{-1} := 1.
    S_lagged = np.concatenate([np.ones((S_a.shape[0], 1)), S_a[:, :-1]], axis=1)
    lam_S = 1.0 - S_a / np.clip(S_lagged, eps, None)

    # Lagged censoring G_{i-1}(X, A): pad with 1 for i = 0.
    G_lagged = np.concatenate([np.ones((G_a.shape[0], 1)), G_a[:, :-1]], axis=1)

    # Per-time indicators 1(T = i, δ^S = 1) and 1(T ≥ i).
    times_col = times[None, :]
    event_at = ((T[:, None] == times_col) & (E[:, None] == 1)).astype(
        np.float64
    )
    at_risk = (T[:, None] >= times_col).astype(np.float64)

    numerator = (
        event_at[:, : t_idx + 1]
        - at_risk[:, : t_idx + 1] * lam_S[:, : t_idx + 1]
    )
    denominator = np.clip(S_a[:, : t_idx + 1], eps, None) * np.clip(
        G_lagged[:, : t_idx + 1], eps, None
    )
    xi_S = (numerator / denominator).sum(axis=1)

    return _Components(
        pi=pi,
        S_0_at_t=S_0[:, t_idx],
        S_1_at_t=S_1[:, t_idx],
        S_a_at_t=np.where(A == 1, S_1[:, t_idx], S_0[:, t_idx]),
        xi_S=xi_S,
        A_tilde=A.astype(np.float64) - pi,
    )
