r"""Integral probability metrics for representation balancing.

An IPM measures the discrepancy between the distributions of latent
representations $\Phi(X)$ across treatment arms. Adding $\beta \cdot
\mathrm{IPM}(\Phi_0, \Phi_1)$ to a CATE estimator's loss penalises
representations that betray the treatment label — the Shalit et al.
(2017) "balanced representation" trick, lifted into survival by SurvITE
(Curth et al., 2021).

Two distances are provided:

- :func:`wasserstein_squared` — the (entropy-regularised) squared
  Wasserstein-2 distance computed with the Sinkhorn iteration from
  Cuturi (2013). Default in the SurvITE paper.
- :func:`mmd_linear` — linear-kernel Maximum Mean Discrepancy, cheap
  and differentiable but coarser than Wasserstein.

Both return a single non-negative scalar; ``0`` when the two batches
are identical.
"""

from __future__ import annotations

import torch
from torch import Tensor


def mmd_linear(phi_0: Tensor, phi_1: Tensor) -> Tensor:
    r"""Linear-kernel Maximum Mean Discrepancy between two batches.

    $$
    \mathrm{MMD}_{\text{lin}}^2(\Phi_0, \Phi_1)
    = \left\| \bar\phi_0 - \bar\phi_1 \right\|_2^2.
    $$

    Cheap and pleasantly behaved as a regulariser, but only catches
    mean shifts — for richer balancing prefer :func:`wasserstein_squared`.
    """
    if phi_0.ndim != 2 or phi_1.ndim != 2 or phi_0.shape[1] != phi_1.shape[1]:
        raise ValueError(
            f"phi_0 and phi_1 must be 2D with matching feature dim; "
            f"got {tuple(phi_0.shape)} and {tuple(phi_1.shape)}"
        )
    if phi_0.shape[0] == 0 or phi_1.shape[0] == 0:
        return torch.zeros((), device=phi_0.device, dtype=phi_0.dtype)
    diff = phi_0.mean(dim=0) - phi_1.mean(dim=0)
    return (diff * diff).sum()


def wasserstein_squared(
    phi_0: Tensor,
    phi_1: Tensor,
    *,
    lam: float = 10.0,
    n_iter: int = 10,
    eps: float = 1e-6,
) -> Tensor:
    r"""Sinkhorn approximation of the squared Wasserstein-2 distance.

    Computes $W_2^2$ between the empirical distributions on the two
    batches' rows under Euclidean ground cost, with entropic
    regularisation strength implicit in ``lam`` (larger = closer to the
    true OT solution, but more iterations needed). The Sinkhorn
    iteration of Cuturi (2013) makes the result differentiable in the
    inputs — exactly what we need for end-to-end training.

    Returns a single scalar. ``0`` when ``phi_0 == phi_1``.

    Parameters
    ----------
    phi_0, phi_1 : ``(n_a, d)`` tensors
        The two batches of representations.
    lam : float, default ``10.0``
        Effective regularisation strength; the legacy SurvITE uses
        this default (which corresponds to a moderately concentrated
        transport plan).
    n_iter : int, default ``10``
        Sinkhorn iterations.
    eps : float, default ``1e-6``
        Stabilisation term added to the Sinkhorn kernel.

    References
    ----------
    Cuturi, M. (2013). *Sinkhorn distances: lightspeed computation of
    optimal transport.* NeurIPS.
    Shalit, U., Johansson, F., & Sontag, D. (2017). *Estimating
    individual treatment effect: generalization bounds and algorithms.*
    ICML.
    """
    if phi_0.ndim != 2 or phi_1.ndim != 2 or phi_0.shape[1] != phi_1.shape[1]:
        raise ValueError(
            f"phi_0 and phi_1 must be 2D with matching feature dim; "
            f"got {tuple(phi_0.shape)} and {tuple(phi_1.shape)}"
        )
    n0, n1 = phi_0.shape[0], phi_1.shape[0]
    if n0 == 0 or n1 == 0:
        return torch.zeros((), device=phi_0.device, dtype=phi_0.dtype)

    device, dtype = phi_0.device, phi_0.dtype
    M = _sqeuclidean(phi_0, phi_1)
    # Uniform marginals; effective regularisation lam / mean(M).
    a = torch.full((n0, 1), 1.0 / n0, device=device, dtype=dtype)
    b = torch.full((n1, 1), 1.0 / n1, device=device, dtype=dtype)
    eff_lam = (lam / (M.mean() + eps)).detach()

    K = torch.exp(-eff_lam * M) + eps
    u = a
    for _ in range(n_iter):
        u = a / (K @ (b / (K.t() @ u)))
    v = b / (K.t() @ u)

    transport = u * (v.t() * K)
    return (transport * M).sum()


def _sqeuclidean(x: Tensor, y: Tensor) -> Tensor:
    """Pairwise squared Euclidean distance matrix, shape ``(n_x, n_y)``."""
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y; clamp tiny negatives from numerics.
    xs = (x * x).sum(dim=1, keepdim=True)
    ys = (y * y).sum(dim=1, keepdim=True)
    return (xs + ys.t() - 2.0 * x @ y.t()).clamp(min=0.0)


_DISTANCES = {
    "wasserstein": wasserstein_squared,
    "mmd": mmd_linear,
}


def get_ipm(name: str):
    """Look up an IPM by name. Supported: ``"wasserstein"``, ``"mmd"``."""
    if name not in _DISTANCES:
        raise ValueError(
            f"IPM must be one of {sorted(_DISTANCES)}, got {name!r}"
        )
    return _DISTANCES[name]
