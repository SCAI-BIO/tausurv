r"""Copula visualization plots.

Three diagnostics for an :class:`~tausurv.copulas.base.ArchimedeanCopula`:

- :func:`contour` -- level curves of the joint CDF
  :math:`C(u_1, u_2)` on the unit square.
- :func:`density` -- heatmap of the joint density
  :math:`c(u_1, u_2)`, with log colour scaling by default since
  Archimedean densities concentrate near corners.
- :func:`scatter` -- sampled :math:`(u_1, u_2)` pairs on the unit square.

Each plot annotates Kendall's :math:`\tau` in the corner and renders with
equal aspect so the geometry of dependence reads honestly. Use the same
copula instance across the three for a complete picture: contour shows the
CDF geometry, density highlights tail concentration, scatter shows what
samples actually look like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from tausurv.copulas.base import ArchimedeanCopula

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection, QuadMesh
    from matplotlib.colorbar import Colorbar
    from matplotlib.contour import QuadContourSet
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Text


@dataclass
class CopulaContourDisplay:
    r"""Result of :func:`contour`.

    Attributes
    ----------
    fig : Figure
    ax : Axes
    contours : QuadContourSet
        The level-curve artist.
    reference : Line2D or None
        The :math:`u_1 = u_2` diagonal, if drawn.
    tau_annotation : Text or None
        Kendall's :math:`\tau` corner annotation, if drawn.
    """

    fig: "Figure"
    ax: "Axes"
    contours: "QuadContourSet | None" = None
    reference: "Line2D | None" = None
    tau_annotation: "Text | None" = None


@dataclass
class CopulaDensityDisplay:
    """Result of :func:`density`.

    Attributes
    ----------
    fig : Figure
    ax : Axes
    mesh : QuadMesh
        The pcolormesh artist.
    colorbar : Colorbar or None
    tau_annotation : Text or None
    """

    fig: "Figure"
    ax: "Axes"
    mesh: "QuadMesh | None" = None
    colorbar: "Colorbar | None" = None
    tau_annotation: "Text | None" = None


@dataclass
class CopulaScatterDisplay:
    """Result of :func:`scatter`.

    Attributes
    ----------
    fig : Figure
    ax : Axes
    points : PathCollection
        The scatter artist.
    tau_annotation : Text or None
    """

    fig: "Figure"
    ax: "Axes"
    points: "PathCollection | None" = None
    tau_annotation: "Text | None" = None


def contour(
    copula: ArchimedeanCopula,
    *,
    ax: "Axes | None" = None,
    n_grid: int = 60,
    levels: "int | list[float] | np.ndarray" = 10,
    filled: bool = False,
    show_diagonal: bool = True,
    annotate_tau: bool = True,
    cmap: str = "viridis",
    color: str | None = None,
    xlabel: str = r"$u_1$",
    ylabel: str = r"$u_2$",
    title: str | None = None,
) -> CopulaContourDisplay:
    r"""Level curves of the joint copula CDF :math:`C(u_1, u_2)`.

    Parameters
    ----------
    copula : ArchimedeanCopula
        Configured copula instance.
    ax : Axes, optional
    n_grid : int, default 60
        Grid resolution on each axis. The grid is on
        :math:`(\varepsilon, 1 - \varepsilon)` to avoid boundary singularities.
    levels : int or sequence, default 10
        Forwarded to :meth:`matplotlib.axes.Axes.contour` /
        :meth:`~matplotlib.axes.Axes.contourf`.
    filled : bool, default False
        Use ``contourf`` (filled bands) instead of ``contour`` (lines).
    show_diagonal : bool, default True
        Draw the :math:`u_1 = u_2` reference. The diagonal is where
        :math:`C` equals the independence copula along the line of equal
        marginals.
    annotate_tau : bool, default True
        Print Kendall's :math:`\tau` in the corner.
    cmap : str, default "viridis"
        Colormap. Used for both line and filled variants.
    color : str, optional
        Single colour for all contour lines (overrides ``cmap``). Ignored
        when ``filled=True``.
    xlabel, ylabel : str
    title : str, optional

    Returns
    -------
    CopulaContourDisplay
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    U1, U2, C = _grid_cdf(copula, n_grid)

    disp = CopulaContourDisplay(fig=fig, ax=ax)

    if filled:
        cs = ax.contourf(U1, U2, C, levels=levels, cmap=cmap)
    elif color is not None:
        cs = ax.contour(U1, U2, C, levels=levels, colors=color, linewidths=0.8)
    else:
        cs = ax.contour(U1, U2, C, levels=levels, cmap=cmap, linewidths=0.8)
    disp.contours = cs

    if show_diagonal:
        (disp.reference,) = ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color="#888888",
            linestyle="--",
            linewidth=0.6,
            alpha=0.6,
            zorder=0,
        )

    if annotate_tau:
        disp.tau_annotation = _tau_text(ax, copula)

    _square_axes(ax, xlabel, ylabel, title)
    return disp


def density(
    copula: ArchimedeanCopula,
    *,
    ax: "Axes | None" = None,
    n_grid: int = 80,
    log_scale: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    annotate_tau: bool = True,
    cmap: str = "viridis",
    xlabel: str = r"$u_1$",
    ylabel: str = r"$u_2$",
    title: str | None = None,
) -> CopulaDensityDisplay:
    r"""Heatmap of the joint copula density :math:`c(u_1, u_2)`.

    Parameters
    ----------
    copula : ArchimedeanCopula
    ax : Axes, optional
    n_grid : int, default 80
        Pixel resolution per axis. Grid is on
        :math:`(\varepsilon, 1 - \varepsilon)`.
    log_scale : bool, default True
        Log-normalise the colour scale. Recommended for Archimedean copulas
        whose density concentrates near corners and spans orders of magnitude.
    vmin, vmax : float, optional
        Manual colour-scale bounds.
    colorbar : bool, default True
    annotate_tau : bool, default True
    cmap : str, default "viridis"
    xlabel, ylabel : str
    title : str, optional

    Returns
    -------
    CopulaDensityDisplay
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    U1, U2, D = _grid_pdf(copula, n_grid)

    if log_scale:
        positive = D[D > 0]
        norm = LogNorm(
            vmin=vmin if vmin is not None else float(np.min(positive)),
            vmax=vmax if vmax is not None else float(np.max(D)),
        )
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    mesh = ax.pcolormesh(U1, U2, D, cmap=cmap, norm=norm, shading="auto")

    disp = CopulaDensityDisplay(fig=fig, ax=ax, mesh=mesh)

    if colorbar:
        disp.colorbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
        disp.colorbar.set_label(r"$c(u_1, u_2)$")

    if annotate_tau:
        disp.tau_annotation = _tau_text(ax, copula, light_background=False)

    _square_axes(ax, xlabel, ylabel, title)
    return disp


def scatter(
    copula: ArchimedeanCopula,
    *,
    n_samples: int = 2000,
    ax: "Axes | None" = None,
    annotate_tau: bool = True,
    seed: int | None = None,
    alpha: float = 0.35,
    marker_size: float = 6.0,
    color: str | None = None,
    xlabel: str = r"$u_1$",
    ylabel: str = r"$u_2$",
    title: str | None = None,
) -> CopulaScatterDisplay:
    r"""Scatter of :math:`(u_1, u_2)` samples from the copula.

    Marshall--Olkin sampling via :meth:`~ArchimedeanCopula.sample`. Useful
    next to :func:`density` for verifying the sampler's geometry matches the
    analytic density.

    Parameters
    ----------
    copula : ArchimedeanCopula
    n_samples : int, default 2000
    ax : Axes, optional
    annotate_tau : bool, default True
    seed : int, optional
        Forwarded to :class:`numpy.random.default_rng`.
    alpha, marker_size : float
    color : str, optional
        Override the marker colour. Defaults to the first colour of the
        active palette.
    xlabel, ylabel : str
    title : str, optional

    Returns
    -------
    CopulaScatterDisplay
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    samples = copula.sample(size=n_samples, d=2, rng=rng)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    sc_kw: dict = {"s": marker_size, "alpha": alpha, "linewidth": 0}
    if color is not None:
        sc_kw["color"] = color
    points = ax.scatter(samples[:, 0], samples[:, 1], **sc_kw)

    disp = CopulaScatterDisplay(fig=fig, ax=ax, points=points)

    if annotate_tau:
        disp.tau_annotation = _tau_text(ax, copula)

    _square_axes(ax, xlabel, ylabel, title)
    return disp


def _grid_cdf(
    copula: ArchimedeanCopula,
    n: int,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(eps, 1.0 - eps, n)
    U1, U2 = np.meshgrid(u, u, indexing="xy")
    grid = np.stack([U1.ravel(), U2.ravel()], axis=-1)
    C = copula.cdf(grid).reshape(U1.shape)
    return U1, U2, C


def _grid_pdf(
    copula: ArchimedeanCopula,
    n: int,
    eps: float = 5e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(eps, 1.0 - eps, n)
    U1, U2 = np.meshgrid(u, u, indexing="xy")
    grid = np.stack([U1.ravel(), U2.ravel()], axis=-1)
    D = copula.pdf(grid).reshape(U1.shape)
    return U1, U2, D


def _square_axes(
    ax: "Axes",
    xlabel: str,
    ylabel: str,
    title: str | None,
) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    if title is not None:
        ax.set_title(title)


def _tau_text(
    ax: "Axes",
    copula: ArchimedeanCopula,
    *,
    light_background: bool = True,
) -> "Text":
    tau = copula.kendalls_tau()
    text_color = "#222222" if light_background else "white"
    return ax.text(
        0.96,
        0.04,
        rf"$\tau = {tau:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=text_color,
    )
