"""Named plotting styles for tausurv.

Four styles, opt-in:

- ``publication``  -- single-column journal default. Compact, restrained.
- ``presentation`` -- slides and posters. Larger fonts, thicker lines.
- ``notebook``     -- interactive analysis. Matplotlib defaults plus the
  Okabe-Ito palette and minor spine/grid cleanup.
- ``minimal``      -- spineless, gridless. For embedding in mixed-media reports
  where surrounding context provides axes.

Apply with ``set_style(name)`` for a session-wide change or ``style_context(name)``
as a scoped context manager. Both accept a palette name override.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Literal

from tausurv.plot.colors import PaletteName, categorical

StyleName = Literal["publication", "presentation", "notebook", "minimal"]


_PUBLICATION: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Source Sans 3",
        "IBM Plex Sans",
        "Inter",
        "Roboto",
        "Open Sans",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
    ],
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.titleweight": "regular",
    "axes.labelsize": 9.0,
    "axes.labelpad": 3.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 8.0,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.4,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "grid.color": "#000000",
    "lines.linewidth": 1.5,
    "lines.solid_capstyle": "round",
    "patch.linewidth": 0.0,
    "figure.figsize": (5.5, 3.4),
    "figure.dpi": 150.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.facecolor": "white",
    "mathtext.fontset": "stixsans",
}


_PRESENTATION: dict[str, Any] = {
    **_PUBLICATION,
    "font.size": 13.0,
    "axes.titlesize": 15.0,
    "axes.labelsize": 13.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 12.0,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.5,
    "figure.figsize": (8.0, 5.0),
    "figure.dpi": 120.0,
}


_NOTEBOOK: dict[str, Any] = {
    **_PUBLICATION,
    "font.size": 10.0,
    "axes.titlesize": 11.0,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 9.0,
    "lines.linewidth": 1.75,
    "figure.figsize": (6.4, 4.0),
}


_MINIMAL: dict[str, Any] = {
    **_PUBLICATION,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.grid": False,
    "xtick.major.size": 0.0,
    "ytick.major.size": 0.0,
}


_STYLES: dict[str, dict[str, Any]] = {
    "publication": _PUBLICATION,
    "presentation": _PRESENTATION,
    "notebook": _NOTEBOOK,
    "minimal": _MINIMAL,
}


def _build_rcparams(
    name: StyleName,
    palette: PaletteName,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if name not in _STYLES:
        raise ValueError(f"unknown style {name!r}; expected one of {sorted(_STYLES)}")
    from cycler import cycler

    palette_colors = list(categorical(_palette_length(palette), palette=palette))
    rc: dict[str, Any] = {
        **_STYLES[name],
        "axes.prop_cycle": cycler(color=palette_colors),
    }
    if overrides:
        rc.update(overrides)
    return rc


def _palette_length(palette: PaletteName) -> int:
    from tausurv.plot.colors import _PALETTES

    return len(_PALETTES[palette])


def set_style(
    name: StyleName = "publication",
    palette: PaletteName = "okabe_ito",
    **overrides: Any,
) -> None:
    """Apply a named tausurv style globally.

    Parameters
    ----------
    name : {"publication", "presentation", "notebook", "minimal"}
        Style preset.
    palette : {"okabe_ito", "tol_bright", "tol_muted"}
        Categorical palette driving ``axes.prop_cycle``.
    **overrides
        Additional rcParam keys, applied last.

    Notes
    -----
    Mutates ``matplotlib.rcParams`` for the active session. Use
    :func:`style_context` for a scoped alternative.
    """
    import matplotlib as mpl

    mpl.rcParams.update(_build_rcparams(name, palette, overrides))


def style_context(
    name: StyleName = "publication",
    palette: PaletteName = "okabe_ito",
    **overrides: Any,
) -> AbstractContextManager[None]:
    """Return a context manager scoping a tausurv style to a ``with`` block.

    Parameters
    ----------
    name : {"publication", "presentation", "notebook", "minimal"}
    palette : {"okabe_ito", "tol_bright", "tol_muted"}
    **overrides
        Additional rcParam keys, applied last.

    Examples
    --------
    >>> with ts.plot.style_context("presentation"):
    ...     fig, ax = plt.subplots()
    ...     ax.plot(t, S)
    """
    import matplotlib as mpl

    return mpl.rc_context(rc=_build_rcparams(name, palette, overrides))
