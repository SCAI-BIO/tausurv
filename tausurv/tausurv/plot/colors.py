"""Color palettes for tausurv plots.

The defaults are research-grounded: Okabe-Ito for categorical (CVD-safe,
grayscale-distinguishable, the most-cited science palette), Tol's qualitative
schemes as alternates, viridis-family for sequential, and ColorBrewer's RdBu
for diverging (hazard ratios, risk differences). See ``docs/plots.md`` in the
workspace for citations and rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


# Okabe & Ito (2008), reordered so that the first colour reached for is the
# primary group's blue rather than black. Wong (Nat. Methods 2011) on CVD-safety.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue            -- primary
    "#E69F00",  # orange          -- comparison
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow          -- poor on white, kept last
    "#000000",  # black           -- reference / null
)

# Paul Tol's "Bright" qualitative scheme (SRON 2021). Strong contrast on white,
# good for slides and posters.
TOL_BRIGHT: tuple[str, ...] = (
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
)

# Paul Tol's "Muted" qualitative scheme. Intentionally distinct from Okabe-Ito
# so a combined figure (e.g. competing-risks stack overlaid on group curves)
# doesn't clash.
TOL_MUTED: tuple[str, ...] = (
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
)

# Two-colour treatment vs control. Okabe-Ito blue + orange. CVD-safe,
# high-contrast in grayscale print.
TREATMENT_CONTROL: tuple[str, str] = (OKABE_ITO[0], OKABE_ITO[1])


PaletteName = Literal["okabe_ito", "tol_bright", "tol_muted"]

_PALETTES: dict[str, tuple[str, ...]] = {
    "okabe_ito": OKABE_ITO,
    "tol_bright": TOL_BRIGHT,
    "tol_muted": TOL_MUTED,
}


def categorical(n: int, palette: PaletteName = "okabe_ito") -> tuple[str, ...]:
    """Return ``n`` colours from a named categorical palette.

    Parameters
    ----------
    n : int
        Number of colours requested.
    palette : {"okabe_ito", "tol_bright", "tol_muted"}
        Named palette. ``"okabe_ito"`` is the default.

    Returns
    -------
    tuple of str
        Hex colour strings, length ``n``.

    Raises
    ------
    ValueError
        If ``palette`` is unknown, or ``n`` exceeds the palette's length.
        Categorical palettes are not silently extended -- past their length,
        perceptual distinctness is lost and a sequential or diverging map is
        the correct choice.
    """
    if palette not in _PALETTES:
        raise ValueError(
            f"unknown palette {palette!r}; expected one of {sorted(_PALETTES)}"
        )
    colours = _PALETTES[palette]
    if not 1 <= n <= len(colours):
        raise ValueError(
            f"requested {n} colours from {palette!r} (length {len(colours)}); "
            "for n outside this range use a sequential or diverging colormap"
        )
    return colours[:n]


def treatment_control() -> tuple[str, str]:
    """Return the default two-colour treatment / control palette."""
    return TREATMENT_CONTROL


def sequential(name: str = "viridis") -> Colormap:
    """Return a sequential matplotlib colormap by name.

    Defaults to ``viridis`` (Smith & van der Walt 2015): perceptually uniform,
    CVD-safe. Use ``"cividis"`` when CVD-safety is the strict priority,
    ``"magma"`` for dark backgrounds.
    """
    from matplotlib import colormaps

    return colormaps[name]


def diverging(name: str = "RdBu_r") -> Colormap:
    """Return a diverging matplotlib colormap by name.

    Defaults to ``RdBu_r`` (ColorBrewer; Harrower & Brewer 2003): blue
    indicates protective / below-reference, red indicates harmful / above-
    reference, with a true neutral centre. Suited to hazard ratios (centred
    at 1) and risk differences (centred at 0).
    """
    from matplotlib import colormaps

    return colormaps[name]
