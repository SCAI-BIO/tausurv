"""Hero figure for the landing and quickstart pages.

The figure is exactly what the landing page's code block produces when run:
the overall Kaplan-Meier curve for the PBC trial cohort with a 95% logit-
transformed confidence band and an at-risk table aligned to the time axis.
We want a first-time visitor's first visual impression to be the same plot
they would generate themselves with the docs' canonical idiom.
"""

from __future__ import annotations

from pathlib import Path

import tausurv as ts


def make(out_path: Path) -> None:
    ts.plot.set_style("publication")
    _, Y, delta = ts.datasets.load_pbc()
    display = ts.plot.km(Y / 365.25, delta, xlabel="years from registration")
    display.fig.savefig(out_path, bbox_inches="tight")


if __name__ == "__main__":
    out = (
        Path(__file__).resolve().parent.parent / "public" / "figures" / "landing_km.svg"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    make(out)
