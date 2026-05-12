"""Generate all SVG figures used in the docs.

Discovers every ``docs/figures/<name>.py`` (excluding leading-underscore
helpers), imports it, and calls its ``make(out_path)`` function. Each figure
script is self-contained and runnable standalone:

    uv run --group docs python docs/figures/thinking_swimmer.py

Run this driver to rebuild everything:

    uv run --group docs python scripts/build_figures.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
FIGURES_DIR = DOCS / "figures"
OUT_DIR = DOCS / "public" / "figures"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(FIGURES_DIR))

    for source in sorted(FIGURES_DIR.glob("*.py")):
        if source.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(source.stem, source)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        make = getattr(module, "make", None)
        if make is None:
            print(f"skip {source.name}: no make()")
            continue
        out_path = OUT_DIR / f"{source.stem}.svg"
        make(out_path)
        print(f"wrote {out_path.relative_to(DOCS)}")


if __name__ == "__main__":
    main()
