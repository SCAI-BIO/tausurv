"""Make sure both causurv and tausurv resolve when tests run from any cwd.

The repo is a uv workspace with two packages — when running via the
workspace venv (``.venv/bin/python``) both are already on the path. When
running with a system Python that's never seen the workspace, we need
to bootstrap the sibling ``tausurv`` package onto ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TAUSURV_ROOT = Path(__file__).resolve().parents[2] / "tausurv"
if _TAUSURV_ROOT.exists() and str(_TAUSURV_ROOT) not in sys.path:
    sys.path.insert(0, str(_TAUSURV_ROOT))
