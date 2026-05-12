"""Public access to the compiled Rust engine.

This module re-exports the contents of the private ``tausurv._tausurv_core``
extension. Other packages (e.g. ``causurv``) should import from here, not
from the underscored name directly.
"""

from tausurv._tausurv_core import (
    LogRankSurvivalTree,
    fit_log_rank_tree,
    version,
)

__all__ = [
    "LogRankSurvivalTree",
    "fit_log_rank_tree",
    "version",
]
