"""Paper-reproduction simulators for benchmarking causal-survival methods.

Each generator reproduces the data-generating process of a specific
paper so that learners can be evaluated against a *known* oracle HTE.
For generic single-event / competing-risks simulators (no causal
component) see :mod:`tausurv.simulations`.
"""

from causurv.simulations.survite import SurvITE

__all__ = ["SurvITE"]
