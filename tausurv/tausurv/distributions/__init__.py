"""Parametric survival distributions (numpy/scipy-backed).

Uniform survival-flavored API across all distributions. Pair with
:mod:`tausurv.nn.distributions` for differentiable torch versions of the
same set.
"""

from tausurv.distributions.base import SurvivalDistribution
from tausurv.distributions.exponential import Exponential
from tausurv.distributions.gompertz import Gompertz
from tausurv.distributions.loglogistic import LogLogistic
from tausurv.distributions.lognormal import LogNormal
from tausurv.distributions.weibull import Weibull

__all__ = [
    "Exponential",
    "Gompertz",
    "LogLogistic",
    "LogNormal",
    "SurvivalDistribution",
    "Weibull",
]
