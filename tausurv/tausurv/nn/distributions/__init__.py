"""Differentiable survival distributions (torch).

Mirror of :mod:`tausurv.distributions` with ``torch.Tensor`` parameters
and outputs. Same uniform survival-flavored API; gradients flow through
distribution parameters.
"""

from tausurv.nn.distributions.base import SurvivalDistribution
from tausurv.nn.distributions.exponential import Exponential
from tausurv.nn.distributions.gompertz import Gompertz
from tausurv.nn.distributions.loglogistic import LogLogistic
from tausurv.nn.distributions.lognormal import LogNormal
from tausurv.nn.distributions.weibull import Weibull

__all__ = [
    "Exponential",
    "Gompertz",
    "LogLogistic",
    "LogNormal",
    "SurvivalDistribution",
    "Weibull",
]
