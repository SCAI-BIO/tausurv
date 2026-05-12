"""Differentiable Archimedean copulas (torch).

Mirror of :mod:`tausurv.copulas` with ``torch.Tensor`` parameters; all
operations are closed-form vectorized torch ops. Gradients flow through
``theta``.
"""

from tausurv.nn.copulas.base import ArchimedeanCopula
from tausurv.nn.copulas.clayton import Clayton
from tausurv.nn.copulas.frank import Frank
from tausurv.nn.copulas.gumbel import Gumbel
from tausurv.nn.copulas.independence import Independence
from tausurv.nn.copulas.joe import Joe

__all__ = [
    "ArchimedeanCopula",
    "Clayton",
    "Frank",
    "Gumbel",
    "Independence",
    "Joe",
]
