r"""Archimedean copulas (numpy/scipy-backed).

Nelsen convention: the generator $\varphi : [0, 1] \to [0, \infty]$ is
decreasing convex with $\varphi(1) = 0$, and the joint CDF is
``phi_inv(sum(phi(u_i)))``.

Mirror of :mod:`tausurv.nn.copulas` for the differentiable torch versions.
"""

from tausurv.copulas.base import ArchimedeanCopula
from tausurv.copulas.clayton import Clayton
from tausurv.copulas.frank import Frank
from tausurv.copulas.gumbel import Gumbel
from tausurv.copulas.independence import Independence
from tausurv.copulas.joe import Joe

__all__ = [
    "ArchimedeanCopula",
    "Clayton",
    "Frank",
    "Gumbel",
    "Independence",
    "Joe",
]
