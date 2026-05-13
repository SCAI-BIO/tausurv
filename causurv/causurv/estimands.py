r"""Causal-survival estimand types.

An **estimand** says *what* causal quantity to estimate. Pass to
:meth:`HTEPredictor.predict_hte` / :meth:`predict_ate` (plug-in
learners), or at construction (orthogonal learners). Estimand-specific
parameters live on the type — :class:`RMSTDiff` requires ``horizon``,
:class:`CIFDiff` requires ``cause``, etc. — and are enforced by the
dataclass at construction, not by deep runtime checks.

Five canonical estimands for binary treatment:

============================  =========================================================
Estimand                      Definition (per-subject, conditional on $X = x$)
============================  =========================================================
:class:`SurvivalDiff`         $\tau_S(t, x) = S^{a_T}(t \mid x) - S^{a_R}(t \mid x)$
:class:`SurvivalRatio`        $S^{a_T}(t \mid x) / S^{a_R}(t \mid x)$
:class:`RMSTDiff`             $\int_0^{\tau} [S^{a_T}(u \mid x) - S^{a_R}(u \mid x)]\,du$
:class:`CIFDiff`              $F^{a_T}_j(t \mid x) - F^{a_R}_j(t \mid x)$ for cause $j$
:class:`CIFRatio`             $F^{a_T}_j / F^{a_R}_j$
============================  =========================================================

where $a_T = $ ``treatment`` and $a_R = $ ``reference``.

Accepted forms in :meth:`predict_hte`:

- **Instance**: ``predict_hte(X, estimand=SurvivalDiff(times=[5, 10]))``
- **Class + kwargs**: ``predict_hte(X, estimand=SurvivalDiff, times=[5, 10])``
- **String + kwargs**: ``predict_hte(X, estimand="survival_diff", times=[5, 10])``

All three resolve to the same :class:`Estimand` instance internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(kw_only=True)
class Estimand:
    r"""Base class for causal-survival estimands.

    Do not instantiate directly. Subclass via one of the five concrete
    estimands (:class:`SurvivalDiff`, :class:`SurvivalRatio`,
    :class:`RMSTDiff`, :class:`CIFDiff`, :class:`CIFRatio`).

    Attributes
    ----------
    contrast : str
        Class-level constant — the contrast name passed to
        :func:`causurv.contrasts.apply_contrast`.
    treatment, reference : int, default ``(1, 0)``
        The arm comparison. The estimand is computed as
        ``arm[treatment] vs arm[reference]``.
    """

    contrast: ClassVar[str]
    treatment: int = 1
    reference: int = 0

    def __post_init__(self) -> None:
        if self.treatment == self.reference:
            raise ValueError(
                f"treatment and reference must differ; got both = {self.treatment}"
            )


@dataclass(kw_only=True)
class SurvivalDiff(Estimand):
    r"""$S^{a_T}(t \mid x) - S^{a_R}(t \mid x)$ at fixed times."""

    contrast: ClassVar[str] = "survival_diff"
    times: ArrayLike

    def __post_init__(self) -> None:
        super().__post_init__()
        self.times = _validate_times(self.times)


@dataclass(kw_only=True)
class SurvivalRatio(Estimand):
    r"""$S^{a_T}(t \mid x) / S^{a_R}(t \mid x)$ at fixed times."""

    contrast: ClassVar[str] = "survival_ratio"
    times: ArrayLike

    def __post_init__(self) -> None:
        super().__post_init__()
        self.times = _validate_times(self.times)


@dataclass(kw_only=True)
class RMSTDiff(Estimand):
    r"""Difference in restricted mean survival time up to ``horizon``.

    $$
    \tau_{\text{RMST}}(x; \tau) =
        \int_0^\tau \big[S^{a_T}(u \mid x) - S^{a_R}(u \mid x)\big]\,du.
    $$

    Returns a per-subject scalar (no time axis on the output).

    Attributes
    ----------
    times : ``(T,)`` array-like
        Integration grid for the trapezoidal rule. Must cover
        ``[t_min, horizon]`` where ``t_min <= horizon <= times[-1]``.
    horizon : float
        Upper integration bound $\tau$.
    """

    contrast: ClassVar[str] = "rmst_diff"
    times: ArrayLike
    horizon: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.times = _validate_times(self.times)
        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0; got {self.horizon}")
        if self.horizon > float(self.times[-1]) + 1e-9:
            raise ValueError(
                f"horizon={self.horizon} exceeds times[-1]={self.times[-1]}"
            )


@dataclass(kw_only=True)
class CIFDiff(Estimand):
    r"""$F^{a_T}_j(t \mid x) - F^{a_R}_j(t \mid x)$ for competing-risks cause $j$."""

    contrast: ClassVar[str] = "cif_diff"
    times: ArrayLike
    cause: int

    def __post_init__(self) -> None:
        super().__post_init__()
        self.times = _validate_times(self.times)
        if self.cause < 1:
            raise ValueError(
                f"cause must be a positive integer (1-indexed cause label); "
                f"got {self.cause}"
            )


@dataclass(kw_only=True)
class CIFRatio(Estimand):
    r"""$F^{a_T}_j(t \mid x) / F^{a_R}_j(t \mid x)$ for competing-risks cause $j$."""

    contrast: ClassVar[str] = "cif_ratio"
    times: ArrayLike
    cause: int

    def __post_init__(self) -> None:
        super().__post_init__()
        self.times = _validate_times(self.times)
        if self.cause < 1:
            raise ValueError(
                f"cause must be a positive integer; got {self.cause}"
            )


_FROM_STRING: dict[str, type[Estimand]] = {
    "survival_diff": SurvivalDiff,
    "survival_ratio": SurvivalRatio,
    "rmst_diff": RMSTDiff,
    "cif_diff": CIFDiff,
    "cif_ratio": CIFRatio,
}


def resolve(
    estimand: "Estimand | type[Estimand] | str",
    **kwargs: Any,
) -> Estimand:
    """Coerce an estimand specification to an :class:`Estimand` instance.

    Three accepted forms:

    - **Instance**: returned as-is; extra ``kwargs`` are an error.
    - **Class object** (a subclass of :class:`Estimand`): instantiated
      with ``**kwargs``.
    - **String** (one of ``"survival_diff"``, ``"survival_ratio"``,
      ``"rmst_diff"``, ``"cif_diff"``, ``"cif_ratio"``): looked up to
      the corresponding class and instantiated with ``**kwargs``.

    Required estimand-specific parameters (``horizon`` for RMST,
    ``cause`` for CIF) are enforced by the dataclass — forgetting one
    raises a clear ``TypeError`` at construction.
    """
    if isinstance(estimand, Estimand):
        if kwargs:
            raise TypeError(
                "when `estimand` is a constructed Estimand instance, "
                "no other estimand-related kwargs are accepted; got "
                f"{sorted(kwargs)}"
            )
        return estimand
    if isinstance(estimand, type) and issubclass(estimand, Estimand):
        return estimand(**kwargs)
    if isinstance(estimand, str):
        cls = _FROM_STRING.get(estimand)
        if cls is None:
            raise ValueError(
                f"estimand must be one of {sorted(_FROM_STRING)} or an "
                f"Estimand class/instance; got string {estimand!r}"
            )
        return cls(**kwargs)
    raise TypeError(
        "estimand must be a string, an Estimand subclass, or an Estimand "
        f"instance; got {type(estimand).__name__}"
    )


def _validate_times(times: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(times, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(
            f"times must be a non-empty 1D sequence; got shape {arr.shape}"
        )
    if not np.all(np.diff(arr) > 0):
        raise ValueError("times must be strictly increasing")
    if arr[0] <= 0:
        raise ValueError(f"times must be positive; got times[0]={arr[0]}")
    return arr
