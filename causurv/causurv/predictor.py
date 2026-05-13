r"""Causal-survival prediction contracts.

The single base class :class:`HTEPredictor` defines the contract every
causurv learner conforms to. Three estimands at three levels of
aggregation:

- :meth:`predict_potential_outcomes` — per-arm $S_a(t \mid x)$ or
  $F_{a, k}(t \mid x)$. Building block for the rest.
- :meth:`predict_hte` — per-subject treatment-effect contrast (CATE).
- :meth:`predict_ate` — population mean of the contrast (ATE).

Some learners only support a subset. Pure-ATE estimators
(e.g., TMLE-for-ATE) may override :meth:`predict_ate` directly and
raise on :meth:`predict_hte`; doubly-robust meta-learners support all
three. Each learner's docstring states which estimands it provides.

**Estimand glossary** (used in docstrings and the literature):

- **CATE** = Conditional Average Treatment Effect = $\mathbb{E}[Y(1) - Y(0) \mid X = x]$.
  In our API: :meth:`predict_hte`. CATE is a *function of x*.
- **ATE** = Average Treatment Effect = $\mathbb{E}[Y(1) - Y(0)]$, a
  *population scalar* (per cause × time). In our API: :meth:`predict_ate`.
- **HTE** = Heterogeneous Treatment Effect — used in this codebase as
  a synonym for CATE, following Curth (2021) and Frauen (2025).
- **ITE** = Individual Treatment Effect = $Y_i(1) - Y_i(0)$ for one
  subject. Generally non-identifiable; we do not pretend to estimate it.
  CATE is the closest we get under standard ignorability assumptions.

**Doubly-robust** is a *property of a learner*, not a method on this
contract. Learners that are DR document the claim in their class
docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from causurv.contrasts import apply_contrast
from causurv.estimands import resolve as resolve_estimand

if TYPE_CHECKING:
    from causurv.estimands import Estimand


@dataclass
class HTEEstimates:
    r"""Container for per-subject treatment-effect predictions.

    Self-describing: carries the result values *plus the* :class:`Estimand`
    that produced them, so any consumer can reconstruct the full
    estimand spec (contrast, time grid, horizon, cause, arms) from the
    result object alone.

    Attributes
    ----------
    values : ``(n, T)`` or ``(n,)`` array
        Per-subject effect. Pointwise across ``estimand.times`` for
        survival/cif contrasts → shape ``(n, T)``. Collapsed to a
        per-subject scalar for time-integrating contrasts like
        ``rmst_diff`` → shape ``(n,)``.
    estimand : :class:`Estimand`
        The specification that produced ``values``. Carries the
        contrast name, time grid, horizon (if any), cause (if any),
        and the arm comparison.

    Notes
    -----
    Most-used estimand attributes are exposed as @property shortcuts
    on this class — ``hte.contrast``, ``hte.treatment``,
    ``hte.reference``, ``hte.cause`` — so callers rarely need to
    reach through ``hte.estimand.*``.

    For the time axis, use ``hte.estimand.times``. Its semantics depend
    on the estimand: evaluation grid for survival/cif contrasts,
    integration grid for ``rmst_diff``. :attr:`is_time_collapsed`
    distinguishes the two.
    """

    values: NDArray[np.float64]
    estimand: "Estimand"

    @property
    def contrast(self) -> str:
        return self.estimand.contrast

    @property
    def treatment(self) -> int:
        return self.estimand.treatment

    @property
    def reference(self) -> int:
        return self.estimand.reference

    @property
    def cause(self) -> int | None:
        return getattr(self.estimand, "cause", None)

    @property
    def is_time_collapsed(self) -> bool:
        """``True`` when :attr:`values` is shape ``(n,)`` (e.g., RMST);
        ``False`` for pointwise-in-time results of shape ``(n, T)``."""
        return self.values.ndim == 1


class HTEPredictor:
    r"""Base class for causurv learners.

    Subclasses implement :meth:`fit` and :meth:`predict_potential_outcomes`;
    :meth:`predict_hte` and :meth:`predict_ate` are derived. Override the
    derived methods only when a learner has a tighter direct path (e.g.,
    a DR-learner that estimates the contrast directly without computing
    per-arm potential outcomes).

    Conventions
    -----------
    - The treatment vector ``A`` is always the 4th positional argument
      to :meth:`fit`, after ``(X, event_time, event_indicator)``.
    - ``A`` is integer-valued with ``A == 0`` always meaning *control*.
      Binary: ``A`` takes values 0 and 1. Multi-arm extends by adding
      values 2, 3, … with no API change.
    - ``treatment=1`` and ``reference=0`` in :meth:`predict_hte` /
      :meth:`predict_ate` make the binary case zero-config. Multi-arm
      callers pass the arms they want to contrast.
    - ``cause`` is for competing risks. ``None`` (default) means
      single-event; an integer ``k \ge 1`` selects cause $k$.
    """

    times_: NDArray[np.float64]

    # Required hooks subclasses implement.

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
        treatment: ArrayLike,
    ) -> "HTEPredictor":
        raise NotImplementedError

    def predict_potential_outcomes(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
        *,
        cause: int | None = None,
    ) -> tuple[NDArray[np.float64], ...]:
        r"""Per-arm conditional outcome on the chosen scale.

        Returns a tuple of arrays, one per treatment arm, each shaped
        ``(n, T)``. For survival-scale predictions ($\text{cause} =
        \text{None}$) the arrays are $S_a(t \mid x)$ values; for
        CIF-scale predictions ($\text{cause} = k$) they are
        $F_{a, k}(t \mid x)$ values.

        The tuple is indexed by treatment-label value: ``out[0]`` is
        the control arm, ``out[1]`` the first treatment, etc. The size
        of the tuple equals the number of arms the model was trained on.
        """
        raise NotImplementedError

    # Derived: callers can override for tighter paths (e.g., DR-learners).

    def predict_hte(
        self,
        X: ArrayLike,
        *,
        estimand: "Estimand | type[Estimand] | str",
        **kwargs: Any,
    ) -> HTEEstimates:
        r"""Conditional Average Treatment Effect (CATE) per subject.

        Parameters
        ----------
        X : ``(n, d)`` array
        estimand : :class:`Estimand`, Estimand subclass, or string
            What to estimate. Accepted forms:

            - **Instance**: ``estimand=SurvivalDiff(times=[5, 10])``
            - **Class + kwargs**: ``estimand=SurvivalDiff, times=[5, 10]``
            - **String + kwargs**: ``estimand="survival_diff", times=[5, 10]``

            See :mod:`causurv.estimands`. Estimand-specific required
            parameters (``horizon`` for RMST, ``cause`` for CIF) are
            enforced by the dataclass at construction.
        **kwargs
            Forwarded to the :class:`Estimand` constructor when
            ``estimand`` is a string or class object.

        Returns
        -------
        :class:`HTEEstimates`
        """
        est = resolve_estimand(estimand, **kwargs)
        return self._predict_hte_impl(X, est)

    def _predict_hte_impl(
        self, X: ArrayLike, estimand: "Estimand"
    ) -> HTEEstimates:
        """Compute the HTE from a resolved :class:`Estimand`.

        Override this in subclasses that need direct-CATE estimation
        (R-learner, DR-learner / OrthoLearner) — :meth:`predict_hte`
        handles input resolution and delegates here.
        """
        times = np.asarray(estimand.times, dtype=np.float64)
        cause = getattr(estimand, "cause", None)
        arms = self.predict_potential_outcomes(X, times, cause=cause)
        if estimand.treatment >= len(arms) or estimand.reference >= len(arms):
            raise ValueError(
                f"treatment={estimand.treatment} or reference="
                f"{estimand.reference} out of range; model has "
                f"{len(arms)} arms (0..{len(arms) - 1})"
            )
        horizon = getattr(estimand, "horizon", None)
        values = apply_contrast(
            arms[estimand.reference],
            arms[estimand.treatment],
            estimand.contrast,
            times=times,
            horizon=horizon,
        )
        return HTEEstimates(values=values, estimand=estimand)

    def predict_ate(
        self,
        X: ArrayLike | None = None,
        *,
        estimand: "Estimand | type[Estimand] | str",
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        r"""Average Treatment Effect (ATE).

        Mean of :meth:`predict_hte` over ``X``. If ``X`` is ``None``,
        uses the training set (subclasses retain it via
        ``self._fit_X``).
        """
        est = resolve_estimand(estimand, **kwargs)
        if X is None:
            X = self._stored_X()
        hte = self._predict_hte_impl(X, est)
        return hte.values.mean(axis=0)

    _fit_X: NDArray[np.float64] | None = None

    def _resolve_times(self, times: ArrayLike | None) -> NDArray[np.float64]:
        if times is not None:
            return np.asarray(times, dtype=np.float64)
        if not hasattr(self, "times_"):
            raise RuntimeError(
                f"{type(self).__name__} has no times_; pass `times` "
                f"explicitly to predict_* methods, or fit the model first."
            )
        return np.asarray(self.times_, dtype=np.float64)

    def _stored_X(self) -> NDArray[np.float64]:
        """Return the cached training $X$ for :meth:`predict_ate`.

        Learners that train via :meth:`fit` assign ``self._fit_X = X``
        and inherit this implementation. Override only when the cached
        $X$ has different provenance (e.g., the SurvITE simulator caches
        the last generated $X$ instead).
        """
        if self._fit_X is None:
            raise RuntimeError(
                f"{type(self).__name__}.predict_ate without `X` requires "
                f"the learner to have been fit; either pass `X` or call "
                f"fit() first."
            )
        return self._fit_X


def _validate_fit_inputs(
    X: ArrayLike,
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    treatment: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int_],
    NDArray[np.int_],
]:
    r"""Standard fit-time validation for ``(X, T, E, A)`` inputs.

    Coerces ``X`` and ``event_time`` to ``float64`` and ``treatment``
    to a contiguous integer dtype; ``event_indicator`` is coerced to an
    integer array but the caller's narrower dtype (e.g., ``int8``) is
    preserved when valid. Validates:

    - ``X`` is 2D.
    - ``event_time``, ``event_indicator``, ``treatment`` share the first
      axis with ``X``.
    - ``treatment`` is non-negative.
    - Treatment labels are contiguous ``0..K-1`` — every arm in that
      range has at least one subject. Sparse arm IDs (``{0, 2}``) would
      cause downstream learners to produce silently wrong outputs
      because they index arms positionally; we reject up front.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    T_arr = np.asarray(event_time, dtype=np.float64)
    E_arr = np.asarray(event_indicator)
    A_arr = np.asarray(treatment, dtype=np.int_)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D (n, d); got shape {X_arr.shape}")
    n = X_arr.shape[0]
    if not (T_arr.shape == (n,) and E_arr.shape == (n,) and A_arr.shape == (n,)):
        raise ValueError(
            "X, event_time, event_indicator, and treatment must share "
            f"first axis; got {X_arr.shape}, {T_arr.shape}, "
            f"{E_arr.shape}, {A_arr.shape}"
        )
    if A_arr.min() < 0:
        raise ValueError(
            f"treatment values must be non-negative integers; "
            f"min was {A_arr.min()}"
        )
    n_arms = int(A_arr.max()) + 1
    counts = np.bincount(A_arr, minlength=n_arms)
    if (counts == 0).any():
        missing = [int(a) for a in range(n_arms) if counts[a] == 0]
        raise ValueError(
            f"treatment labels must be contiguous integers 0..{n_arms - 1}; "
            f"arms {missing} have no subjects"
        )
    return X_arr, T_arr, E_arr, A_arr
