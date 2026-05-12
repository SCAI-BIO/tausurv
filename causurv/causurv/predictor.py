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

import numpy as np
from numpy.typing import ArrayLike, NDArray

from causurv.contrasts import apply_contrast, is_cif_scale, is_survival_scale


@dataclass
class HTEEstimates:
    r"""Container for per-subject treatment-effect predictions.

    Carries the contrast name, treatment/reference arm labels, time
    grid, and optional CR cause so the predictions are self-describing.

    Attributes
    ----------
    values : ``(n, T)`` or ``(n,)`` array
        Per-subject effect on the chosen contrast. Pointwise across
        ``times`` for survival/cif contrasts; scalar per subject for
        contrasts that integrate over time (``rmst_diff``).
    contrast : str
        The contrast name (e.g., ``"survival_diff"``).
    treatment, reference : int
        The arm comparison: ``arm_treatment`` vs ``arm_reference``.
    times : ``(T,)`` array or ``None``
        Time grid the values are evaluated at. ``None`` for time-collapsed
        contrasts like ``rmst_diff``.
    cause : int or ``None``
        Competing-risks cause (1-indexed). ``None`` for single-event.
    """

    values: NDArray[np.float64]
    contrast: str
    treatment: int
    reference: int
    times: NDArray[np.float64] | None = None
    cause: int | None = None


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
        times: ArrayLike | None = None,
        *,
        contrast: str = "survival_diff",
        cause: int | None = None,
        treatment: int = 1,
        reference: int = 0,
        horizon: float | None = None,
    ) -> HTEEstimates:
        r"""Conditional Average Treatment Effect (CATE) per subject.

        Resolves the requested contrast against
        :meth:`predict_potential_outcomes` and packages the result with
        metadata. See :mod:`causurv.contrasts` for the contrast vocabulary.

        Some learners override this with a direct CATE estimator
        (R-learner, DR-learner) — in that case the override should still
        return an :class:`HTEEstimates`.
        """
        # Caller-facing validation: cause must match the contrast's scale.
        if is_cif_scale(contrast) and cause is None:
            raise ValueError(
                f"contrast={contrast!r} requires `cause` (an integer cause "
                f"index); got cause=None"
            )
        if is_survival_scale(contrast) and cause is not None:
            raise ValueError(
                f"contrast={contrast!r} is a survival-scale contrast and "
                f"cannot take `cause` (got cause={cause}); use a cif_* "
                f"contrast for cause-specific effects"
            )

        times_arr = self._resolve_times(times)
        arms = self.predict_potential_outcomes(X, times_arr, cause=cause)
        if treatment >= len(arms) or reference >= len(arms):
            raise ValueError(
                f"treatment={treatment} or reference={reference} out of "
                f"range; model has {len(arms)} arms (0..{len(arms) - 1})"
            )

        values = apply_contrast(
            arms[reference],
            arms[treatment],
            contrast,
            times=times_arr,
            horizon=horizon,
        )
        # rmst_diff collapses the time axis; everything else keeps it.
        out_times = None if contrast == "rmst_diff" else times_arr
        return HTEEstimates(
            values=values,
            contrast=contrast,
            treatment=treatment,
            reference=reference,
            times=out_times,
            cause=cause,
        )

    def predict_ate(
        self,
        X: ArrayLike | None = None,
        times: ArrayLike | None = None,
        *,
        contrast: str = "survival_diff",
        cause: int | None = None,
        treatment: int = 1,
        reference: int = 0,
        horizon: float | None = None,
    ) -> NDArray[np.float64]:
        r"""Average Treatment Effect (ATE).

        Default: the mean of :meth:`predict_hte`'s output over ``X``.
        If ``X`` is ``None``, uses the training set (subclasses must
        retain it). Subclasses with a more efficient direct path (e.g.,
        AIPW or TMLE for the ATE) should override.
        """
        if X is None:
            X = self._stored_X()
        hte = self.predict_hte(
            X,
            times,
            contrast=contrast,
            cause=cause,
            treatment=treatment,
            reference=reference,
            horizon=horizon,
        )
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
