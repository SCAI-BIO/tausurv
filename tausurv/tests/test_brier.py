from __future__ import annotations

import numpy as np
import pytest

from tausurv.metrics import brier
from tausurv.step import StepFunction


def test_brier_single_subject_no_censoring():
    # No censoring -> G(t) = 1 everywhere -> weights are 1.
    # At t=1: Y=2 > 1, subject is at-risk control. err = (1 - 0.8)^2 = 0.04.
    # At t=3: Y=2 <= 3 and delta=1, subject is case.  err = 0.5^2 = 0.25.
    out = brier.score(
        event_time=[2.0],
        event_indicator=[1],
        survival=[[0.8, 0.5]],
        time_grid=[1.0, 3.0],
    )
    np.testing.assert_allclose(out, [0.04, 0.25])


def test_brier_perfect_predictions_is_zero():
    # Subject 0 has event by t=2 (true survival=0); predict 0.
    # Subject 1 alive past t=2 (true survival=1); predict 1.
    out = brier.score(
        event_time=[1.0, 3.0],
        event_indicator=[1, 1],
        survival=[[0.0], [1.0]],
        time_grid=[2.0],
    )
    np.testing.assert_array_equal(out, [0.0])


def test_brier_worst_predictions_is_one_no_censoring():
    # Same as above but predict the opposite -> per-subject error 1 each,
    # averaged over n=2 -> 1.
    out = brier.score(
        event_time=[1.0, 3.0],
        event_indicator=[1, 1],
        survival=[[1.0], [0.0]],
        time_grid=[2.0],
    )
    np.testing.assert_array_equal(out, [1.0])


def test_brier_drops_censored_before_t():
    # Subject 0 censored at t=1 (before grid point 2) — contributes 0.
    # Subject 1 alive past t=2 (control), predicted 0.5 -> err = 0.25.
    # Average over n=2 -> 0.125.
    out = brier.score(
        event_time=[1.0, 3.0],
        event_indicator=[0, 1],
        survival=[[0.9], [0.5]],
        time_grid=[2.0],
    )
    # G computed from KM on (Y=[1,3], 1-delta=[1,0]):
    #   time=[1,3], value=[0.5, 0.5].
    # G(2, side="right") = value at largest time <= 2 = G(1) = 0.5.
    # control weight = 1 / 0.5 = 2. Subject 1 control contribution = 2 * 0.25 = 0.5.
    # Sum / n = 0.5 / 2 = 0.25.
    np.testing.assert_allclose(out, [0.25])


def test_brier_custom_censoring_survival_overrides_default():
    # Constant Ĝ = 0.5 doubles every weight; both case and control errors scale.
    custom_G = StepFunction(time=np.array([0.5]), value=np.array([0.5]))
    out = brier.score(
        event_time=[2.0],
        event_indicator=[1],
        survival=[[0.8, 0.5]],
        time_grid=[1.0, 3.0],
        censoring_survival=custom_G,
    )
    # At t=1: control err = 0.04, weight = 1/0.5 = 2 -> 0.08. /n=1 -> 0.08.
    # At t=3: case   err = 0.25, weight = 1/0.5 = 2 -> 0.50. /n=1 -> 0.50.
    np.testing.assert_allclose(out, [0.08, 0.50])


def test_integrated_brier_matches_hand_computed():
    # Reuses the single-subject case above: BS([1, 3]) = [0.04, 0.25].
    # Trapezoidal integration: (0.04 + 0.25) / 2 * (3 - 1) = 0.29.
    # Normalized by span 2 -> 0.145.
    ibs = brier.integrated(
        event_time=[2.0],
        event_indicator=[1],
        survival=[[0.8, 0.5]],
        time_grid=[1.0, 3.0],
    )
    assert ibs == pytest.approx(0.145)


def test_integrated_brier_rejects_degenerate_grid():
    with pytest.raises(ValueError, match="positive interval"):
        brier.integrated(
            event_time=[1.0],
            event_indicator=[1],
            survival=[[0.5]],
            time_grid=[1.0],
        )
