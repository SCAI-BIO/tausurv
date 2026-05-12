from __future__ import annotations

import numpy as np
import pytest

from tausurv.metrics import auc
from tausurv.step import StepFunction


def test_uno_perfect_concordance_no_censoring():
    # At t=2.5: cases = {Y=1, Y=2}, controls = {Y=3, Y=4}; risk_score ranks
    # cases above controls -> every pair concordant.
    out = auc.uno(
        event_time=[1, 2, 3, 4],
        event_indicator=[1, 1, 1, 1],
        risk_score=[4, 3, 2, 1],
        time_grid=[2.5],
    )
    np.testing.assert_array_equal(out, [1.0])


def test_uno_perfect_anticoncordance():
    out = auc.uno(
        event_time=[1, 2, 3, 4],
        event_indicator=[1, 1, 1, 1],
        risk_score=[1, 2, 3, 4],
        time_grid=[2.5],
    )
    np.testing.assert_array_equal(out, [0.0])


def test_uno_all_tied_is_half():
    out = auc.uno(
        event_time=[1, 2, 3, 4],
        event_indicator=[1, 1, 1, 1],
        risk_score=[1, 1, 1, 1],
        time_grid=[2.5],
    )
    np.testing.assert_array_equal(out, [0.5])


def test_uno_returns_nan_when_no_cases_or_controls():
    # t=0.5: no events yet (no cases). t=5.0: everyone has had event (no controls).
    out = auc.uno(
        event_time=[1, 2, 3, 4],
        event_indicator=[1, 1, 1, 1],
        risk_score=[1, 2, 3, 4],
        time_grid=[0.5, 5.0],
    )
    assert np.isnan(out[0])
    assert np.isnan(out[1])


def test_uno_custom_censoring_survival_overrides_default():
    # Constant Ĝ scales every case weight equally; ratio unchanged.
    custom_G = StepFunction(time=np.array([0.5]), value=np.array([0.5]))
    out = auc.uno(
        event_time=[1, 2, 3, 4],
        event_indicator=[1, 1, 1, 1],
        risk_score=[4, 3, 2, 1],
        time_grid=[2.5],
        censoring_survival=custom_G,
    )
    np.testing.assert_array_equal(out, [1.0])


def test_blanche_perfect_concordance():
    out = auc.blanche(
        event_time=[1, 2, 3, 4],
        event_indicator=[1, 1, 1, 1],
        marker=np.array([[4], [3], [2], [1]]),
        time_grid=[2.5],
    )
    np.testing.assert_array_equal(out, [1.0])


def test_blanche_reduces_to_uno_for_constant_marker():
    # When marker is constant in t, Blanche should match Uno exactly.
    rng = np.random.default_rng(0)
    n = 60
    risk = rng.uniform(0.0, 1.0, n)
    event_time = rng.uniform(0.5, 5.0, n)
    event_indicator = rng.integers(0, 2, n)
    time_grid = np.linspace(0.5, 4.0, 5)
    marker = np.broadcast_to(risk[:, None], (n, len(time_grid)))

    uno_auc = auc.uno(event_time, event_indicator, risk, time_grid)
    blanche_auc = auc.blanche(event_time, event_indicator, marker, time_grid)
    np.testing.assert_allclose(blanche_auc, uno_auc, equal_nan=True)


def test_integrated_auc_heagerty_zheng_hand_computed():
    # delta=[1,1,1,1] -> KM at unique event times = [3/4, 1/2, 1/4, 0].
    # S evaluated right-continuous at time_grid [1.5, 2.5, 3.5] = [3/4, 1/2, 1/4].
    # drops = [1/4, 1/4]; auc midpoints = [(0 + 0.5)/2, (0.5 + 1/3)/2] = [0.25, 5/12].
    # weighted sum = 1/4 * 1/4 + 1/4 * 5/12 = 1/16 + 5/48 = 8/48 = 1/6.
    # total drop = 3/4 - 1/4 = 1/2. Result = (1/6) / (1/2) = 1/3.
    out = auc.integrated(
        event_time=[1, 2, 3, 4],
        event_indicator=[1, 1, 1, 1],
        auc_per_time=[0.0, 0.5, 1 / 3],
        time_grid=[1.5, 2.5, 3.5],
    )
    assert out == pytest.approx(1 / 3)


def test_integrated_auc_rejects_single_point_grid():
    with pytest.raises(ValueError, match="at least 2"):
        auc.integrated([1.0], [1.0], [0.7], [1.0])


def test_integrated_auc_rejects_no_events_in_range():
    # All censored -> S stays at 1 across the grid -> total_drop = 0.
    with pytest.raises(ValueError, match="no marginal events"):
        auc.integrated(
            event_time=[1, 2, 3],
            event_indicator=[0, 0, 0],
            auc_per_time=[0.5, 0.5, 0.5],
            time_grid=[1.0, 2.0, 3.0],
        )
