from __future__ import annotations

import math

import numpy as np
import pytest

from tausurv.metrics import scoring


def test_nll_event_uses_density_in_bin():
    # One event subject at Y=1; S = [1.0, 0.5, 0.0] at t_grid=[0.5, 1.0, 2.0].
    # t_idx for Y=1 is 1 (right-continuous step). S_prev = S[0] = 1.0, S_at_Y = 0.5.
    # density = 1.0 - 0.5 = 0.5; -log(0.5) = log(2).
    out = scoring.nll(
        event_time=[1.0],
        event_indicator=[1],
        survival=[[1.0, 0.5, 0.0]],
        times=[0.5, 1.0, 2.0],
    )
    assert out == pytest.approx(math.log(2.0))


def test_nll_censored_uses_survival_at_Y():
    # Censored at Y=1; S(Y=1) = 0.5. -log(0.5) = log(2).
    out = scoring.nll(
        event_time=[1.0],
        event_indicator=[0],
        survival=[[1.0, 0.5, 0.0]],
        times=[0.5, 1.0, 2.0],
    )
    assert out == pytest.approx(math.log(2.0))


def test_nll_perfect_predictions_low():
    # Two subjects: events at known times with peaked density at those bins.
    # Subject 0: event at t=1. S = [1, 0.001, 0]; density at bin 1 ~= 1.
    # Subject 1: event at t=2. S = [1, 1, 0.001]; density at bin 2 ~= 1.
    # NLL = -mean([log(1 - 0.001), log(1 - 0.001)]) ~ 0.001.
    out = scoring.nll(
        event_time=[1.0, 2.0],
        event_indicator=[1, 1],
        survival=[[1.0, 0.001, 0.0], [1.0, 1.0, 0.001]],
        times=[0.5, 1.0, 2.0],
    )
    assert out < 0.01


def test_crps_event_subject_hand_computed():
    # One event subject at Y=1 with S=[1, 0.5, 0] at t_grid=[0, 1, 2].
    # eta(t) = I[Y > t] = [1, 0, 0]. error_sq = [0, 0.25, 0].
    # upper_idx for event = last grid index = 2.
    # Both trapezoids included: 0.5*(0+0.25)*1 + 0.5*(0.25+0)*1 = 0.25.
    out = scoring.crps(
        event_time=[1.0],
        event_indicator=[1],
        survival=[[1.0, 0.5, 0.0]],
        times=[0.0, 1.0, 2.0],
    )
    assert out == pytest.approx(0.25)


def test_crps_censored_subject_truncates_at_Y():
    # Censored at Y=1.5; integrate only up to t_grid[1] = 1.
    # S=[1, 0.5, 0]. eta (assuming alive)=[1, 1, ?]. error_sq up to t=1: [0, 0.25].
    # upper_idx for Y=1.5: searchsorted right of 1.5 in [0,1,2] = 2; -1 = 1.
    # Include trapezoid (0,1): k+1=1 <= 1 ✓. Trapezoid value = 0.5*(0+0.25)*1 = 0.125.
    out = scoring.crps(
        event_time=[1.5],
        event_indicator=[0],
        survival=[[1.0, 0.5, 0.0]],
        times=[0.0, 1.0, 2.0],
    )
    assert out == pytest.approx(0.125)


def test_crps_perfect_prediction_is_zero():
    # If S(t) = 1 before event and 0 after, error is identically zero.
    out = scoring.crps(
        event_time=[1.0],
        event_indicator=[1],
        survival=[[1.0, 0.0, 0.0]],
        times=[0.0, 1.0, 2.0],
    )
    assert out == pytest.approx(0.0)


def test_crps_worst_prediction_integrates_squared_one():
    # Predict S(t) = 0 always; event subject with Y=2 gives eta=[1,1,0] at [0,1,2].
    # error_sq = [1, 1, 0]. Trapezoids: 0.5*(1+1)*1 + 0.5*(1+0)*1 = 1.0 + 0.5 = 1.5.
    out = scoring.crps(
        event_time=[2.0],
        event_indicator=[1],
        survival=[[0.0, 0.0, 0.0]],
        times=[0.0, 1.0, 2.0],
    )
    assert out == pytest.approx(1.5)
