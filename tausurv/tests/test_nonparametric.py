from __future__ import annotations

import numpy as np
import pytest

from tausurv import nonparametric


def test_kaplan_meier_all_events_distinct():
    km = nonparametric.kaplan_meier([1, 2, 3], [1, 1, 1])
    np.testing.assert_array_equal(km.time, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(km.value, [2 / 3, 1 / 3, 0])


def test_kaplan_meier_all_censored_stays_at_one():
    km = nonparametric.kaplan_meier([1, 2, 3], [0, 0, 0])
    np.testing.assert_array_equal(km.time, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(km.value, [1, 1, 1])


def test_kaplan_meier_tied_event_times():
    km = nonparametric.kaplan_meier([1, 1, 2], [1, 1, 1])
    np.testing.assert_array_equal(km.time, [1.0, 2.0])
    np.testing.assert_allclose(km.value, [1 / 3, 0])


def test_kaplan_meier_censoring_does_not_drop_survival():
    km = nonparametric.kaplan_meier([1, 2, 3], [1, 0, 1])
    np.testing.assert_array_equal(km.time, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(km.value, [2 / 3, 2 / 3, 0])


def test_kaplan_meier_unsorted_input():
    km = nonparametric.kaplan_meier([3, 1, 2], [1, 1, 1])
    np.testing.assert_array_equal(km.time, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(km.value, [2 / 3, 1 / 3, 0])


def test_kaplan_meier_callable_at_arbitrary_times():
    # The returned StepFunction can be evaluated at any times, not just
    # the unique observation grid.
    km = nonparametric.kaplan_meier([1, 2, 3], [1, 1, 1])
    np.testing.assert_allclose(km([0.5, 1.5, 2.5, 5.0]), [1.0, 2 / 3, 1 / 3, 0])


def test_censoring_distribution_swaps_event_indicator():
    # delta=[1,0,1,0,0] -> reverse-KM treats the three censorings as events.
    # Equivalent to kaplan_meier on 1-delta = [0,1,0,1,1].
    event_time = [1, 2, 3, 4, 5]
    event_indicator = [1, 0, 1, 0, 0]
    G = nonparametric.censoring_distribution(event_time, event_indicator)
    expected = nonparametric.kaplan_meier(event_time, [0, 1, 0, 1, 1])
    np.testing.assert_array_equal(G.time, expected.time)
    np.testing.assert_allclose(G.value, expected.value)


def test_nelson_aalen_hand_computed():
    # n_at_risk = [3, 2, 1]; d = [1, 1, 1].
    # increments = [1/3, 1/2, 1]; cumsum = [1/3, 5/6, 11/6].
    na = nonparametric.nelson_aalen([1, 2, 3], [1, 1, 1])
    np.testing.assert_array_equal(na.time, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(na.value, [1 / 3, 5 / 6, 11 / 6])
    assert na.baseline == 0.0


def test_nelson_aalen_no_events_stays_at_zero():
    na = nonparametric.nelson_aalen([1, 2, 3], [0, 0, 0])
    np.testing.assert_array_equal(na.value, [0.0, 0.0, 0.0])


def test_aalen_johansen_single_cause_complements_km():
    # With a single cause, CIF = 1 - KM. Cross-check.
    event_time = [1, 2, 3, 4]
    event_indicator = [1, 1, 0, 1]
    cif = nonparametric.aalen_johansen(event_time, event_indicator, cause=1)
    km = nonparametric.kaplan_meier(event_time, event_indicator)
    np.testing.assert_array_equal(cif.time, km.time)
    np.testing.assert_allclose(cif.value, 1.0 - km.value)


def test_aalen_johansen_competing_risks_hand_computed():
    # Y=[1,2,3], E=[1,2,1]; n_at_risk=[3,2,1]; S=[2/3, 1/3, 0]; S_left=[1, 2/3, 1/3].
    # Cause 1 d_k=[1,0,1]; CIF_1 increments = [1*1/3, 2/3*0, 1/3*1] = [1/3, 0, 1/3].
    # Cause 2 d_k=[0,1,0]; CIF_2 increments = [0, 2/3*1/2, 0] = [0, 1/3, 0].
    cif1 = nonparametric.aalen_johansen([1, 2, 3], [1, 2, 1], cause=1)
    cif2 = nonparametric.aalen_johansen([1, 2, 3], [1, 2, 1], cause=2)
    np.testing.assert_allclose(cif1.value, [1 / 3, 1 / 3, 2 / 3])
    np.testing.assert_allclose(cif2.value, [0.0, 1 / 3, 1 / 3])
    # CIF_1 + CIF_2 = 1 - KM(any-cause).
    km_any = nonparametric.kaplan_meier([1, 2, 3], [1, 1, 1])
    np.testing.assert_allclose(cif1.value + cif2.value, 1.0 - km_any.value)


def test_aalen_johansen_rejects_nonpositive_cause():
    with pytest.raises(ValueError, match="cause"):
        nonparametric.aalen_johansen([1, 2], [1, 1], cause=0)
