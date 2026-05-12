from __future__ import annotations

import numpy as np
import pytest

from tausurv.step import StepFunction


def test_step_function_right_continuous_default():
    sf = StepFunction(
        time=np.array([1.0, 2.0, 3.0]),
        value=np.array([0.8, 0.5, 0.2]),
    )
    np.testing.assert_array_equal(
        sf([0.5, 1.0, 1.5, 2.0, 3.5]),
        [1.0, 0.8, 0.8, 0.5, 0.2],
    )


def test_step_function_left_continuous_override():
    sf = StepFunction(
        time=np.array([1.0, 2.0, 3.0]),
        value=np.array([0.8, 0.5, 0.2]),
    )
    # At t=1.0 the left limit is the baseline (before the drop);
    # at t=2.0 the left limit is 0.8.
    np.testing.assert_array_equal(
        sf([0.5, 1.0, 1.5, 2.0, 3.5], side="left"),
        [1.0, 1.0, 0.8, 0.8, 0.2],
    )


def test_step_function_baseline_for_cif():
    # CIF starts at 0 below the first event time.
    cif = StepFunction(
        time=np.array([1.0, 2.0]),
        value=np.array([0.3, 0.5]),
        baseline=0.0,
    )
    np.testing.assert_array_equal(cif([0.5, 1.0, 2.0]), [0.0, 0.3, 0.5])


def test_step_function_rejects_unknown_side():
    sf = StepFunction(time=np.array([1.0]), value=np.array([0.5]))
    with pytest.raises(ValueError, match="side"):
        sf([0.5], side="middle")


def test_step_function_is_immutable():
    sf = StepFunction(time=np.array([1.0]), value=np.array([0.5]))
    with pytest.raises(AttributeError):
        sf.value = np.array([0.1])
