"""Tests for the Estimand types and the resolver.

Each estimand dataclass enforces its own required parameters at
construction (no separate runtime check needed). The resolver accepts
instance, class+kwargs, or string+kwargs forms — same dataclass
constructor under all three.
"""

from __future__ import annotations

import numpy as np
import pytest

from causurv.estimands import (
    CIFDiff,
    CIFRatio,
    Estimand,
    RMSTDiff,
    SurvivalDiff,
    SurvivalRatio,
    resolve,
)


def test_survival_diff_basic_construction():
    e = SurvivalDiff(times=[5, 10, 20])
    assert e.contrast == "survival_diff"
    assert e.treatment == 1 and e.reference == 0
    np.testing.assert_array_equal(e.times, np.array([5.0, 10.0, 20.0]))


def test_survival_ratio_construction():
    e = SurvivalRatio(times=[1, 2, 3])
    assert e.contrast == "survival_ratio"


def test_rmst_diff_requires_horizon():
    with pytest.raises(TypeError, match="horizon"):
        RMSTDiff(times=[1, 2, 3])  # type: ignore[call-arg]


def test_rmst_diff_horizon_must_be_positive():
    with pytest.raises(ValueError, match="horizon must be > 0"):
        RMSTDiff(times=[1, 2, 3], horizon=0.0)


def test_rmst_diff_horizon_within_times_range():
    with pytest.raises(ValueError, match="exceeds times"):
        RMSTDiff(times=[1, 2, 3], horizon=5.0)


def test_cif_diff_requires_cause():
    with pytest.raises(TypeError, match="cause"):
        CIFDiff(times=[1, 2, 3])  # type: ignore[call-arg]


def test_cif_diff_cause_must_be_positive():
    with pytest.raises(ValueError, match="positive integer"):
        CIFDiff(times=[1, 2, 3], cause=0)


def test_cif_ratio_requires_cause():
    with pytest.raises(TypeError, match="cause"):
        CIFRatio(times=[1, 2, 3])  # type: ignore[call-arg]


def test_treatment_and_reference_must_differ():
    with pytest.raises(ValueError, match="must differ"):
        SurvivalDiff(times=[1], treatment=0, reference=0)


def test_times_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        SurvivalDiff(times=[5, 5, 10])
    with pytest.raises(ValueError, match="strictly increasing"):
        SurvivalDiff(times=[10, 5])


def test_times_must_be_positive():
    with pytest.raises(ValueError, match="times must be positive"):
        SurvivalDiff(times=[0, 5])
    with pytest.raises(ValueError, match="times must be positive"):
        SurvivalDiff(times=[-1, 5])


def test_times_must_be_non_empty():
    with pytest.raises(ValueError, match="non-empty"):
        SurvivalDiff(times=[])


def test_resolve_passes_instance_through():
    e = SurvivalDiff(times=[1, 2])
    assert resolve(e) is e


def test_resolve_instance_with_extra_kwargs_raises():
    e = SurvivalDiff(times=[1, 2])
    with pytest.raises(TypeError, match="no other estimand-related kwargs"):
        resolve(e, times=[3, 4])


def test_resolve_class_with_kwargs():
    e = resolve(SurvivalDiff, times=[5, 10])
    assert isinstance(e, SurvivalDiff)
    np.testing.assert_array_equal(e.times, [5.0, 10.0])


def test_resolve_string_with_kwargs():
    e = resolve("rmst_diff", times=[1, 2, 3, 4, 5], horizon=4.0)
    assert isinstance(e, RMSTDiff)
    assert e.horizon == 4.0


def test_resolve_unknown_string_raises():
    with pytest.raises(ValueError, match="must be one of"):
        resolve("nonsense_diff", times=[1])


def test_resolve_unknown_type_raises():
    with pytest.raises(TypeError, match="must be a string"):
        resolve(42, times=[1])  # type: ignore[arg-type]


def test_estimand_subclasses_are_kw_only():
    """Constructed with positional args should fail."""
    with pytest.raises(TypeError):
        SurvivalDiff([5, 10])  # type: ignore[misc]


def test_estimand_base_is_not_meant_to_be_instantiated():
    """Estimand has no `contrast` ClassVar populated, so subclasses
    must set one. The base itself has no `times` field."""
    # Instantiating the bare base shouldn't be useful; ensure the
    # ClassVar is unset.
    assert not hasattr(Estimand, "contrast") or not isinstance(
        getattr(Estimand, "contrast", None), str
    )
