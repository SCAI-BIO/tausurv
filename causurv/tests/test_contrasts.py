"""Tests for the contrast vocabulary."""

from __future__ import annotations

import numpy as np
import pytest

from causurv.contrasts import apply_contrast, is_cif_scale, is_survival_scale


def _arms():
    # Two 3-subject, 4-time arrays.
    s_ref = np.array(
        [
            [1.0, 0.9, 0.7, 0.5],
            [1.0, 0.8, 0.6, 0.3],
            [1.0, 0.95, 0.85, 0.7],
        ]
    )
    s_treat = np.array(
        [
            [1.0, 0.95, 0.85, 0.7],
            [1.0, 0.9, 0.75, 0.55],
            [1.0, 0.97, 0.9, 0.8],
        ]
    )
    return s_ref, s_treat


def test_survival_diff_is_arithmetic():
    s_ref, s_treat = _arms()
    out = apply_contrast(s_ref, s_treat, "survival_diff")
    np.testing.assert_array_equal(out, s_treat - s_ref)


def test_survival_ratio_is_arithmetic_with_zero_protection():
    s_ref, s_treat = _arms()
    out = apply_contrast(s_ref, s_treat, "survival_ratio")
    np.testing.assert_allclose(out, s_treat / s_ref)

    # At a zero reference, we should still get a finite ratio (not NaN/inf).
    s_ref0 = s_ref.copy()
    s_ref0[0, 3] = 0.0
    out = apply_contrast(s_ref0, s_treat, "survival_ratio")
    assert np.isfinite(out).all()


def test_cif_diff_and_ratio_are_just_arithmetic_too():
    f_ref = np.array([[0.0, 0.1, 0.3, 0.5]])
    f_treat = np.array([[0.0, 0.05, 0.15, 0.3]])
    np.testing.assert_array_equal(
        apply_contrast(f_ref, f_treat, "cif_diff"), f_treat - f_ref
    )
    out = apply_contrast(f_ref, f_treat, "cif_ratio")
    # First column has F_ref=0, hitting the clip; check non-degenerate cols.
    expected = f_treat[:, 1:] / f_ref[:, 1:]
    np.testing.assert_allclose(out[:, 1:], expected)


def test_rmst_diff_integrates_correctly_on_a_known_case():
    # Constant gap: arm_treat = arm_ref + 0.1 everywhere → integral over
    # [0, 4] = 0.1 * 4 = 0.4.
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    s_ref = np.array([[1.0, 0.8, 0.6, 0.4, 0.2]])
    s_treat = s_ref + 0.1
    out = apply_contrast(
        s_ref, s_treat, "rmst_diff", times=times, horizon=4.0
    )
    np.testing.assert_allclose(out, [0.4], atol=1e-12)


def test_rmst_diff_truncates_at_horizon():
    """The integral runs only up to ``horizon``, even if ``times``
    extends further."""
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    s_ref = np.array([[1.0, 0.8, 0.6, 0.4, 0.2]])
    s_treat = s_ref + 0.2
    full = apply_contrast(s_ref, s_treat, "rmst_diff", times=times, horizon=4.0)
    half = apply_contrast(s_ref, s_treat, "rmst_diff", times=times, horizon=2.0)
    np.testing.assert_allclose(full, [0.8], atol=1e-12)
    np.testing.assert_allclose(half, [0.4], atol=1e-12)


def test_rmst_diff_requires_horizon_and_times():
    s_ref, s_treat = _arms()
    with pytest.raises(ValueError, match="horizon"):
        apply_contrast(s_ref, s_treat, "rmst_diff")


def test_rmst_diff_rejects_horizon_above_times_max():
    times = np.array([0.0, 1.0, 2.0])
    s_ref = np.array([[1.0, 0.5, 0.0]])
    s_treat = np.array([[1.0, 0.6, 0.0]])
    with pytest.raises(ValueError, match="exceeds max time"):
        apply_contrast(s_ref, s_treat, "rmst_diff", times=times, horizon=10.0)


def test_unknown_contrast_raises():
    s_ref, s_treat = _arms()
    with pytest.raises(ValueError, match="contrast must be one of"):
        apply_contrast(s_ref, s_treat, "zzz")


def test_scale_dispatch_helpers():
    for c in ("survival_diff", "survival_ratio", "rmst_diff"):
        assert is_survival_scale(c) and not is_cif_scale(c), c
    for c in ("cif_diff", "cif_ratio"):
        assert is_cif_scale(c) and not is_survival_scale(c), c
