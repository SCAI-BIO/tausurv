from __future__ import annotations

import numpy as np
import pytest

from tausurv.metrics import auc


@pytest.fixture
def random_auc_data(random_survival_data):
    event_time, event_indicator, risk_score = random_survival_data
    t_max = float(np.quantile(event_time, 0.9))
    time_grid = np.linspace(0.1, t_max, 15)
    return event_time, event_indicator, risk_score, time_grid


def test_uno_auc_matches_sksurv(random_auc_data):
    sksurv_metrics = pytest.importorskip("sksurv.metrics")
    event_time, event_indicator, risk_score, time_grid = random_auc_data

    structured = np.array(
        list(zip(event_indicator.astype(bool), event_time)),
        dtype=[("event", bool), ("time", float)],
    )

    ours = auc.uno(event_time, event_indicator, risk_score, time_grid)
    theirs_per_time, _ = sksurv_metrics.cumulative_dynamic_auc(
        structured, structured, risk_score, time_grid
    )
    np.testing.assert_allclose(ours, theirs_per_time, equal_nan=True)


def test_blanche_auc_matches_sksurv_with_time_varying_marker(random_auc_data):
    sksurv_metrics = pytest.importorskip("sksurv.metrics")
    event_time, event_indicator, risk_score, time_grid = random_auc_data

    # Make a synthetic time-varying marker for Blanche.
    marker = risk_score[:, None] + 0.1 * np.arange(len(time_grid))[None, :]

    structured = np.array(
        list(zip(event_indicator.astype(bool), event_time)),
        dtype=[("event", bool), ("time", float)],
    )

    ours = auc.blanche(event_time, event_indicator, marker, time_grid)
    theirs_per_time, _ = sksurv_metrics.cumulative_dynamic_auc(
        structured, structured, marker, time_grid
    )
    np.testing.assert_allclose(ours, theirs_per_time, equal_nan=True)


def test_integrated_auc_close_to_sksurv(random_auc_data):
    # Both implementations follow the Heagerty-Zheng survival-weighted
    # definition, but the discrete approximation on a finite grid differs
    # slightly between libraries (sksurv-specific binning details that aren't
    # part of the published formula). Per-time AUC matches exactly above;
    # this test asserts the integrated values agree to ~0.1%.
    sksurv_metrics = pytest.importorskip("sksurv.metrics")
    event_time, event_indicator, risk_score, time_grid = random_auc_data

    structured = np.array(
        list(zip(event_indicator.astype(bool), event_time)),
        dtype=[("event", bool), ("time", float)],
    )

    our_per_time = auc.uno(event_time, event_indicator, risk_score, time_grid)
    our_mean = auc.integrated(event_time, event_indicator, our_per_time, time_grid)
    _, their_mean = sksurv_metrics.cumulative_dynamic_auc(
        structured, structured, risk_score, time_grid
    )
    assert our_mean == pytest.approx(their_mean, rel=1e-3)
