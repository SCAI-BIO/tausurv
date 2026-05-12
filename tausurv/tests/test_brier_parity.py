from __future__ import annotations

import numpy as np
import pytest

from tausurv.metrics import brier


@pytest.fixture
def random_brier_data(random_survival_data):
    """Augments the shared random_survival_data fixture with a synthetic
    predicted-survival matrix on a grid bounded by the data range."""
    event_time, event_indicator, risk_score = random_survival_data
    t_max = float(np.quantile(event_time, 0.9))
    time_grid = np.linspace(0.05, t_max, 25)
    survival = np.exp(-np.exp(risk_score)[:, None] * time_grid[None, :])
    return event_time, event_indicator, survival, time_grid


def test_brier_score_matches_sksurv(random_brier_data):
    sksurv_metrics = pytest.importorskip("sksurv.metrics")
    event_time, event_indicator, survival, time_grid = random_brier_data

    structured = np.array(
        list(zip(event_indicator.astype(bool), event_time)),
        dtype=[("event", bool), ("time", float)],
    )

    ours = brier.score(event_time, event_indicator, survival, time_grid)
    _, theirs = sksurv_metrics.brier_score(structured, structured, survival, time_grid)
    np.testing.assert_allclose(ours, theirs)


def test_integrated_brier_matches_sksurv(random_brier_data):
    sksurv_metrics = pytest.importorskip("sksurv.metrics")
    event_time, event_indicator, survival, time_grid = random_brier_data

    structured = np.array(
        list(zip(event_indicator.astype(bool), event_time)),
        dtype=[("event", bool), ("time", float)],
    )

    ours = brier.integrated(event_time, event_indicator, survival, time_grid)
    theirs = sksurv_metrics.integrated_brier_score(
        structured, structured, survival, time_grid
    )
    assert ours == pytest.approx(theirs)
