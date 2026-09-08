from __future__ import annotations

import numpy as np
import pytest

from tausurv.metrics import concordance


def test_harrell_matches_sksurv(random_survival_data):
    sksurv_metrics = pytest.importorskip("sksurv.metrics")
    event_time, event_indicator, risk_score = random_survival_data

    ours = concordance.harrell(event_time, event_indicator, risk_score)
    theirs, *_ = sksurv_metrics.concordance_index_censored(
        event_indicator.astype(bool), event_time, risk_score
    )
    assert ours == pytest.approx(theirs)


def test_harrell_matches_lifelines(random_survival_data):
    lifelines_utils = pytest.importorskip("lifelines.utils")
    event_time, event_indicator, risk_score = random_survival_data

    ours = concordance.harrell(event_time, event_indicator, risk_score)
    # lifelines expects a survival-like score (higher = longer survival).
    theirs = lifelines_utils.concordance_index(event_time, -risk_score, event_indicator)
    assert ours == pytest.approx(theirs)


def test_uno_matches_sksurv(random_survival_data):
    sksurv_metrics = pytest.importorskip("sksurv.metrics")
    event_time, event_indicator, risk_score = random_survival_data

    tau = float(np.quantile(event_time, 0.75))

    structured = np.array(
        list(zip(event_indicator.astype(bool), event_time)),
        dtype=[("event", bool), ("time", float)],
    )

    ours = concordance.uno(event_time, event_indicator, risk_score, tau=tau)
    theirs, *_ = sksurv_metrics.concordance_index_ipcw(
        structured, structured, risk_score, tau=tau
    )
    assert ours == pytest.approx(theirs)


def test_antolini_matches_pycox(random_survival_data):
    pycox_eval = pytest.importorskip("pycox.evaluation")
    pd = pytest.importorskip("pandas")
    event_time, event_indicator, risk_score = random_survival_data

    t_max = float(np.quantile(event_time, 0.9))
    time_grid = np.linspace(0.05, t_max, 25)
    survival = np.exp(-np.exp(risk_score)[:, None] * time_grid[None, :])

    ours = concordance.antolini(event_time, event_indicator, survival, time_grid)

    # pycox expects a (n_times, n_subjects) DataFrame indexed by time.
    surv_df = pd.DataFrame(survival.T, index=time_grid)
    ev = pycox_eval.EvalSurv(surv_df, event_time, event_indicator, censor_surv="km")
    theirs = ev.concordance_td("antolini")
    assert ours == pytest.approx(theirs)


@pytest.fixture
def tied_survival_data(random_survival_data):
    # Rounding produces tied times, both event-event and event-censored.
    event_time, event_indicator, risk_score = random_survival_data
    return np.round(event_time, 1), event_indicator, risk_score


def test_harrell_matches_references_with_tied_times(tied_survival_data):
    sksurv_metrics = pytest.importorskip("sksurv.metrics")
    lifelines_utils = pytest.importorskip("lifelines.utils")
    event_time, event_indicator, risk_score = tied_survival_data
    assert len(np.unique(event_time)) < len(event_time)

    ours = concordance.harrell(event_time, event_indicator, risk_score)
    sksurv_c, *_ = sksurv_metrics.concordance_index_censored(
        event_indicator.astype(bool), event_time, risk_score
    )
    lifelines_c = lifelines_utils.concordance_index(
        event_time, -risk_score, event_indicator
    )
    assert ours == pytest.approx(sksurv_c)
    assert ours == pytest.approx(lifelines_c)


def test_uno_matches_sksurv_with_tied_times(tied_survival_data):
    sksurv_metrics = pytest.importorskip("sksurv.metrics")
    event_time, event_indicator, risk_score = tied_survival_data
    tau = float(np.quantile(event_time, 0.75))
    structured = np.array(
        list(zip(event_indicator.astype(bool), event_time, strict=True)),
        dtype=[("event", bool), ("time", float)],
    )

    ours = concordance.uno(event_time, event_indicator, risk_score, tau=tau)
    theirs, *_ = sksurv_metrics.concordance_index_ipcw(
        structured, structured, risk_score, tau=tau
    )
    assert ours == pytest.approx(theirs)


def test_antolini_matches_pycox_with_tied_times(tied_survival_data):
    pycox_eval = pytest.importorskip("pycox.evaluation")
    pd = pytest.importorskip("pandas")
    event_time, event_indicator, risk_score = tied_survival_data
    time_grid = np.linspace(0.05, float(np.quantile(event_time, 0.9)), 25)
    survival = np.exp(-np.exp(risk_score)[:, None] * time_grid[None, :])

    ours = concordance.antolini(event_time, event_indicator, survival, time_grid)
    surv_df = pd.DataFrame(survival.T, index=time_grid)
    ev = pycox_eval.EvalSurv(surv_df, event_time, event_indicator, censor_surv="km")
    assert ours == pytest.approx(ev.concordance_td("antolini"))
