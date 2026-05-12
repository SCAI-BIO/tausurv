from __future__ import annotations

import numpy as np
import pytest

from tausurv import nonparametric


def test_kaplan_meier_matches_sksurv(random_survival_data):
    sksurv_nonparametric = pytest.importorskip("sksurv.nonparametric")
    event_time, event_indicator, _ = random_survival_data

    km = nonparametric.kaplan_meier(event_time, event_indicator)
    their_time, their_surv = sksurv_nonparametric.kaplan_meier_estimator(
        event_indicator.astype(bool), event_time
    )

    np.testing.assert_allclose(km.time, their_time)
    np.testing.assert_allclose(km.value, their_surv)


def test_kaplan_meier_matches_lifelines(random_survival_data):
    lifelines = pytest.importorskip("lifelines")
    event_time, event_indicator, _ = random_survival_data

    km = nonparametric.kaplan_meier(event_time, event_indicator)

    kmf = lifelines.KaplanMeierFitter().fit(event_time, event_indicator)
    their_surv = kmf.predict(km.time).to_numpy()

    np.testing.assert_allclose(km.value, their_surv)


def test_nelson_aalen_matches_sksurv(random_survival_data):
    sksurv_nonparametric = pytest.importorskip("sksurv.nonparametric")
    event_time, event_indicator, _ = random_survival_data

    na = nonparametric.nelson_aalen(event_time, event_indicator)
    their_time, their_cum_hazard = sksurv_nonparametric.nelson_aalen_estimator(
        event_indicator.astype(bool), event_time
    )

    np.testing.assert_allclose(na.time, their_time)
    np.testing.assert_allclose(na.value, their_cum_hazard)


def test_aalen_johansen_matches_lifelines():
    lifelines = pytest.importorskip("lifelines")

    rng = np.random.default_rng(0)
    n = 200
    event_time = rng.exponential(2.0, n)
    # 0 = censored, 1 or 2 = cause
    event_indicator = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])

    cif = nonparametric.aalen_johansen(event_time, event_indicator, cause=1)

    ajf = lifelines.AalenJohansenFitter(calculate_variance=False)
    ajf.fit(event_time, event_indicator, event_of_interest=1)
    their_cif = ajf.cumulative_density_.iloc[:, 0]

    # lifelines indexes by time (including time 0 sometimes); evaluate at our grid.
    their_at_ours = np.interp(cif.time, their_cif.index.values, their_cif.values)
    np.testing.assert_allclose(cif.value, their_at_ours, atol=1e-10)
