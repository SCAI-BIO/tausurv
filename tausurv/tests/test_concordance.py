from __future__ import annotations

import numpy as np
import pytest

from tausurv.metrics import concordance
from tausurv.step import StepFunction


def test_harrell_perfect_concordance():
    event_time = np.array([1.0, 2.0, 3.0, 4.0])
    event_indicator = np.array([1, 1, 1, 1])
    risk_score = np.array([4.0, 3.0, 2.0, 1.0])
    assert concordance.harrell(event_time, event_indicator, risk_score) == 1.0


def test_harrell_perfect_anticoncordance():
    event_time = np.array([1.0, 2.0, 3.0, 4.0])
    event_indicator = np.array([1, 1, 1, 1])
    risk_score = np.array([1.0, 2.0, 3.0, 4.0])
    assert concordance.harrell(event_time, event_indicator, risk_score) == 0.0


def test_harrell_all_tied_risk_is_half():
    event_time = np.array([1.0, 2.0, 3.0])
    event_indicator = np.array([1, 1, 1])
    risk_score = np.array([1.0, 1.0, 1.0])
    assert concordance.harrell(event_time, event_indicator, risk_score) == 0.5


def test_harrell_censored_partner_counted():
    # Censored cases are valid as the "later" partner but never as the index.
    event_time = np.array([1.0, 2.0, 3.0])
    event_indicator = np.array([1, 0, 1])
    risk_score = np.array([3.0, 1.0, 2.0])
    # comparable pairs: (0, 1), (0, 2); both concordant.
    assert concordance.harrell(event_time, event_indicator, risk_score) == 1.0


def test_harrell_no_comparable_pairs_raises():
    event_time = np.array([1.0, 2.0, 3.0])
    event_indicator = np.array([0, 0, 0])
    risk_score = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="no comparable pairs"):
        concordance.harrell(event_time, event_indicator, risk_score)


def test_harrell_hand_computed():
    # 5 comparable pairs (0,1), (0,2), (0,3), (1,2), (1,3).
    # (0,*): r_0=0.9 beats r_1=0.1, r_2=0.5, r_3=0.7 -> 3 concordant.
    # (1,*): r_1=0.1 below r_2=0.5, r_3=0.7         -> 2 discordant.
    # Tied: 0. C = 3 / 5 = 0.6.
    event_time = np.array([1.0, 2.0, 3.0, 4.0])
    event_indicator = np.array([1, 1, 0, 1])
    risk_score = np.array([0.9, 0.1, 0.5, 0.7])
    assert concordance.harrell(
        event_time, event_indicator, risk_score
    ) == pytest.approx(0.6)


def test_uno_reduces_to_harrell_when_no_censoring():
    # With delta = 1 everywhere, G(t) = 1 and IPCW weights are all 1, so Uno
    # collapses to Harrell on the same data.
    event_time = np.array([1.0, 2.0, 3.0, 4.0])
    event_indicator = np.array([1, 1, 0, 1])
    risk_score = np.array([0.9, 0.1, 0.5, 0.7])
    # Take only the events for a clean no-censoring comparison.
    mask = event_indicator == 1
    uno_c = concordance.uno(
        event_time[mask], event_indicator[mask], risk_score[mask], tau=10.0
    )
    harrell_c = concordance.harrell(
        event_time[mask], event_indicator[mask], risk_score[mask]
    )
    assert uno_c == pytest.approx(harrell_c)


def test_uno_tau_truncates_pairs():
    # Only Y_i < tau can be indexed; Y_0 = 1 is the only such index.
    # Pairs (0, 1) and (0, 2) are both concordant. C = 1.
    event_time = np.array([1.0, 2.0, 3.0])
    event_indicator = np.array([1, 1, 1])
    risk_score = np.array([3.0, 2.0, 1.0])
    assert concordance.uno(event_time, event_indicator, risk_score, tau=2.0) == 1.0


def test_uno_custom_censoring_survival_overrides_default():
    # Constant Ĝ(t) = 0.5 makes every weight 1/0.5^2 = 4, scaling num and den
    # equally — so the ratio matches the no-censoring case.
    event_time = np.array([1.0, 2.0, 3.0, 4.0])
    event_indicator = np.array([1, 1, 1, 1])
    risk_score = np.array([4.0, 3.0, 2.0, 1.0])
    custom_G = StepFunction(time=np.array([0.5]), value=np.array([0.5]))
    assert (
        concordance.uno(
            event_time,
            event_indicator,
            risk_score,
            tau=10.0,
            censoring_survival=custom_G,
        )
        == 1.0
    )


def test_uno_no_comparable_pairs_raises():
    event_time = np.array([1.0, 2.0, 3.0])
    event_indicator = np.array([0, 0, 0])
    risk_score = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="no comparable pairs"):
        concordance.uno(event_time, event_indicator, risk_score, tau=10.0)


def test_antolini_perfect_concordance():
    # At each case's event time, the case has the lowest predicted survival.
    event_time = np.array([1.0, 2.0, 3.0])
    event_indicator = np.array([1, 1, 1])
    time_grid = np.array([1.0, 2.0, 3.0])
    survival = np.array(
        [
            [0.1, 0.05, 0.01],
            [0.5, 0.30, 0.10],
            [0.8, 0.70, 0.60],
        ]
    )
    assert concordance.antolini(event_time, event_indicator, survival, time_grid) == 1.0


def test_antolini_perfect_anticoncordance():
    event_time = np.array([1.0, 2.0, 3.0])
    event_indicator = np.array([1, 1, 1])
    time_grid = np.array([1.0, 2.0, 3.0])
    survival = np.array(
        [
            [0.8, 0.70, 0.60],
            [0.5, 0.30, 0.10],
            [0.1, 0.05, 0.01],
        ]
    )
    assert concordance.antolini(event_time, event_indicator, survival, time_grid) == 0.0


def test_antolini_all_tied_predictions_is_half():
    # Every subject shares the same survival curve — every comparison is tied.
    event_time = np.array([1.0, 2.0, 3.0])
    event_indicator = np.array([1, 1, 1])
    time_grid = np.array([1.0, 2.0, 3.0])
    survival = np.tile(np.array([0.5, 0.3, 0.1]), (3, 1))
    assert concordance.antolini(event_time, event_indicator, survival, time_grid) == 0.5


def test_antolini_reduces_to_harrell_for_monotone_survival():
    # If S(t | x_i) = exp(-r_i * t), then S(t | x_i) < S(t | x_j) iff r_i > r_j
    # at any t > 0, so Antolini and Harrell agree on the same risk ordering.
    rng = np.random.default_rng(0)
    n = 30
    risk_score = rng.uniform(0.1, 2.0, size=n)
    event_time = rng.uniform(0.1, 5.0, size=n)
    event_indicator = rng.integers(0, 2, size=n)
    time_grid = np.linspace(0.01, 5.0, 40)
    survival = np.exp(-risk_score[:, None] * time_grid[None, :])

    a = concordance.antolini(event_time, event_indicator, survival, time_grid)
    h = concordance.harrell(event_time, event_indicator, risk_score)
    assert a == pytest.approx(h)


def test_antolini_no_comparable_pairs_raises():
    event_time = np.array([1.0, 2.0])
    event_indicator = np.array([0, 0])
    time_grid = np.array([1.0, 2.0])
    survival = np.array([[0.5, 0.2], [0.7, 0.4]])
    with pytest.raises(ValueError, match="no comparable pairs"):
        concordance.antolini(event_time, event_indicator, survival, time_grid)


def test_blanche_perfect_concordance_no_censoring():
    # Cases (Y<=tau, event): patients 0, 1. Controls (Y>tau): patients 2, 3.
    # Risks rank cases above controls -> every pair concordant.
    event_time = np.array([1.0, 2.0, 3.0, 4.0])
    event_indicator = np.array([1, 1, 1, 1])
    risk_score = np.array([4.0, 3.0, 2.0, 1.0])
    assert concordance.blanche(event_time, event_indicator, risk_score, tau=2.5) == 1.0


def test_blanche_perfect_anticoncordance():
    event_time = np.array([1.0, 2.0, 3.0, 4.0])
    event_indicator = np.array([1, 1, 1, 1])
    risk_score = np.array([1.0, 2.0, 3.0, 4.0])
    assert concordance.blanche(event_time, event_indicator, risk_score, tau=2.5) == 0.0


def test_blanche_all_tied_is_half():
    event_time = np.array([1.0, 2.0, 3.0, 4.0])
    event_indicator = np.array([1, 1, 1, 1])
    risk_score = np.array([1.0, 1.0, 1.0, 1.0])
    assert concordance.blanche(event_time, event_indicator, risk_score, tau=2.5) == 0.5


def test_blanche_ipcw_weighting_hand_computed():
    # Y       = [1, 2, 3, 4, 5]
    # delta   = [1, 0, 1, 1, 0]
    # r       = [3, 4, 1, 5, 2]
    # tau     = 3.5
    # Cases (Y<=tau, event): i=0 (Y=1, r=3),   i=2 (Y=3, r=1).
    # Controls (Y>tau):      j=3 (Y=4, r=5),   j=4 (Y=5, r=2).
    # KM on (Y, 1-delta) = [0,1,0,0,1] yields G:
    #   time=[1,2,3,4,5], value=[1, 0.75, 0.75, 0.75, 0].
    # G(Y_i^-): G(1^-)=1, G(3^-)=0.75. w_case = [1, 4/3].
    # Pair scores:
    #   (r=3 vs r=5): 0    (r=3 vs r=2): 1
    #   (r=1 vs r=5): 0    (r=1 vs r=2): 0
    # numerator   = 1*1 + 1*0 + (4/3)*0 + (4/3)*0 = 1
    # denominator = (1 + 4/3) * 2 controls = 14/3
    # C = 3/14.
    event_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event_indicator = np.array([1, 0, 1, 1, 0])
    risk_score = np.array([3.0, 4.0, 1.0, 5.0, 2.0])
    assert concordance.blanche(
        event_time, event_indicator, risk_score, tau=3.5
    ) == pytest.approx(3 / 14)


def test_blanche_custom_censoring_survival_overrides_default():
    # Constant Ĝ(t) = 0.5 scales every case weight equally; ratio unchanged.
    event_time = np.array([1.0, 2.0, 3.0, 4.0])
    event_indicator = np.array([1, 1, 1, 1])
    risk_score = np.array([4.0, 3.0, 2.0, 1.0])
    custom_G = StepFunction(time=np.array([0.5]), value=np.array([0.5]))
    assert (
        concordance.blanche(
            event_time,
            event_indicator,
            risk_score,
            tau=2.5,
            censoring_survival=custom_G,
        )
        == 1.0
    )


def test_blanche_no_cases_raises():
    event_time = np.array([1.0, 2.0, 3.0])
    event_indicator = np.array([0, 0, 0])
    risk_score = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="case"):
        concordance.blanche(event_time, event_indicator, risk_score, tau=2.0)


def test_blanche_no_controls_raises():
    event_time = np.array([1.0, 2.0, 3.0])
    event_indicator = np.array([1, 1, 1])
    risk_score = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="control"):
        concordance.blanche(event_time, event_indicator, risk_score, tau=10.0)
