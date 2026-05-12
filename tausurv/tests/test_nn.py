from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tausurv import simulations
from tausurv.nn import (
    MLP,
    CoxPHLoss,
    DeepHit,
    DeepHitLoss,
    DeepHitRankingLoss,
    DeepSurv,
    LogisticHazard,
    LogisticHazardLoss,
    PMFLoss,
    functional,
)


def test_mlp_forward_shape():
    model = MLP(in_features=5, out_features=1, hidden_dim=16, n_blocks=2)
    x = torch.randn(8, 5)
    y = model(x)
    assert y.shape == (8, 1)


def test_mlp_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation"):
        MLP(in_features=5, out_features=1, activation="swish")


def test_mlp_rejects_unknown_norm():
    with pytest.raises(ValueError, match="norm"):
        MLP(in_features=5, out_features=1, norm="rms")


def test_deepsurv_forward_shape():
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2)
    x = torch.randn(8, 5)
    y = model(x)
    assert y.shape == (8,)


def test_cox_nll_hand_computed():
    # Y=[3, 1, 2], delta=[1, 1, 0], log_risks=[0.5, 1.0, 0.3].
    # Sort desc by Y: [3, 2, 1] -> orig indices [0, 2, 1].
    # log_risks_sorted = [0.5, 0.3, 1.0]; delta_sorted = [1, 0, 1].
    # logcumsumexp = [0.5, log(e^.5 + e^.3), log(e^.5 + e^.3 + e^1.0)].
    log_risk = torch.tensor([0.5, 1.0, 0.3])
    event_time = torch.tensor([3.0, 1.0, 2.0])
    event_indicator = torch.tensor([1, 1, 0], dtype=torch.float32)

    expected_lcse_last = math.log(math.exp(0.5) + math.exp(0.3) + math.exp(1.0))
    expected_sum = -((0.5 - 0.5) + (1.0 - expected_lcse_last))
    expected_mean = expected_sum / 2

    out_mean = functional.cox_nll(
        log_risk, event_time, event_indicator, reduction="mean"
    )
    out_sum = functional.cox_nll(log_risk, event_time, event_indicator, reduction="sum")
    assert torch.allclose(out_mean, torch.tensor(expected_mean), atol=1e-6)
    assert torch.allclose(out_sum, torch.tensor(expected_sum), atol=1e-6)


def test_cox_nll_perfect_ranking_lower_than_random():
    rng = np.random.default_rng(0)
    n = 100
    Y = rng.uniform(0, 10, n).astype(np.float32)
    delta = rng.choice([0, 1], n).astype(np.float32)
    perfect = torch.tensor(-Y)  # higher risk -> shorter time
    random = torch.tensor(rng.normal(size=n).astype(np.float32))

    nll_perfect = functional.cox_nll(perfect, torch.tensor(Y), torch.tensor(delta))
    nll_random = functional.cox_nll(random, torch.tensor(Y), torch.tensor(delta))
    assert nll_perfect < nll_random


def test_deephit_forward_shape_single_event():
    # Default n_causes=1 -> output is (n, 1, n_bins) — always 3D.
    model = DeepHit(in_features=5, n_bins=10, hidden_dim=16)
    x = torch.randn(8, 5)
    logits = model(x)
    assert logits.shape == (8, 1, 10)


def test_deephit_forward_shape_competing_risks():
    model = DeepHit(in_features=5, n_bins=10, n_causes=3, hidden_dim=16)
    x = torch.randn(8, 5)
    logits = model(x)
    assert logits.shape == (8, 3, 10)


def test_deephit_rejects_invalid_n_causes():
    with pytest.raises(ValueError, match="n_causes"):
        DeepHit(in_features=5, n_bins=10, n_causes=0)


def test_pmf_nll_competing_risks_hand_computed():
    # 1 subject, 2 causes, 3 bins. Subject is event of cause 2, bin 0.
    # All logits zero -> uniform PMF over 2*3=6 cells, each = 1/6.
    # log(1/6) for the (cause=2, bin=0) cell.
    logits = torch.zeros(1, 2, 3)
    event_time = torch.tensor([0.5])
    event_indicator = torch.tensor([2])
    time_bins = torch.tensor([1.0, 2.0, 3.0])
    loss = functional.pmf_nll(logits, event_time, event_indicator, time_bins)
    assert torch.allclose(loss, torch.tensor(math.log(6.0)), atol=1e-6)


def test_pmf_nll_perfect_prediction_is_low():
    # Event at bin 0 (Y=0.5 falls in (0, 1]), confident at bin 0.
    logits = torch.tensor([[10.0, -10.0, -10.0]])
    event_time = torch.tensor([0.5])
    event_indicator = torch.tensor([1.0])
    time_bins = torch.tensor([1.0, 2.0, 3.0])
    loss = functional.pmf_nll(logits, event_time, event_indicator, time_bins)
    assert loss < 0.01


def test_pmf_nll_wrong_prediction_is_high():
    # Event at bin 0 but model puts all mass in bin 2.
    logits = torch.tensor([[-10.0, -10.0, 10.0]])
    event_time = torch.tensor([0.5])
    event_indicator = torch.tensor([1.0])
    time_bins = torch.tensor([1.0, 2.0, 3.0])
    loss = functional.pmf_nll(logits, event_time, event_indicator, time_bins)
    assert loss > 5.0


def test_pmf_nll_censored_uses_survival_past_bin():
    # Subject censored in bin 1; loss = -log(1 - cdf at bin 1) = -log(P(T > bin 1)).
    # With uniform logits, pmf = [1/3, 1/3, 1/3], cdf = [1/3, 2/3, 1].
    # 1 - cdf[1] = 1/3. -log(1/3) ~ 1.0986.
    logits = torch.tensor([[0.0, 0.0, 0.0]])
    event_time = torch.tensor([1.5])  # bin 1
    event_indicator = torch.tensor([0.0])
    time_bins = torch.tensor([1.0, 2.0, 3.0])
    loss = functional.pmf_nll(logits, event_time, event_indicator, time_bins)
    assert torch.allclose(loss, torch.tensor(math.log(3.0)), atol=1e-6)


def test_deephit_ranking_hand_computed():
    # Two events; pair (0, 1) is comparable. Confident predictions:
    # case 0 has all mass in bin 0; case 1 has all mass in bin 2.
    # F_0(T_0=bin 0) -> 1.0; F_1(T_0=bin 0) -> 0.0; diff -> 1.0.
    # With sigma=1, penalty = exp(-1).
    logits = torch.tensor([[10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]])
    event_time = torch.tensor([0.5, 2.5])
    event_indicator = torch.tensor([1.0, 1.0])
    time_bins = torch.tensor([1.0, 2.0, 3.0])
    out = functional.deephit_ranking(
        logits, event_time, event_indicator, time_bins, sigma=1.0
    )
    assert torch.allclose(out, torch.tensor(math.exp(-1.0)), atol=1e-3)


def test_deephit_ranking_zero_when_no_comparable_pairs():
    # Both subjects censored -> no case in any pair -> zero loss.
    logits = torch.randn(3, 5)
    event_time = torch.tensor([1.0, 2.0, 3.0])
    event_indicator = torch.tensor([0.0, 0.0, 0.0])
    time_bins = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    out = functional.deephit_ranking(logits, event_time, event_indicator, time_bins)
    assert out.item() == 0.0


def test_deephit_loss_recovers_nll_at_alpha_one():
    rng = np.random.default_rng(0)
    logits = torch.tensor(rng.normal(size=(20, 5)).astype(np.float32))
    event_time = torch.tensor(rng.uniform(0, 5, 20).astype(np.float32))
    event_indicator = torch.tensor(rng.choice([0, 1], 20).astype(np.float32))
    time_bins = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    combined = functional.deephit_loss(
        logits, event_time, event_indicator, time_bins, alpha=1.0
    )
    nll_only = functional.pmf_nll(logits, event_time, event_indicator, time_bins)
    assert torch.allclose(combined, nll_only)


def test_logistic_hazard_forward_shape():
    model = LogisticHazard(in_features=5, n_bins=10, hidden_dim=16)
    x = torch.randn(8, 5)
    logits = model(x)
    assert logits.shape == (8, 10)


def test_logistic_hazard_nll_perfect_prediction_is_low():
    # Event at bin 0; logits[0] very high -> h_0 ~ 1; log h_0 ~ 0.
    logits = torch.tensor([[10.0, -10.0, -10.0]])
    event_time = torch.tensor([0.5])
    event_indicator = torch.tensor([1.0])
    time_bins = torch.tensor([1.0, 2.0, 3.0])
    loss = functional.logistic_hazard_nll(
        logits, event_time, event_indicator, time_bins
    )
    assert loss < 0.01


def test_logistic_hazard_nll_censored_credits_survival():
    # Censored at bin 2 (Y=2.5); contribution = log(1 - h_0) + log(1 - h_1) + log(1 - h_2).
    # With logits = [-10, -10, -10] -> h ~ 0 everywhere -> log(1 - h) ~ 0.
    logits = torch.tensor([[-10.0, -10.0, -10.0]])
    event_time = torch.tensor([2.5])
    event_indicator = torch.tensor([0.0])
    time_bins = torch.tensor([1.0, 2.0, 3.0])
    loss = functional.logistic_hazard_nll(
        logits, event_time, event_indicator, time_bins
    )
    assert loss < 0.01


def test_loss_classes_match_their_functional_form():
    rng = np.random.default_rng(0)
    n, K = 20, 5
    logits = torch.tensor(rng.normal(size=(n, K)).astype(np.float32))
    log_risk = torch.tensor(rng.normal(size=n).astype(np.float32))
    event_time = torch.tensor(rng.uniform(0, 5, n).astype(np.float32))
    event_indicator = torch.tensor(rng.choice([0, 1], n).astype(np.float32))
    time_bins = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    assert torch.allclose(
        CoxPHLoss()(log_risk, event_time, event_indicator),
        functional.cox_nll(log_risk, event_time, event_indicator),
    )
    assert torch.allclose(
        PMFLoss()(logits, event_time, event_indicator, time_bins),
        functional.pmf_nll(logits, event_time, event_indicator, time_bins),
    )
    assert torch.allclose(
        DeepHitRankingLoss(sigma=0.2)(logits, event_time, event_indicator, time_bins),
        functional.deephit_ranking(
            logits, event_time, event_indicator, time_bins, sigma=0.2
        ),
    )
    assert torch.allclose(
        DeepHitLoss(alpha=0.3, sigma=0.2)(
            logits, event_time, event_indicator, time_bins
        ),
        functional.deephit_loss(
            logits, event_time, event_indicator, time_bins, alpha=0.3, sigma=0.2
        ),
    )
    assert torch.allclose(
        LogisticHazardLoss()(logits, event_time, event_indicator, time_bins),
        functional.logistic_hazard_nll(logits, event_time, event_indicator, time_bins),
    )


