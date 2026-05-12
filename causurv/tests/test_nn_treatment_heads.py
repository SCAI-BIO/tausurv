"""Tests for the treatment-specific heads module."""

from __future__ import annotations

import pytest
import torch

from causurv.nn.modules import TreatmentSpecificHeads


def test_returns_list_of_per_arm_logits():
    heads = TreatmentSpecificHeads(n_arms=2, in_features=8, out_features=10)
    phi = torch.randn(16, 8)
    out = heads(phi)
    assert isinstance(out, list) and len(out) == 2
    for arm in out:
        assert arm.shape == (16, 10)


def test_supports_k_arms():
    heads = TreatmentSpecificHeads(n_arms=4, in_features=6, out_features=5)
    phi = torch.randn(7, 6)
    out = heads(phi)
    assert len(out) == 4
    for arm in out:
        assert arm.shape == (7, 5)


def test_each_arm_has_independent_parameters():
    """The heads must not share parameters (that would be S-learner, not T)."""
    heads = TreatmentSpecificHeads(n_arms=2, in_features=4, out_features=3)
    params_arm0 = list(heads.heads[0].parameters())
    params_arm1 = list(heads.heads[1].parameters())
    for p0, p1 in zip(params_arm0, params_arm1):
        # Different objects, different values (random init).
        assert p0 is not p1


def test_rejects_too_few_arms():
    with pytest.raises(ValueError, match="n_arms must be >= 2"):
        TreatmentSpecificHeads(n_arms=1, in_features=4, out_features=3)
