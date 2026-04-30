"""Tests for routing module."""

import torch
import pytest

from tsn_affinity.routing.affinity_metrics import (
    compute_hybrid_affinity,
    normalize_scores,
    compute_diag_stats,
)


class TestNormalizeScores:
    def test_empty_dict(self):
        result = normalize_scores({})
        assert result == {}

    def test_single_element(self):
        result = normalize_scores({0: 5.0})
        assert result[0] == 0.0

    def test_normalizes_to_range(self):
        scores = {0: 1.0, 1: 5.0, 2: 10.0}
        result = normalize_scores(scores)
        assert 0.0 <= result[0] <= 1.0
        assert 0.0 <= result[1] <= 1.0
        assert 0.0 <= result[2] <= 1.0
        assert abs(result[2] - 1.0) < 1e-6


class TestComputeHybridAffinity:
    def test_basic(self):
        score = compute_hybrid_affinity(6.0, 12.5, 12.0, 25.0, 0.7)
        assert score >= 0.0

    def test_action_only_contribution(self):
        score = compute_hybrid_affinity(12.0, 100.0, 12.0, 25.0, 0.7)
        assert score >= 0.0


class TestComputeDiagStats:
    def test_basic(self):
        z = torch.randn(10, 5, 32)
        mu, var = compute_diag_stats(z)
        assert mu.shape == (32,)
        assert var.shape == (32,)

    def test_with_mask(self):
        z = torch.randn(10, 5, 32)
        mask = torch.ones(10, 5)
        mask[:3, :] = 0
        mu, var = compute_diag_stats(z, mask)
        assert mu.shape == (32,)

    def test_empty_input(self):
        z = torch.randn(0, 32)
        mu, var = compute_diag_stats(z)
        assert mu.shape == (32,)
        assert var.shape == (32,)