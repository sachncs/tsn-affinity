"""Tests for TSNSparseLinear."""

import torch

from tsn_affinity.sparse.linear import TSNSparseLinear


class TestTSNSparseLinear:
    def test_forward_uses_mask(self):
        layer = TSNSparseLinear(10, 5, keep_ratio=0.5, allow_weight_reuse=False)
        x = torch.randn(2, 10)
        out = layer(x)
        assert out.shape == (2, 5)

    def test_score_parameter_exists(self):
        layer = TSNSparseLinear(10, 5, keep_ratio=0.5)
        assert layer.score is not None
        assert layer.score.shape == (5, 10)

    def test_reset_scores(self):
        layer = TSNSparseLinear(10, 5, keep_ratio=0.5)
        old_score = layer.score.clone()
        layer.reset_scores()
        assert not torch.equal(layer.score, old_score)

    def test_active_mask_override(self):
        layer = TSNSparseLinear(10, 5, keep_ratio=0.5)
        override_mask = torch.ones(5, 10, dtype=torch.float32)
        layer.active_weight_mask = override_mask
        out = layer(x=torch.randn(2, 10))
        assert out.shape == (2, 5)

    def test_free_mask_respected(self):
        layer = TSNSparseLinear(10, 5, keep_ratio=0.5, allow_weight_reuse=False)
        free_mask = torch.ones(5, 10)
        layer.occupied_weight_mask = free_mask
        mask = layer.get_free_weight_mask()
        assert mask is not None
        assert mask.shape == (5, 10)
        assert (mask.bool() == (~free_mask.bool())).all()
