"""Tests for sparse layer implementations."""

import torch
import pytest

from tsn_affinity.sparse.sparse_linear import TSNSparseLinear
from tsn_affinity.sparse.sparse_conv2d import TSNSparseConv2d
from tsn_affinity.sparse.sparse_embedding import TSNSparseEmbedding
from tsn_affinity.sparse.topk_ste import TopKMaskSTE


class TestTopKMaskSTE:
    def test_forward_selects_top_k(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = TopKMaskSTE.apply(scores, 0.4, None)
        assert mask.sum() == 2

    def test_backward_passes_gradient(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        mask = TopKMaskSTE.apply(scores, 0.4, None)
        loss = mask.sum()
        loss.backward()
        assert scores.grad is not None


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


class TestTSNSparseConv2d:
    def test_forward_uses_mask(self):
        layer = TSNSparseConv2d(3, 16, kernel_size=3, keep_ratio=0.5)
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)
        assert out.shape == (2, 16, 30, 30)

    def test_score_parameter_exists(self):
        layer = TSNSparseConv2d(3, 16, kernel_size=3, keep_ratio=0.5)
        assert layer.score is not None


class TestTSNSparseEmbedding:
    def test_forward_uses_mask(self):
        layer = TSNSparseEmbedding(100, 32, keep_ratio=0.5)
        x = torch.randint(0, 100, (10,))
        out = layer(x)
        assert out.shape == (10, 32)

    def test_score_parameter_exists(self):
        layer = TSNSparseEmbedding(100, 32, keep_ratio=0.5)
        assert layer.score is not None
        assert layer.score.shape == (100, 32)