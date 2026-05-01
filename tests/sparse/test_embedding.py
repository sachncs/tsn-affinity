"""Tests for TSNSparseEmbedding."""

import torch

from tsn_affinity.sparse.embedding import TSNSparseEmbedding


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
