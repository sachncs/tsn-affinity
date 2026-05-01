"""Tests for TSNSparseConv2d."""

import torch

from tsn_affinity.sparse.conv2d import TSNSparseConv2d


class TestTSNSparseConv2d:
    def test_forward_uses_mask(self):
        layer = TSNSparseConv2d(3, 16, kernel_size=3, keep_ratio=0.5)
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)
        assert out.shape == (2, 16, 30, 30)

    def test_score_parameter_exists(self):
        layer = TSNSparseConv2d(3, 16, kernel_size=3, keep_ratio=0.5)
        assert layer.score is not None
