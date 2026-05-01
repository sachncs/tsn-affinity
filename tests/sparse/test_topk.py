"""Tests for TopKMaskSTE."""

import pytest
import torch

from tsn_affinity.sparse.topk import TopKMaskSTE


class TestTopKMaskSTE:
    @pytest.mark.parametrize("keep_ratio", [0.1, 0.4, 0.5, 0.9])
    def test_forward_selects_top_k(self, keep_ratio):
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = TopKMaskSTE.apply(scores, keep_ratio, None)
        expected_k = max(1, int(keep_ratio * len(scores)))
        assert mask.sum() == expected_k

    def test_backward_passes_gradient(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        mask = TopKMaskSTE.apply(scores, 0.4, None)
        loss = mask.sum()
        loss.backward()
        assert scores.grad is not None
