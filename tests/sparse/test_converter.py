"""Tests for sparse module converter."""

import torch

from tsn_affinity.core.decision_transformer import DecisionTransformer
from tsn_affinity.sparse.converter import (
    SparseConversionConfig,
    convert_to_sparse,
    iter_sparse_modules,
    rebuild_optimizer,
)


class TestIterSparseModules:
    def test_finds_sparse_modules(self):
        model = DecisionTransformer(
            obs_shape=(4,), n_actions=2, d_model=32, n_layers=1, n_heads=2
        )
        convert_to_sparse(model, SparseConversionConfig(keep_ratio=0.5))
        modules = list(iter_sparse_modules(model))
        assert len(modules) > 0

    def test_empty_for_standard_model(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 5))
        modules = list(iter_sparse_modules(model))
        assert len(modules) == 0


class TestRebuildOptimizer:
    def test_rebuilds_adamw(self):
        model = DecisionTransformer(
            obs_shape=(4,), n_actions=2, d_model=32, n_layers=1, n_heads=2
        )
        old_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        new_opt = rebuild_optimizer(old_opt, model.parameters())
        assert new_opt is not None
