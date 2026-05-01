"""Tests for CopyManager."""

import torch

from tsn_affinity.core.config import ModelConfig
from tsn_affinity.core.decision_transformer import DecisionTransformer
from tsn_affinity.sparse.converter import SparseConversionConfig
from tsn_affinity.strategies.copy_manager import CopyManager


class TestCopyManager:
    def test_create_initial_copy(self):
        model = DecisionTransformer(
            obs_shape=(4,), n_actions=2, d_model=32, n_layers=1, n_heads=2
        )
        optimizer = torch.optim.AdamW(model.parameters())
        cm = CopyManager(
            device="cpu",
            model_config=ModelConfig(obs_shape=(4,), n_actions=2),
            sparse_config=SparseConversionConfig(keep_ratio=0.5),
        )
        copy_id = cm.create_initial_copy(model, optimizer)
        assert copy_id == 0
        assert len(cm.copies) == 1

    def test_activate_copy(self):
        model = DecisionTransformer(
            obs_shape=(4,), n_actions=2, d_model=32, n_layers=1, n_heads=2
        )
        optimizer = torch.optim.AdamW(model.parameters())
        cm = CopyManager(
            device="cpu",
            model_config=ModelConfig(obs_shape=(4,), n_actions=2),
            sparse_config=SparseConversionConfig(keep_ratio=0.5),
        )
        cm.create_initial_copy(model, optimizer)
        active = cm.activate_copy(0)
        assert active is model

    def test_sync_public_state_is_noop(self):
        model = DecisionTransformer(
            obs_shape=(4,), n_actions=2, d_model=32, n_layers=1, n_heads=2
        )
        optimizer = torch.optim.AdamW(model.parameters())
        cm = CopyManager(
            device="cpu",
            model_config=ModelConfig(obs_shape=(4,), n_actions=2),
            sparse_config=SparseConversionConfig(keep_ratio=0.5),
        )
        cm.create_initial_copy(model, optimizer)
        cm.sync_public_state_to_active_copy()

    def test_get_copy_id_for_task(self):
        model = DecisionTransformer(
            obs_shape=(4,), n_actions=2, d_model=32, n_layers=1, n_heads=2
        )
        optimizer = torch.optim.AdamW(model.parameters())
        cm = CopyManager(
            device="cpu",
            model_config=ModelConfig(obs_shape=(4,), n_actions=2),
            sparse_config=SparseConversionConfig(keep_ratio=0.5),
        )
        cm.create_initial_copy(model, optimizer)
        cm.copies[0].per_task_masks[0] = {}
        assert cm.get_copy_id_for_task(0) == 0
        assert cm.get_copy_id_for_task(99) is None
