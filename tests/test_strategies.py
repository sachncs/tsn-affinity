"""Tests for strategy implementations."""

import torch
import numpy as np
import pytest

from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.strategies.tsn_affinity import AffinityRoutingConfig
from tsn_affinity.strategies.tsn_core import TSNCoreStrategy
from tsn_affinity.strategies.tsn_affinity import TSNAffinityStrategy
from tsn_affinity.data.trajectory import Trajectory


def make_simple_trajectories(n_trajs=3, traj_len=20, obs_dim=4, n_actions=2, seed=42):
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n_trajs):
        obs = rng.normal(0.5, 0.25, (traj_len, obs_dim)).astype(np.float32)
        obs = np.clip(obs, 0, 1)
        actions = rng.integers(0, n_actions, size=traj_len).astype(np.int64)
        rewards = rng.standard_normal(traj_len).astype(np.float32) * 0.1
        timesteps = np.arange(traj_len, dtype=np.float32)
        returns_to_go = np.cumsum(rewards[::-1])[::-1].astype(np.float32)
        trajs.append(Trajectory(obs, actions, rewards, timesteps, returns_to_go))
    return trajs


class TestTSNCoreStrategy:
    def test_train_task_basic(self):
        obs_shape = (4,)
        n_actions = 2
        strategy = TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )

        trajs = make_simple_trajectories(n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0)
        result = strategy.train_task(trajs, steps=5, batch_size=2)

        assert "loss" in result
        assert "keep_ratio" in result
        assert strategy.current_task_id == 0

    def test_after_task_collects_masks(self):
        obs_shape = (4,)
        n_actions = 2
        strategy = TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )

        trajs = make_simple_trajectories(n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0)
        strategy.train_task(trajs, steps=5, batch_size=2)
        strategy.after_task(trajs)

        assert 0 in strategy.per_task_masks
        assert 0 in strategy.task_keep_ratios

    def test_multi_task_training(self):
        obs_shape = (4,)
        n_actions = 2
        strategy = TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )

        for task_id in range(3):
            trajs = make_simple_trajectories(n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=task_id * 100)
            strategy.train_task(trajs, steps=5, batch_size=2)
            strategy.after_task(trajs)

        assert strategy.current_task_id == 3
        assert len(strategy.per_task_masks) == 3

    def test_set_eval_task(self):
        obs_shape = (4,)
        n_actions = 2
        strategy = TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )

        trajs = make_simple_trajectories(n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0)
        strategy.train_task(trajs, steps=5, batch_size=2)
        strategy.after_task(trajs)

        strategy.set_eval_task(0)
        assert strategy.active_eval_task == 0

        strategy.clear_eval_task()
        assert strategy.active_eval_task is None


class TestTSNAffinityStrategy:
    def test_initialization(self):
        obs_shape = (4,)
        n_actions = 2
        affinity_config = AffinityRoutingConfig(
            mode="action",
            action_threshold=12.0,
        )
        strategy = TSNAffinityStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
            affinity_config=affinity_config,
        )

        assert len(strategy.copy_states) == 1
        assert strategy.current_copy_id == 0

    def test_task_similarity_tracking(self):
        obs_shape = (4,)
        n_actions = 2
        affinity_config = AffinityRoutingConfig(mode="action")
        strategy = TSNAffinityStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
            affinity_config=affinity_config,
        )

        trajs = make_simple_trajectories(n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0)
        strategy.train_task(trajs, steps=5, batch_size=2)
        strategy.after_task(trajs)

        assert 0 in strategy.task_similarity
        assert "copy_id" in strategy.task_similarity[0]

    def test_routing_to_existing_copy(self):
        obs_shape = (4,)
        n_actions = 2
        # Use high threshold and disable relative threshold to force routing to existing copy
        affinity_config = AffinityRoutingConfig(
            mode="action",
            action_threshold=50.0,
            relative_threshold=False,  # Disable relative threshold
        )
        strategy = TSNAffinityStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
            affinity_config=affinity_config,
        )

        for task_id in range(3):
            trajs = make_simple_trajectories(n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=task_id * 100)
            strategy.train_task(trajs, steps=5, batch_size=2)
            strategy.after_task(trajs)

        # With relative_threshold=False and high action_threshold, all tasks should use copy 0
        for task_id in range(3):
            assert strategy.task_to_copy[task_id] == 0

    def test_routing_creates_new_copy_when_dissimilar(self):
        """Test that routing creates new copy when tasks are dissimilar (low relative margin)."""
        obs_shape = (4,)
        n_actions = 2
        affinity_config = AffinityRoutingConfig(
            mode="action",
            action_threshold=0.0,
            relative_threshold=True,
            copy_creation_margin=2.5,  # Create new copy when margin is large
        )
        strategy = TSNAffinityStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
            affinity_config=affinity_config,
        )

        for task_id in range(3):
            trajs = make_simple_trajectories(n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=task_id * 100)
            strategy.train_task(trajs, steps=5, batch_size=2)
            strategy.after_task(trajs)

        # Each task should get its own copy since they're dissimilar
        for task_id in range(3):
            assert strategy.task_to_copy[task_id] == task_id


class TestDecisionTransformerAct:
    """Tests for Decision Transformer act() method - online inference."""

    def test_act_produces_valid_action(self):
        """act() should produce an action index within valid range."""
        obs_shape = (4,)
        n_actions = 2
        strategy = TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )

        # Initialize history
        strategy.model.reset_history()

        # Create a valid observation
        obs = torch.randn(4)

        # act() should return an integer action
        action = strategy.model.act(obs, returns_to_go=0.0, deterministic=True)
        assert isinstance(action, int)
        assert 0 <= action < n_actions

    def test_act_handles_empty_history(self):
        """act() should work when history is empty (first step)."""
        obs_shape = (4,)
        n_actions = 2
        strategy = TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )

        # Start with empty history
        strategy.model.reset_history()

        obs = torch.randn(4)
        action = strategy.model.act(obs, returns_to_go=0.0, deterministic=True)

        assert isinstance(action, int)
        assert 0 <= action < n_actions

    def test_act_accumulates_history(self):
        """act() should accumulate history across calls."""
        obs_shape = (4,)
        n_actions = 2
        strategy = TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )

        strategy.model.reset_history()

        obs = torch.randn(4)

        # First call
        action1 = strategy.model.act(obs, returns_to_go=0.0, deterministic=True)
        assert len(strategy.model._obs_history) == 1
        assert len(strategy.model._act_history) == 1

        # Second call
        action2 = strategy.model.act(obs, returns_to_go=0.0, deterministic=True)
        assert len(strategy.model._obs_history) == 2
        assert len(strategy.model._act_history) == 2

    def test_act_with_different_returns_to_go(self):
        """act() should accept different returns_to_go values."""
        obs_shape = (4,)
        n_actions = 2
        strategy = TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )

        strategy.model.reset_history()
        obs = torch.randn(4)

        # Different returns_to_go values should not raise
        action1 = strategy.model.act(obs, returns_to_go=0.0, deterministic=True)
        action2 = strategy.model.act(obs, returns_to_go=100.0, deterministic=True)
        action3 = strategy.model.act(obs, returns_to_go=-50.0, deterministic=True)

        assert isinstance(action1, int)
        assert isinstance(action2, int)
        assert isinstance(action3, int)

    def test_reset_history_clears_state(self):
        """reset_history() should clear all history."""
        obs_shape = (4,)
        n_actions = 2
        strategy = TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )

        strategy.model.reset_history()
        obs = torch.randn(4)

        # Add some history
        strategy.model.act(obs, returns_to_go=0.0, deterministic=True)
        strategy.model.act(obs, returns_to_go=0.0, deterministic=True)

        assert len(strategy.model._obs_history) == 2
        assert len(strategy.model._act_history) == 2

        # Reset
        strategy.model.reset_history()

        assert len(strategy.model._obs_history) == 0
        assert len(strategy.model._act_history) == 0
        assert len(strategy.model._rtg_history) == 0
        assert len(strategy.model._ts_history) == 0