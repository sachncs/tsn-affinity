"""Tests for TSNAffinityStrategy."""

from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.strategies.tsn_affinity import (
    AffinityRoutingConfig,
    TSNAffinityStrategy,
)


class TestTSNAffinityStrategy:
    def test_initialization(self):
        affinity_config = AffinityRoutingConfig(
            mode="action",
            action_threshold=12.0,
        )
        strategy = TSNAffinityStrategy(
            obs_shape=(4,),
            n_actions=2,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=(4,), n_actions=2),
            sparse_config=SparseConfig(keep_ratio=0.5),
            affinity_config=affinity_config,
        )

        assert len(strategy.copy_states) == 1
        assert strategy.current_copy_id == 0

    def test_task_similarity_tracking(self, simple_trajectories):
        affinity_config = AffinityRoutingConfig(mode="action")
        strategy = TSNAffinityStrategy(
            obs_shape=(4,),
            n_actions=2,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=(4,), n_actions=2),
            sparse_config=SparseConfig(keep_ratio=0.5),
            affinity_config=affinity_config,
        )

        trajs = simple_trajectories(
            n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0
        )
        strategy.train_task(trajs, steps=5, batch_size=2)
        strategy.after_task(trajs)

        assert 0 in strategy.task_similarity
        assert "copy_id" in strategy.task_similarity[0]

    def test_routing_to_existing_copy(self, simple_trajectories):
        affinity_config = AffinityRoutingConfig(
            mode="action",
            action_threshold=50.0,
            relative_threshold=False,
        )
        strategy = TSNAffinityStrategy(
            obs_shape=(4,),
            n_actions=2,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=(4,), n_actions=2),
            sparse_config=SparseConfig(keep_ratio=0.5),
            affinity_config=affinity_config,
        )

        for task_id in range(3):
            trajs = simple_trajectories(
                n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=task_id * 100
            )
            strategy.train_task(trajs, steps=5, batch_size=2)
            strategy.after_task(trajs)

        for task_id in range(3):
            assert strategy.task_to_copy[task_id] == 0

    def test_routing_creates_new_copy_when_dissimilar(self, simple_trajectories):
        affinity_config = AffinityRoutingConfig(
            mode="action",
            action_threshold=0.0,
            relative_threshold=True,
            copy_creation_margin=2.5,
        )
        strategy = TSNAffinityStrategy(
            obs_shape=(4,),
            n_actions=2,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=(4,), n_actions=2),
            sparse_config=SparseConfig(keep_ratio=0.5),
            affinity_config=affinity_config,
        )

        for task_id in range(3):
            trajs = simple_trajectories(
                n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=task_id * 100
            )
            strategy.train_task(trajs, steps=5, batch_size=2)
            strategy.after_task(trajs)

        for task_id in range(3):
            assert strategy.task_to_copy[task_id] == task_id
