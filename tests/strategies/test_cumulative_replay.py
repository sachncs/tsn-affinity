"""Tests for CumulativeReplayStrategy."""

from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.strategies.cumulative_replay import CumulativeReplayStrategy


class TestCumulativeReplayStrategy:
    def test_train_task(self, simple_trajectories):
        strategy = CumulativeReplayStrategy(
            obs_shape=(4,),
            n_actions=2,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=(4,), n_actions=2),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )

        trajs = simple_trajectories(
            n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0
        )
        result = strategy.train_task(trajs, steps=5, batch_size=2)
        assert "loss" in result
        assert "keep_ratio" in result

    def test_replay_buffer_populated(self, simple_trajectories):
        strategy = CumulativeReplayStrategy(
            obs_shape=(4,),
            n_actions=2,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=(4,), n_actions=2),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )

        trajs = simple_trajectories(
            n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0
        )
        strategy.train_task(trajs, steps=5, batch_size=2)
        strategy.after_task(trajs)

        assert len(strategy.replay_buffer) > 0
