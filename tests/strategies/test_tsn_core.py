"""Tests for TSNCoreStrategy."""

from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.strategies.tsn_core import TSNCoreStrategy


class TestTSNCoreStrategy:
    def test_train_task_basic(self, simple_trajectories):
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

        trajs = simple_trajectories(
            n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0
        )
        result = strategy.train_task(trajs, steps=5, batch_size=2)

        assert "loss" in result
        assert "keep_ratio" in result
        assert strategy.current_task_id == 0

    def test_after_task_collects_masks(self, simple_trajectories):
        strategy = TSNCoreStrategy(
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

        assert 0 in strategy.per_task_masks
        assert 0 in strategy.task_keep_ratios

    def test_multi_task_training(self, simple_trajectories):
        strategy = TSNCoreStrategy(
            obs_shape=(4,),
            n_actions=2,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=(4,), n_actions=2),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )

        for task_id in range(3):
            trajs = simple_trajectories(
                n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=task_id * 100
            )
            strategy.train_task(trajs, steps=5, batch_size=2)
            strategy.after_task(trajs)

        assert strategy.current_task_id == 3
        assert len(strategy.per_task_masks) == 3

    def test_set_eval_task(self, simple_trajectories):
        strategy = TSNCoreStrategy(
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

        strategy.set_eval_task(0)
        assert strategy.active_eval_task == 0

        strategy.clear_eval_task()
        assert strategy.active_eval_task is None
