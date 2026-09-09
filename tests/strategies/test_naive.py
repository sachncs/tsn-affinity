"""Tests for NaiveStrategy."""

from tsn_affinity.strategies.naive import NaiveStrategy


class TestNaiveStrategy:
    def test_inherits_from_base(self):
        strategy = NaiveStrategy(obs_shape=(4,), n_actions=2, device="cpu")
        from tsn_affinity.strategies.base import BaseStrategy

        assert isinstance(strategy, BaseStrategy)

    def test_train_task(self, simple_trajectories):
        strategy = NaiveStrategy(obs_shape=(4,), n_actions=2, device="cpu")
        trajs = simple_trajectories(
            n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0
        )
        result = strategy.train_task(trajs, steps=5, batch_size=2)
        assert "loss" in result
        assert "keep_ratio" in result

    def test_has_task_mask_returns_false(self):
        strategy = NaiveStrategy(obs_shape=(4,), n_actions=2, device="cpu")
        assert not strategy.has_task_mask(0)
