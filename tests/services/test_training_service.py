"""Tests for TrainingService."""

from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.services.training_service import TrainingService
from tsn_affinity.strategies.tsn_core import TSNCoreStrategy


class TestTrainingService:
    def test_train_tasks(self, simple_trajectories):
        strategy = TSNCoreStrategy(
            obs_shape=(4,),
            n_actions=2,
            seq_len=10,
            device="cpu",
            model_config=ModelConfig(obs_shape=(4,), n_actions=2),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )
        service = TrainingService(strategy)

        task_trajs = [
            simple_trajectories(
                n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=i * 100
            )
            for i in range(2)
        ]
        results = service.train_tasks(task_trajs, steps=5, batch_size=2)

        assert 0 in results
        assert 1 in results
        assert strategy.current_task_id == 2
