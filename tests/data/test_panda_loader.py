"""Tests for Panda data loader."""

import pickle

import numpy as np
import torch

from tsn_affinity.data.loaders.panda_loader import (
    load_panda_offline_pkl,
    make_minibatches_panda,
    unpack_batch_continuous,
)
from tsn_affinity.data.schemas.trajectory import Trajectory


class TestLoadPandaOfflinePkl:
    def test_loads_trajectories(self, tmp_path):
        data = [
            {
                "obs": np.random.randn(10, 4).astype(np.float32),
                "actions": np.random.randn(10, 2).astype(np.float32),
                "rewards": np.random.randn(10).astype(np.float32),
            }
        ]
        pkl_path = tmp_path / "data.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        trajs = load_panda_offline_pkl(str(pkl_path))
        assert len(trajs) == 1
        assert isinstance(trajs[0], Trajectory)
        assert trajs[0].obs.shape == (10, 4)
        assert trajs[0].actions.shape == (10, 2)


class TestMakeMinibatchesPanda:
    def test_yields_batches(self):
        obs = np.random.randn(20, 4).astype(np.float32)
        actions = np.random.randn(20, 2).astype(np.float32)
        rewards = np.random.randn(20).astype(np.float32)
        timesteps = np.arange(20, dtype=np.float32)
        returns_to_go = np.cumsum(rewards[::-1])[::-1].astype(np.float32)
        traj = Trajectory(obs, actions, rewards, timesteps, returns_to_go)

        gen = make_minibatches_panda([traj], seq_len=5, batch_size=2, device="cpu")
        obs_t, actions_t, rtg_t, ts_t, mask_t = next(gen)

        assert obs_t.shape == (2, 5, 4)
        assert actions_t.shape == (2, 5, 2)
        assert rtg_t.shape == (2, 5, 1)
        assert ts_t.shape == (2, 5)
        assert mask_t.shape == (2, 5)


class TestUnpackBatchContinuous:
    def test_unpacks_5_tuple(self):
        batch = (
            torch.randn(2, 5, 4),
            torch.randn(2, 5, 2),
            torch.randn(2, 5, 1),
            torch.zeros(2, 5, dtype=torch.long),
            torch.ones(2, 5),
        )
        result = unpack_batch_continuous(batch)
        assert len(result) == 5
        assert torch.equal(result[4], batch[4])

    def test_unpacks_4_tuple(self):
        batch = (
            torch.randn(2, 5, 4),
            torch.randn(2, 5, 2),
            torch.randn(2, 5, 1),
            torch.zeros(2, 5, dtype=torch.long),
        )
        result = unpack_batch_continuous(batch)
        assert len(result) == 5
        assert result[4].shape == (2, 5)
        assert torch.all(result[4] == 1.0)
