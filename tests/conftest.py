"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest

from tsn_affinity.data.schemas.trajectory import Trajectory


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_trajectories(rng):
    def _make(n_trajs=3, traj_len=20, obs_dim=4, n_actions=2, seed=42):
        local_rng = np.random.default_rng(seed)
        trajs = []
        for _ in range(n_trajs):
            obs = local_rng.normal(0.5, 0.25, (traj_len, obs_dim)).astype(np.float32)
            obs = np.clip(obs, 0, 1)
            actions = local_rng.integers(0, n_actions, size=traj_len).astype(np.int64)
            rewards = local_rng.standard_normal(traj_len).astype(np.float32) * 0.1
            timesteps = np.arange(traj_len, dtype=np.float32)
            returns_to_go = np.cumsum(rewards[::-1])[::-1].astype(np.float32)
            trajs.append(Trajectory(obs, actions, rewards, timesteps, returns_to_go))
        return trajs

    return _make


@pytest.fixture
def model_config():
    from tsn_affinity.core.config import ModelConfig

    return ModelConfig(obs_shape=(4,), n_actions=2, seq_len=10)


@pytest.fixture
def sparse_config():
    from tsn_affinity.core.config import SparseConfig

    return SparseConfig(keep_ratio=0.5)
