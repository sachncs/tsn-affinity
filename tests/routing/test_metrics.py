"""Tests for routing metrics and affinity computations."""

import numpy as np
import torch

from tsn_affinity.data.schemas.trajectory import Trajectory
from tsn_affinity.routing.metrics import (
    _AffinityBatchLoader,
    compute_action_affinity,
    compute_action_affinity_batch,
    compute_diag_stats,
    compute_hybrid_affinity,
    compute_latent_affinity,
    compute_latent_affinity_batch,
    extract_obs_latents,
    normalize_scores,
)


class TestAffinityBatchLoader:
    def test_empty_trajectories(self):
        loader = _AffinityBatchLoader([], 10, 32, "cpu")
        obs, actions, rtg, ts, mask = loader.next_batch()
        assert obs.shape[1] == 10

    def test_single_trajectory(self):
        obs_data = np.random.randn(10, 4).astype(np.float32)
        traj = Trajectory(
            obs=obs_data,
            actions=np.zeros(10, dtype=np.int64),
            rewards=np.zeros(10, dtype=np.float32),
            timesteps=np.arange(10, dtype=np.float32),
            returns_to_go=np.zeros(10, dtype=np.float32),
        )
        loader = _AffinityBatchLoader([traj], 5, 8, "cpu")
        obs, actions, rtg, ts, mask = loader.next_batch()
        assert obs.shape == (8, 5, 4)
        assert actions.shape == (8, 5)


class TestComputeDiagStats:
    def test_basic(self):
        z = torch.randn(10, 5, 32)
        mu, var = compute_diag_stats(z)
        assert mu.shape == (32,)
        assert var.shape == (32,)

    def test_with_mask(self):
        z = torch.randn(10, 5, 32)
        mask = torch.ones(10, 5)
        mask[:3, :] = 0
        mu, var = compute_diag_stats(z, mask)
        assert mu.shape == (32,)

    def test_empty_input(self):
        z = torch.randn(0, 32)
        mu, var = compute_diag_stats(z)
        assert mu.shape == (32,)
        assert var.shape == (32,)

    def test_3d_input_with_mask(self):
        z = torch.randn(8, 4, 16)
        mask = torch.ones(8, 4)
        mask[:2, :] = 0
        mu, var = compute_diag_stats(z, mask)
        assert mu.shape == (16,)
        assert var.shape == (16,)


class TestComputeHybridAffinity:
    def test_basic(self):
        score = compute_hybrid_affinity(6.0, 12.5, 12.0, 25.0, 0.7)
        assert score >= 0.0

    def test_action_only_contribution(self):
        score = compute_hybrid_affinity(12.0, 100.0, 12.0, 25.0, 0.7)
        assert score >= 0.0

    def test_zero_thresholds_handled(self):
        score = compute_hybrid_affinity(1.0, 1.0, 0.0, 0.0, 0.5)
        assert score >= 0

    def test_high_alpha_favors_action(self):
        score_action = compute_hybrid_affinity(0.5, 10.0, 1.0, 10.0, 0.9)
        score_latent = compute_hybrid_affinity(10.0, 0.5, 1.0, 10.0, 0.1)
        assert score_action < score_latent


class TestNormalizeScores:
    def test_empty_dict(self):
        result = normalize_scores({})
        assert result == {}

    def test_single_element(self):
        result = normalize_scores({0: 5.0})
        assert result[0] == 0.0

    def test_normalizes_to_range(self):
        scores = {0: 1.0, 1: 5.0, 2: 10.0}
        result = normalize_scores(scores)
        assert 0.0 <= result[0] <= 1.0
        assert 0.0 <= result[1] <= 1.0
        assert 0.0 <= result[2] <= 1.0
        assert abs(result[2] - 1.0) < 1e-6

    def test_identical_values(self):
        scores = {0: 5.0, 1: 5.0, 2: 5.0}
        result = normalize_scores(scores)
        assert all(v == 0.0 for v in result.values())

    def test_all_zeros(self):
        scores = {0: 0.0, 1: 0.0}
        result = normalize_scores(scores)
        assert all(v == 0.0 for v in result.values())


class TestExtractObsLatents:
    def test_3d_observations(self):
        obs = torch.randn(2, 5, 10)
        mock_encoder = torch.nn.Linear(10, 8)
        z = extract_obs_latents(obs, mock_encoder)
        assert z.shape == (2, 5, 8)

    def test_5d_observations(self):
        obs = torch.randn(2, 5, 3, 8, 8)
        mock_encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 8 * 8, 32),
        )
        z = extract_obs_latents(obs, mock_encoder)
        assert z.shape == (2, 5, 32)

    def test_uint8_input_normalized(self):
        obs = torch.randint(0, 256, (2, 5, 3, 8, 8), dtype=torch.uint8)
        mock_encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 8 * 8, 32),
        )
        z = extract_obs_latents(obs, mock_encoder)
        assert z.shape == (2, 5, 32)


class _MockDTModel(torch.nn.Module):
    def __init__(self, out_dim=2):
        super().__init__()
        self.fc = torch.nn.Linear(4, out_dim)

    def forward(self, obs, actions, rtg, ts, attention_mask=None):
        return self.fc(obs)


class TestComputeActionAffinity:
    def test_empty_trajectories(self):
        mock_model = torch.nn.Identity()
        score = compute_action_affinity(
            model=mock_model,
            task_trajs=[],
            seq_len=10,
            n_batches=1,
            batch_size=4,
            device="cpu",
        )
        assert score == float("inf")

    def test_with_trajectories(self, simple_trajectories):
        trajs = simple_trajectories(
            n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0
        )
        mock_model = _MockDTModel(out_dim=2)
        score = compute_action_affinity(
            model=mock_model,
            task_trajs=trajs,
            seq_len=5,
            n_batches=1,
            batch_size=2,
            device="cpu",
        )
        assert isinstance(score, float)


class TestComputeLatentAffinity:
    def test_missing_stats_returns_inf(self):
        score = compute_latent_affinity(
            source_task_id=0,
            task_memory_obs=torch.randn(10, 4),
            obs_encoder=torch.nn.Identity(),
            stored_stats={},
            device="cpu",
        )
        assert score == float("inf")

    def test_with_valid_stats(self):
        stored_stats = {
            0: (torch.zeros(8), torch.ones(8)),
        }
        encoder = torch.nn.Linear(4, 8)
        score = compute_latent_affinity(
            source_task_id=0,
            task_memory_obs=torch.randn(10, 4),
            obs_encoder=encoder,
            stored_stats=stored_stats,
            device="cpu",
        )
        assert isinstance(score, float)
        assert score >= 0.0


class TestComputeActionAffinityBatch:
    def test_empty_trajectories(self):
        scores = compute_action_affinity_batch(
            models=[torch.nn.Identity()],
            task_ids=[0],
            task_trajs=[],
            seq_len=10,
            n_batches=1,
            batch_size=4,
            device="cpu",
        )
        assert scores[0] == float("inf")

    def test_with_trajectories(self, simple_trajectories):
        trajs = simple_trajectories(
            n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0
        )
        model = _MockDTModel(out_dim=2)
        scores = compute_action_affinity_batch(
            models=[model],
            task_ids=[0],
            task_trajs=trajs,
            seq_len=5,
            n_batches=1,
            batch_size=2,
            device="cpu",
        )
        assert isinstance(scores[0], float)


class TestComputeLatentAffinityBatch:
    def test_empty_observations(self):
        scores = compute_latent_affinity_batch(
            task_memory_obs=torch.zeros(0, 4),
            obs_encoder=torch.nn.Identity(),
            stored_stats={},
            target_task_ids=[0],
            device="cpu",
        )
        assert scores[0] == float("inf")

    def test_missing_stats(self):
        scores = compute_latent_affinity_batch(
            task_memory_obs=torch.randn(10, 4),
            obs_encoder=torch.nn.Linear(4, 8),
            stored_stats={},
            target_task_ids=[0],
            device="cpu",
        )
        assert scores[0] == float("inf")

    def test_with_valid_stats(self):
        stored_stats = {
            0: (torch.zeros(8), torch.ones(8)),
        }
        scores = compute_latent_affinity_batch(
            task_memory_obs=torch.randn(10, 4),
            obs_encoder=torch.nn.Linear(4, 8),
            stored_stats=stored_stats,
            target_task_ids=[0],
            device="cpu",
        )
        assert isinstance(scores[0], float)


class TestMultiBatchLoader:
    def test_empty_trajectories(self):
        from tsn_affinity.routing.metrics import _MultiBatchLoader

        loader = _MultiBatchLoader([], 10, 4, "cpu", 2)
        batches = list(loader)
        assert len(batches) == 1
        obs, actions, rtg, ts, mask = batches[0]
        assert obs.shape[1] == 10

    def test_with_trajectories(self, simple_trajectories):
        from tsn_affinity.routing.metrics import _MultiBatchLoader

        trajs = simple_trajectories(
            n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0
        )
        loader = _MultiBatchLoader(trajs, 5, 2, "cpu", 2)
        batches = list(loader)
        assert len(batches) == 2
        obs, actions, rtg, ts, mask = batches[0]
        assert obs.shape == (2, 5, 4)


class TestMakeTempMinibatches:
    def test_yields_batches(self, simple_trajectories):
        from tsn_affinity.routing.metrics import make_temp_minibatches

        trajs = simple_trajectories(
            n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0
        )
        gen = make_temp_minibatches(trajs, 5, 2, "cpu")
        obs, actions, rtg, ts = next(gen)
        assert obs.shape == (2, 5, 4)
        assert actions.shape == (2, 5)


class TestUnpackTempBatch:
    def test_unpacks_4_tuple(self, simple_trajectories):
        from tsn_affinity.routing.metrics import (
            make_temp_minibatches,
            unpack_temp_batch,
        )

        trajs = simple_trajectories(
            n_trajs=2, traj_len=20, obs_dim=4, n_actions=2, seed=0
        )
        gen = make_temp_minibatches(trajs, 5, 2, "cpu")
        batch = next(gen)
        result = unpack_temp_batch(batch)
        assert len(result) == 5
