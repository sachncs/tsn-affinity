"""Tests for affinity routing metrics and edge cases."""

import torch
import numpy as np
import pytest

from tsn_affinity.routing.affinity_metrics import (
    compute_action_affinity,
    compute_latent_affinity,
    compute_hybrid_affinity,
    normalize_scores,
    compute_diag_stats,
    extract_obs_latents,
    _AffinityBatchLoader,
)
from tsn_affinity.data.trajectory import Trajectory


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
    def test_3d_input_with_mask(self):
        z = torch.randn(8, 4, 16)
        mask = torch.ones(8, 4)
        mask[:2, :] = 0
        mu, var = compute_diag_stats(z, mask)
        assert mu.shape == (16,)
        assert var.shape == (16,)

    def test_all_masked(self):
        z = torch.randn(8, 4, 16)
        mask = torch.zeros(8, 4)
        mu, var = compute_diag_stats(z, mask)
        assert mu.shape == (16,)
        assert var.shape == (16,)


class TestComputeHybridAffinity:
    def test_zero_thresholds_handled(self):
        score = compute_hybrid_affinity(1.0, 1.0, 0.0, 0.0, 0.5)
        assert score >= 0

    def test_high_alpha_favors_action(self):
        score_action = compute_hybrid_affinity(0.5, 10.0, 1.0, 10.0, 0.9)
        score_latent = compute_hybrid_affinity(10.0, 0.5, 1.0, 10.0, 0.1)
        assert score_action < score_latent


class TestNormalizeScores:
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


class TestComputeActionAffinity:
    def test_empty_trajectories(self):
        mock_model = torch.nn.Identity()
        score = compute_action_affinity(
            model=mock_model,
            source_task_id=0,
            task_trajs=[],
            obs_encoder=torch.nn.Identity(),
            seq_len=10,
            n_batches=1,
            batch_size=4,
            device="cpu",
        )
        assert score == float("inf")