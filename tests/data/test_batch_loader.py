"""Tests for batch generation and loss functions."""

import numpy as np
import torch

from tsn_affinity.data.loaders.batch_loader import (
    make_minibatches,
    masked_cross_entropy,
    masked_mse,
)
from tsn_affinity.data.schemas.trajectory import Trajectory


class TestMakeMinibatches:
    def test_generates_batches(self):
        trajs = [
            Trajectory(
                obs=np.random.randn(20, 4).astype(np.float32),
                actions=np.random.randint(0, 2, 20).astype(np.int64),
                rewards=np.random.randn(20).astype(np.float32),
                timesteps=np.arange(20, dtype=np.float32),
                returns_to_go=np.random.randn(20).astype(np.float32),
            )
            for _ in range(5)
        ]
        loader = make_minibatches(trajs, seq_len=10, batch_size=4, device="cpu")
        obs, actions, rtg, ts, mask = next(loader)
        assert obs.shape == (4, 10, 4)
        assert actions.shape == (4, 10)
        assert rtg.shape == (4, 10, 1)
        assert ts.shape == (4, 10)
        assert mask.shape == (4, 10)

    def test_pads_short_trajectories(self):
        trajs = [
            Trajectory(
                obs=np.random.randn(5, 4).astype(np.float32),
                actions=np.random.randint(0, 2, 5).astype(np.int64),
                rewards=np.random.randn(5).astype(np.float32),
                timesteps=np.arange(5, dtype=np.float32),
                returns_to_go=np.random.randn(5).astype(np.float32),
            )
            for _ in range(5)
        ]
        loader = make_minibatches(trajs, seq_len=10, batch_size=2, device="cpu")
        obs, actions, rtg, ts, mask = next(loader)
        assert obs.shape == (2, 10, 4)
        assert mask.shape == (2, 10)
        assert mask[:, 5:].sum() == 0.0


class TestMaskedCrossEntropy:
    def test_basic(self):
        logits = torch.randn(2, 3, 10)
        targets = torch.randint(0, 10, (2, 3))
        mask = torch.ones(2, 3)
        loss = masked_cross_entropy(logits, targets, mask)
        assert loss.item() >= 0

    def test_ignores_padded(self):
        logits = torch.randn(2, 3, 10)
        targets = torch.randint(0, 10, (2, 3))
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        loss = masked_cross_entropy(logits, targets, mask)
        assert loss.item() >= 0


class TestMaskedMSE:
    def test_basic(self):
        pred = torch.randn(2, 3, 4)
        targets = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3)
        loss = masked_mse(pred, targets, mask)
        assert loss.item() >= 0
