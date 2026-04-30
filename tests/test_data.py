"""Tests for data module (trajectory and batch generation)."""

import numpy as np
import torch
import pytest

from tsn_affinity.data.trajectory import Trajectory, discount_cumsum
from tsn_affinity.data.batch_generator import make_minibatches, masked_cross_entropy, masked_mse


class TestTrajectory:
    def test_init(self):
        traj = Trajectory(
            obs=np.random.randn(100, 4),
            actions=np.random.randint(0, 2, 100),
            rewards=np.random.randn(100),
            timesteps=np.arange(100),
            returns_to_go=np.random.randn(100),
        )
        assert len(traj) == 100

    def test_len(self):
        traj = Trajectory(
            obs=np.random.randn(50, 4),
            actions=np.random.randint(0, 2, 50),
            rewards=np.random.randn(50),
            timesteps=np.arange(50),
            returns_to_go=np.random.randn(50),
        )
        assert len(traj) == 50


class TestDiscountCumsum:
    def test_no_discount(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        out = discount_cumsum(x, gamma=1.0)
        expected = np.array([10.0, 9.0, 7.0, 4.0])
        np.testing.assert_array_almost_equal(out, expected)

    def test_with_discount(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        out = discount_cumsum(x, gamma=0.9)
        expected = np.array([1.0 + 0.9*2 + 0.9**2*3 + 0.9**3*4,
                            2.0 + 0.9*3 + 0.9**2*4,
                            3.0 + 0.9*4,
                            4.0])
        np.testing.assert_array_almost_equal(out, expected)

    def test_single_element(self):
        x = np.array([5.0])
        out = discount_cumsum(x)
        assert out[0] == 5.0


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