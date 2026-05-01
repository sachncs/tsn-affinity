"""Tests for trajectory schemas and utilities."""

import numpy as np

from tsn_affinity.data.schemas.trajectory import Trajectory, discount_cumsum


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
        expected = np.array(
            [
                1.0 + 0.9 * 2 + 0.9**2 * 3 + 0.9**3 * 4,
                2.0 + 0.9 * 3 + 0.9**2 * 4,
                3.0 + 0.9 * 4,
                4.0,
            ]
        )
        np.testing.assert_array_almost_equal(out, expected)

    def test_single_element(self):
        x = np.array([5.0])
        out = discount_cumsum(x)
        assert out[0] == 5.0
