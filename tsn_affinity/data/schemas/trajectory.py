"""Trajectory data structure and discount cumsum utility."""

from __future__ import annotations

import numpy as np


class Trajectory:
    """Container for a single episode trajectory.

    Attributes:
        obs: Observation sequence (numpy array).
        actions: Action sequence (numpy array).
        rewards: Reward sequence (numpy array).
        timesteps: Timestep indices (numpy array).
        returns_to_go: Returns-to-go targets (numpy array).
    """

    def __init__(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        timesteps: np.ndarray,
        returns_to_go: np.ndarray,
    ) -> None:
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.timesteps = timesteps
        self.returns_to_go = returns_to_go

    def __len__(self) -> int:
        return len(self.actions)


def discount_cumsum(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Compute discounted cumulative sum.

    Computes the return (cumulative discounted reward) at each timestep:
    G_t = sum_{t'=t}^{T} gamma^{t'-t} * r_{t'}

    Args:
        x: Array of values (e.g., rewards).
        gamma: Discount factor (default 1.0 = no discount).

    Returns:
        Array of discounted cumulative sums with same shape as input.
    """
    out = np.zeros_like(x, dtype=np.float32)
    run = 0.0
    for t in reversed(range(len(x))):
        run = x[t] + gamma * run
        out[t] = run
    return out
