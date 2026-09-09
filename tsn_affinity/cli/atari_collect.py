#!/usr/bin/env python3
r"""Collect trajectories from a single Atari game using a random policy.

This CLI deliberately collects random-policy rollouts because
TSN-Affinity is an offline reinforcement-learning algorithm. The
resulting dataset is suitable as a starting point for offline training,
*not* as a learned benchmark in itself. See ``scripts/run_atari_benchmark.py``
for the end-to-end training, evaluation, and human-normalized scoring
pipeline.

Usage:
    python -m tsn_affinity.cli.atari_collect \\
        --game Breakout --output runs/breakout_random
"""

import argparse
import json
import logging
import os
from typing import Any

import numpy as np

from tsn_affinity.core.logging_config import setup_logging
from tsn_affinity.data.trajectory import Trajectory, discount_cumsum

logger = logging.getLogger("tsn_affinity")


def _require_gymnasium() -> Any:
    """Import gymnasium and raise a helpful error if it is not installed.

    Returns:
        The imported ``gymnasium`` module.

    Raises:
        ImportError: When gymnasium is not installed.
    """
    import gymnasium as gym

    return gym


def preprocess_atari_frame(frame: np.ndarray, target_size: int = 84) -> np.ndarray:
    """Preprocess Atari frame: resize and convert to channels-first.

    Args:
        frame: Original frame ``(H, W, C)`` in RGB.
        target_size: Target width and height after resize.

    Returns:
        Preprocessed frame ``(C, H, W)`` in channels-first format.
    """
    from PIL import Image

    img = Image.fromarray(frame)
    img = img.resize((target_size, target_size), Image.LANCZOS)  # type: ignore[attr-defined]
    arr = np.array(img, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def collect_random_trajectories(
    game: str,
    n_trajectories: int,
    max_steps_per_trajectory: int,
    target_size: int = 84,
    seed: int = 0,
) -> list[Trajectory]:
    """Collect trajectories from an Atari environment using a random policy.

    Args:
        game: Atari game name (bare, for example ``"Breakout"``).
        n_trajectories: Number of trajectories to collect.
        max_steps_per_trajectory: Maximum steps per trajectory.
        target_size: Target frame size for preprocessing.
        seed: Random seed for the environment.

    Returns:
        List of :class:`Trajectory` objects.
    """
    gym = _require_gymnasium()
    env = gym.make(game + "NoFrameskip-v4")
    env.reset(seed=seed)
    env.action_space.seed(seed)

    trajectories = []
    for _traj_idx in range(n_trajectories):
        obs, _ = env.reset()
        obs_history = [preprocess_atari_frame(obs, target_size)]
        actions_history = []
        rewards_history = []
        timesteps_history = []

        for _step in range(max_steps_per_trajectory):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)

            actions_history.append(action)
            rewards_history.append(reward)
            timesteps_history.append(float(_step))
            obs_history.append(preprocess_atari_frame(obs, target_size))

            if terminated or truncated:
                break

        obs_arr = np.array(obs_history[:-1], dtype=np.uint8)
        actions_arr = np.array(actions_history, dtype=np.int64)
        rewards_arr = np.array(rewards_history, dtype=np.float32)
        timesteps_arr = np.array(timesteps_history, dtype=np.float32)
        returns_to_go = discount_cumsum(rewards_arr, gamma=0.99)

        trajectories.append(
            Trajectory(
                obs=obs_arr,
                actions=actions_arr,
                rewards=rewards_arr,
                timesteps=timesteps_arr,
                returns_to_go=returns_to_go,
            )
        )

    env.close()
    return trajectories


def main() -> None:
    """Collect random-policy trajectories and save them as a JSON manifest."""
    parser = argparse.ArgumentParser(
        description=(
            "Collect random-policy trajectories from a single Atari game. "
            "Use this to build an offline dataset; run a learned "
            "policy with `tsn-atari-eval` to score it."
        )
    )
    parser.add_argument(
        "--game",
        type=str,
        required=True,
        help="Atari game name (bare, for example 'Breakout').",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=10,
        help="Number of trajectories to collect.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum steps per trajectory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the trajectory manifest.",
    )

    args = parser.parse_args()

    setup_logging()

    trajs = collect_random_trajectories(
        game=args.game,
        n_trajectories=args.n_trajectories,
        max_steps_per_trajectory=args.max_steps,
        seed=args.seed,
    )

    os.makedirs(args.output, exist_ok=True)
    manifest = {
        "game": args.game,
        "n_trajectories": len(trajs),
        "max_steps": args.max_steps,
        "seed": args.seed,
        "policy": "random",
        "trajectories": [
            {
                "obs": traj.obs.shape,
                "actions": traj.actions.tolist(),
                "rewards": traj.rewards.tolist(),
                "returns_to_go": traj.returns_to_go.tolist(),
            }
            for traj in trajs
        ],
    }
    with open(os.path.join(args.output, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "Collected %d random trajectories from %s to %s",
        len(trajs),
        args.game,
        args.output,
    )


if __name__ == "__main__":
    main()
