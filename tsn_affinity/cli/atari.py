#!/usr/bin/env python3
"""Entry point for running Atari benchmark with TSN-Affinity.

Usage:
    python -m scripts.run_atari --strategy tsn_affinity --output runs/atari_tsn_affinity
"""

import argparse
import json
import logging
import os

import gymnasium as gym
import numpy as np
import torch

from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.core.logging_config import setup_logging
from tsn_affinity.data.trajectory import Trajectory, discount_cumsum
from tsn_affinity.run.analysis import compute_final_metrics
from tsn_affinity.strategies.base import BaseStrategy
from tsn_affinity.strategies.tsn_affinity import (
    AffinityRoutingConfig,
    TSNAffinityStrategy,
)
from tsn_affinity.strategies.tsn_core import TSNCoreStrategy

logger = logging.getLogger("tsn_affinity")

# The 5 Atari games from the paper
ATARI_GAMES = [
    "Breakout",  # Task 0
    "Alien",  # Task 1
    "Atlantis",  # Task 2
    "Boxing",  # Task 3
    "Centipede",  # Task 4
]


def preprocess_atari_frame(frame: np.ndarray, target_size: int = 84) -> np.ndarray:
    """Preprocess Atari frame: resize and convert to channels-first.

    Args:
        frame: Original frame (H, W, C) in RGB.
        target_size: Target width/height after resize.

    Returns:
        Preprocessed frame (C, H, W) in channels-first format for PyTorch.
    """
    from PIL import Image

    img = Image.fromarray(frame)
    img = img.resize((target_size, target_size), Image.LANCZOS)  # type: ignore[attr-defined]
    arr = np.array(img, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def collect_atari_trajectories(
    game: str,
    n_trajectories: int,
    max_steps_per_trajectory: int,
    target_size: int = 84,
    seed: int = 0,
) -> list[Trajectory]:
    """Collect trajectories from an Atari environment using random policy.

    Args:
        game: Atari game name.
        n_trajectories: Number of trajectories to collect.
        max_steps_per_trajectory: Maximum steps per trajectory.
        target_size: Target frame size for preprocessing.
        seed: Random seed for environment.

    Returns:
        List of Trajectory objects.
    """
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


def greedy_rollout(
    strategy,
    env,
    max_steps: int = 2000,
) -> float:
    """Run greedy action selection rollout and return total return.

    Uses the Decision Transformer's act() method for online inference.
    """
    strategy.model.reset_history()
    strategy.model.eval()

    obs, _ = env.reset()
    obs = preprocess_atari_frame(obs)
    obs = torch.as_tensor(obs, dtype=torch.float32)

    total_return = 0.0
    cumulative_return = 0.0

    n_env_actions = env.action_space.n

    for _step in range(max_steps):
        with torch.no_grad():
            action = strategy.model.act(obs, cumulative_return, deterministic=True)
            action = max(0, min(action, n_env_actions - 1))

        obs, reward, terminated, truncated, _ = env.step(action)
        obs = preprocess_atari_frame(obs)
        obs = torch.as_tensor(obs, dtype=torch.float32)

        cumulative_return += reward
        total_return += reward

        if terminated or truncated:
            break

    return float(total_return)


def compute_normalized_score(
    agent_return: float,
    random_return: float,
    human_return: float,
) -> float:
    """Compute normalized score as in the paper.

    Score = (agent - random) / (human - random)
    Higher is better (1.0 = human, 0.0 = random)
    """
    denominator = human_return - random_return
    if denominator <= 0:
        return 0.0
    return (agent_return - random_return) / denominator


def run_atari_benchmark(
    strategy_name: str,
    output_dir: str,
    device: str = "cpu",
    n_trajectories: int = 10,
    max_steps: int = 2000,
    train_steps: int = 2000,
    eval_steps: int = 100,
    n_eval_runs: int = 3,
):
    """Run Atari benchmark with real game data.

    The 5 games from the paper: Breakout -> Alien -> Atlantis -> Boxing -> Centipede
    """
    n_tasks = len(ATARI_GAMES)

    logger.info("Collecting Atari trajectories...")
    task_trajs_all = []
    obs_shapes = []
    n_actions_list = []

    for task_id, game in enumerate(ATARI_GAMES):
        logger.info("  Collecting %d trajectories for %s...", n_trajectories, game)
        trajs = collect_atari_trajectories(
            game=game,
            n_trajectories=n_trajectories,
            max_steps_per_trajectory=max_steps,
            target_size=84,
            seed=task_id * 1000,
        )
        task_trajs_all.append(trajs)

        obs_shape = tuple(trajs[0].obs[0].shape)
        n_actions = int(trajs[0].actions.max()) + 1
        obs_shapes.append(obs_shape)
        n_actions_list.append(n_actions)
        logger.info("    obs_shape=%s, n_actions=%d", obs_shape, n_actions)

    obs_shape = (3, 84, 84)
    n_actions = 18

    logger.info("Running Atari benchmark with strategy: %s", strategy_name)
    logger.info("  Games: %s", ATARI_GAMES)
    logger.info("  Trajectories per game: %d", n_trajectories)
    logger.info("  Max steps per trajectory: %d", max_steps)
    logger.info("  Training steps per task: %d", train_steps)

    if strategy_name == "tsn_affinity":
        affinity_config = AffinityRoutingConfig(
            mode="action",
            action_threshold=12.0,
            routing_n_batches=4,
            routing_batch_size=64,
        )
        strategy: BaseStrategy = TSNAffinityStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=20,
            device=device,
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
            affinity_config=affinity_config,
        )
    elif strategy_name == "tsn_core":
        strategy = TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=20,
            device=device,
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    human_baselines = {
        "Breakout": 31.8,
        "Alien": 6867.0,
        "Atlantis": 29009.0,
        "Boxing": 71.8,
        "Centipede": 4016.0,
    }
    random_baselines = {
        "Breakout": 1.7,
        "Alien": 227.0,
        "Atlantis": 369.0,
        "Boxing": -3.6,
        "Centipede": 148.0,
    }

    logger.info("Computing random baseline returns...")
    random_returns = []
    for task_id in range(n_tasks):
        game = ATARI_GAMES[task_id]
        trajs = task_trajs_all[task_id]
        mean_return = float(np.mean([traj.rewards.sum() for traj in trajs]))
        random_returns.append(mean_return)
        logger.info(
            "  %s: random_return=%.1f (baseline=%.1f)",
            game,
            mean_return,
            random_baselines[game],
        )

    performance_matrix = np.zeros((n_tasks, n_tasks))
    returns_matrix = np.zeros((n_tasks, n_tasks))
    task_similarity = {}

    for task_id in range(n_tasks):
        logger.info("--- Task %d: %s ---", task_id, ATARI_GAMES[task_id])

        strategy.train_task(task_trajs_all[task_id], steps=train_steps, batch_size=32)
        strategy.after_task(task_trajs_all[task_id])

        for eval_task_id in range(n_tasks):
            game = ATARI_GAMES[eval_task_id]
            strategy.set_eval_task(eval_task_id)

            eval_rollouts = 3
            total_returns = []
            for _ in range(eval_rollouts):
                env = gym.make(game + "NoFrameskip-v4")
                ret = greedy_rollout(strategy, env, max_steps=1000)
                env.close()
                total_returns.append(ret)

            agent_return = float(np.mean(total_returns))
            returns_matrix[task_id, eval_task_id] = agent_return

            human_ret = human_baselines[game]
            random_ret = random_baselines[game]
            normalized = compute_normalized_score(agent_return, random_ret, human_ret)
            performance_matrix[task_id, eval_task_id] = normalized

        if hasattr(strategy, "task_similarity"):
            task_similarity[task_id] = strategy.task_similarity.get(task_id, {})

        strategy.clear_eval_task()

        logger.info(
            "  Task %d diagonal: return=%.1f normalized=%.4f",
            task_id,
            returns_matrix[task_id, task_id],
            performance_matrix[task_id, task_id],
        )

    logger.info("=" * 60)
    logger.info("Benchmark Results (NormAvg - Human-Normalized)")
    logger.info("=" * 60)

    metrics = compute_final_metrics(performance_matrix)
    logger.info("  Strategy: %s", strategy_name)
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name.upper(), value)

    logger.info("Raw Returns Matrix:")
    for i, game in enumerate(ATARI_GAMES):
        logger.info("  %s: %s", game, returns_matrix[i, :])

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "performance_matrix.npy"), performance_matrix)
    np.save(os.path.join(output_dir, "returns_matrix.npy"), returns_matrix)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(
            {
                "strategy": strategy_name,
                "games": ATARI_GAMES,
                "metrics": {k: float(v) for k, v in metrics.items()},
                "human_baselines": human_baselines,
                "random_baselines": random_baselines,
                "random_returns": [float(x) for x in random_returns],
                "returns_matrix": returns_matrix.tolist(),
                "task_similarity": task_similarity,
            },
            f,
            indent=2,
        )

    logger.info("Results saved to: %s", output_dir)
    return metrics


def main() -> None:
    """Run the Atari TSN-Affinity benchmark from the command line."""
    parser = argparse.ArgumentParser(description="Run Atari TSN-Affinity benchmark")
    parser.add_argument(
        "--strategy",
        type=str,
        default="tsn_affinity",
        choices=["tsn_affinity", "tsn_core"],
        help="Strategy name",
    )
    parser.add_argument(
        "--output", type=str, default="runs/atari_tsn_affinity", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda/cpu)")
    parser.add_argument(
        "--n-trajectories", type=int, default=10, help="Number of trajectories per game"
    )
    parser.add_argument(
        "--max-steps", type=int, default=2000, help="Max steps per trajectory"
    )
    parser.add_argument(
        "--train-steps", type=int, default=2000, help="Training steps per task"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=100, help="Eval steps per task"
    )

    args = parser.parse_args()

    setup_logging()

    run_atari_benchmark(
        args.strategy,
        args.output,
        args.device,
        args.n_trajectories,
        args.max_steps,
        args.train_steps,
        args.eval_steps,
    )


if __name__ == "__main__":
    main()
