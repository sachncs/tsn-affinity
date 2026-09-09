#!/usr/bin/env python3
"""End-to-end Atari benchmark for TSN-Affinity.

This script trains a TSN-Affinity strategy on random-policy Atari
trajectories and evaluates it with greedy rollouts. The numbers it
reports are the agent's actual performance on the target environment;
they are not synthetic and are not a marketing claim. Use this CLI to
reproduce the benchmark numbers in the README.

Usage:
    tsn-atari --strategy tsn_affinity --output runs/atari_tsn_affinity
"""

import argparse
import json
import logging
import os
from typing import Any

import numpy as np
import torch

from tsn_affinity.benchmarks.atari_baselines import ATARI_BASELINES, ATARI_GAMES
from tsn_affinity.benchmarks.metrics import StandardCLMetrics
from tsn_affinity.cli.atari_collect import (
    collect_random_trajectories,
    preprocess_atari_frame,
)
from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.core.exceptions import BenchmarkError
from tsn_affinity.core.logging_config import setup_logging
from tsn_affinity.data.trajectory import Trajectory
from tsn_affinity.run.analysis import compute_final_metrics
from tsn_affinity.strategies.base import BaseStrategy
from tsn_affinity.strategies.tsn_affinity import (
    RoutingConfig,
    TSNAffinityStrategy,
)
from tsn_affinity.strategies.tsn_core import TSNCoreStrategy

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


def greedy_rollout(
    strategy: BaseStrategy,
    env: Any,
    max_steps: int = 2000,
) -> float:
    """Run greedy action selection rollout and return total return.

    Uses the Decision Transformer's :meth:`act` method for online
    inference.

    Args:
        strategy: The strategy whose model should be evaluated.
        env: Gymnasium environment to roll out.
        max_steps: Maximum environment steps per rollout.

    Returns:
        Cumulative reward collected during the rollout.
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
            action = max(0, min(int(action), n_env_actions - 1))

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

    The score is ``(agent - random) / (human - random)``. Higher is
    better (1.0 = human, 0.0 = random).

    Args:
        agent_return: Agent's average return.
        random_return: Random-policy baseline return.
        human_return: Human baseline return.

    Returns:
        Human-normalized score in the same scale as the inputs.
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
    n_eval_runs: int = 3,
    seed: int = 0,
) -> dict[str, float]:
    """Run Atari benchmark with real game data.

    The 5 games from the paper: Breakout -> Alien -> Atlantis -> Boxing
    -> Centipede. Random-policy trajectories are collected first, then a
    TSN strategy is trained on them and evaluated with greedy rollouts.

    Args:
        strategy_name: Either ``"tsn_affinity"`` or ``"tsn_core"``.
        output_dir: Directory to save benchmark results.
        device: Torch device (``"cpu"`` or ``"cuda"``).
        n_trajectories: Number of trajectories per game.
        max_steps: Maximum steps per trajectory.
        train_steps: Training steps per task.
        n_eval_runs: Number of greedy rollouts per (task, eval_task)
            pair.
        seed: Random seed.

    Returns:
        Dictionary of computed metrics.
    """
    gym = _require_gymnasium()
    n_tasks = len(ATARI_GAMES)

    logger.info("Collecting Atari trajectories...")
    task_trajs_all: list[list[Trajectory]] = []

    for task_id, game in enumerate(ATARI_GAMES):
        logger.info("  Collecting %d trajectories for %s...", n_trajectories, game)
        trajs = collect_random_trajectories(
            game=game,
            n_trajectories=n_trajectories,
            max_steps_per_trajectory=max_steps,
            target_size=84,
            seed=seed + task_id * 1000,
        )
        task_trajs_all.append(trajs)

    obs_shape = (3, 84, 84)
    n_actions = 18

    logger.info("Running Atari benchmark with strategy: %s", strategy_name)
    logger.info("  Games: %s", ATARI_GAMES)
    logger.info("  Trajectories per game: %d", n_trajectories)
    logger.info("  Max steps per trajectory: %d", max_steps)
    logger.info("  Training steps per task: %d", train_steps)

    if strategy_name == "tsn_affinity":
        affinity_config = RoutingConfig(
            mode="action",
            action_threshold=12.0,
            routing_n_batches=4,
            routing_batch_size=64,
            seed=seed,
        )
        strategy: BaseStrategy = TSNAffinityStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=20,
            device=device,
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.3),
            affinity_config=affinity_config,
            seed=seed,
        )
    elif strategy_name == "tsn_core":
        strategy = TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=20,
            device=device,
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.3),
            seed=seed,
        )
    else:
        raise BenchmarkError(f"Unknown strategy: {strategy_name}")

    human_baselines = {g: ATARI_BASELINES[g]["human"] for g in ATARI_GAMES}
    random_baselines = {g: ATARI_BASELINES[g]["random"] for g in ATARI_GAMES}

    logger.info("Computing random baseline returns...")
    random_returns: list[float] = []
    for task_id, game in enumerate(ATARI_GAMES):
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
    task_similarity: dict = {}

    for task_id in range(n_tasks):
        logger.info("--- Task %d: %s ---", task_id, ATARI_GAMES[task_id])

        strategy.train_task(task_trajs_all[task_id], steps=train_steps, batch_size=32)
        strategy.after_task(task_trajs_all[task_id])

        for eval_task_id in range(n_tasks):
            game = ATARI_GAMES[eval_task_id]
            strategy.set_eval_task(eval_task_id)

            eval_rollouts = 3
            total_returns: list[float] = []
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

    metrics = compute_final_metrics(performance_matrix)
    cl = StandardCLMetrics(performance_matrix)
    metrics.update(cl.summary())

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
                "human_baselines_source": (
                    "Mnih et al. (2015). Human-level control through deep "
                    "reinforcement learning. Nature 518, 529-533; ALE "
                    "NoFrameskip-v4 baselines."
                ),
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
    parser = argparse.ArgumentParser(
        description=(
            "Train TSN-Affinity on random-policy Atari trajectories and "
            "evaluate with greedy rollouts. Requires gymnasium."
        )
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="tsn_affinity",
        choices=["tsn_affinity", "tsn_core"],
        help="Strategy name.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/atari_tsn_affinity",
        help="Output directory.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda/cpu).")
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=10,
        help="Number of trajectories per game.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=2000, help="Max steps per trajectory."
    )
    parser.add_argument(
        "--train-steps", type=int, default=2000, help="Training steps per task."
    )
    parser.add_argument(
        "--eval-rollouts",
        type=int,
        default=3,
        help="Number of greedy rollouts per evaluation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")

    args = parser.parse_args()

    setup_logging()

    run_atari_benchmark(
        strategy_name=args.strategy,
        output_dir=args.output,
        device=args.device,
        n_trajectories=args.n_trajectories,
        max_steps=args.max_steps,
        train_steps=args.train_steps,
        n_eval_runs=args.eval_rollouts,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
