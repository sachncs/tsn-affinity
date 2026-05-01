#!/usr/bin/env python3
"""Entry point for running Panda continuous control benchmark.

Usage:
    python -m scripts.run_panda --data data/panda_tasks.pkl --strategy tsn_affinity
"""

import argparse
import json
import logging
import os

import numpy as np
import torch

from tsn_affinity.benchmarks.metrics import StandardCLMetrics
from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.core.logging_config import setup_logging
from tsn_affinity.data.panda import load_panda_offline_pkl, make_minibatches_panda
from tsn_affinity.data.trajectory import Trajectory
from tsn_affinity.strategies.base import BaseStrategy
from tsn_affinity.strategies.tsn_affinity import (
    AffinityRoutingConfig,
    TSNAffinityStrategy,
)
from tsn_affinity.strategies.tsn_core import TSNCoreStrategy

logger = logging.getLogger("tsn_affinity")


def run_panda_benchmark(
    data_path: str,
    strategy_name: str,
    output_dir: str,
    device: str = "cuda",
    train_steps: int = 2000,
    batch_size: int = 64,
    seq_len: int = 20,
):
    """Run Panda benchmark with continuous control data.

    Args:
        data_path: Path to Panda offline trajectory data (pickle file).
        strategy_name: Strategy to evaluate.
        output_dir: Directory to save results.
        device: Device for training.
        train_steps: Training steps per task.
        batch_size: Batch size.
        seq_len: Sequence length for DT.
    """
    logger.info("Loading Panda data from %s", data_path)
    all_trajs = load_panda_offline_pkl(data_path)
    if not all_trajs:
        raise ValueError(f"No trajectories loaded from {data_path}")

    # Infer obs shape and action dimension from data
    obs_shape = tuple(all_trajs[0].obs[0].shape)
    n_actions = all_trajs[0].actions.shape[-1] if all_trajs[0].actions.ndim > 1 else 1

    logger.info("Loaded %d trajectories", len(all_trajs))
    logger.info("  obs_shape: %s, n_actions: %d", obs_shape, n_actions)

    # Simple task splitting: split trajectories into 3 tasks by index
    n_tasks = 3
    trajs_per_task = len(all_trajs) // n_tasks
    task_trajs_all: list[list[Trajectory]] = []
    for i in range(n_tasks):
        start = i * trajs_per_task
        end = start + trajs_per_task if i < n_tasks - 1 else len(all_trajs)
        task_trajs_all.append(all_trajs[start:end])

    if strategy_name == "tsn_affinity":
        affinity_config = AffinityRoutingConfig(mode="hybrid", hybrid_alpha=0.5)
        strategy: BaseStrategy = TSNAffinityStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=seq_len,
            device=device,
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.3),
            affinity_config=affinity_config,
        )
    elif strategy_name == "tsn_core":
        strategy = TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=seq_len,
            device=device,
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.3),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    logger.info("Training %s on %d tasks...", strategy_name, n_tasks)
    performance_matrix = np.zeros((n_tasks, n_tasks))

    for task_id in range(n_tasks):
        logger.info("--- Task %d ---", task_id)
        strategy.train_task(
            task_trajs_all[task_id], steps=train_steps, batch_size=batch_size
        )
        strategy.after_task(task_trajs_all[task_id])

        for eval_task_id in range(n_tasks):
            strategy.set_eval_task(eval_task_id)
            loader = make_minibatches_panda(
                task_trajs_all[eval_task_id], seq_len, batch_size, device
            )
            total_loss = 0.0
            count = 0
            for i, (obs, actions, rtg, ts, mask) in enumerate(loader):
                if i >= 10:
                    break
                with torch.no_grad():
                    logits = strategy.model(obs, actions, rtg, ts, attention_mask=mask)
                    loss = torch.nn.functional.mse_loss(logits, actions)
                total_loss += float(loss.detach().item())
                count += 1

            avg_loss = total_loss / max(count, 1)
            performance_matrix[task_id, eval_task_id] = 1.0 / (1.0 + avg_loss)

        strategy.clear_eval_task()
        logger.info(
            "  Task %d diagonal: %.4f", task_id, performance_matrix[task_id, task_id]
        )

    metrics = StandardCLMetrics(performance_matrix).compute_all()
    logger.info("=" * 60)
    logger.info("Panda Benchmark Results")
    logger.info("=" * 60)
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name.upper(), value)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "performance_matrix.npy"), performance_matrix)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(
            {
                "strategy": strategy_name,
                "metrics": {k: float(v) for k, v in metrics.items()},
                "performance_matrix": performance_matrix.tolist(),
            },
            f,
            indent=2,
        )

    logger.info("Results saved to: %s", output_dir)


def main() -> None:
    """Run the Panda TSN-Affinity benchmark from the command line."""
    parser = argparse.ArgumentParser(description="Run Panda TSN-Affinity benchmark")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to Panda offline data"
    )
    parser.add_argument(
        "--strategy", type=str, default="tsn_affinity", help="Strategy name"
    )
    parser.add_argument(
        "--output", type=str, default="runs/panda_tsn_affinity", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--train-steps", type=int, default=2000, help="Training steps per task"
    )

    args = parser.parse_args()

    setup_logging()

    run_panda_benchmark(
        args.data,
        args.strategy,
        args.output,
        args.device,
        args.train_steps,
    )


if __name__ == "__main__":
    main()
