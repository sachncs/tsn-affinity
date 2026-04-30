#!/usr/bin/env python3
"""Reproducible benchmark script for TSN-Affinity continual learning.

This script runs controlled benchmarks comparing tsn_core vs tsn_affinity
on synthetic tasks with ground-truth metrics.

Usage:
    python bin/benchmark.py --output runs/benchmark_results
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tsn_affinity.benchmarks.metrics import StandardCLMetrics
from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.data.trajectory import Trajectory
from tsn_affinity.strategies.tsn_affinity import AffinityRoutingConfig, TSNAffinityStrategy
from tsn_affinity.strategies.tsn_core import TSNCoreStrategy


def make_synthetic_trajectories(
    n_trajs: int,
    traj_len: int,
    obs_dim: int,
    n_actions: int,
    task_id: int,
    seed: int,
) -> List[Trajectory]:
    """Create synthetic trajectories with task-specific characteristics.

    Each task has distinct action distribution and reward patterns to simulate
    different task domains.
    """
    rng = np.random.default_rng(seed)

    # Task-specific parameters for diversity
    base_action = task_id % n_actions
    action_probs = np.zeros(n_actions)
    action_probs[base_action] = 0.6
    action_probs[(base_action + 1) % n_actions] = 0.3
    remaining = 1.0 - action_probs.sum()
    action_probs[:] += remaining / n_actions

    trajs = []
    for t in range(n_trajs):
        # Generate observations
        obs = rng.normal(0.5, 0.25, (traj_len, obs_dim)).astype(np.float32)
        obs = np.clip(obs, 0, 1)

        # Sample actions from task-specific distribution
        actions = rng.choice(n_actions, size=traj_len, p=action_probs).astype(np.int64)

        # Generate rewards with task-specific patterns
        rewards = np.zeros(traj_len, dtype=np.float32)
        for i in range(traj_len):
            if actions[i] == base_action:
                rewards[i] = rng.normal(1.0, 0.1)
            else:
                rewards[i] = rng.normal(-0.1, 0.2)

        timesteps = np.arange(traj_len, dtype=np.float32)
        returns_to_go = np.cumsum(rewards[::-1])[::-1].astype(np.float32)

        trajs.append(Trajectory(obs, actions, rewards, timesteps, returns_to_go))

    return trajs


def run_single_benchmark(
    strategy_name: str,
    n_tasks: int,
    trajs_per_task: int,
    traj_len: int,
    obs_dim: int,
    n_actions: int,
    train_steps: int,
    device: str,
    seed: int,
) -> Dict:
    """Run a single benchmark with specified parameters."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create strategy
    if strategy_name == "tsn_affinity":
        affinity_config = AffinityRoutingConfig(
            mode="action",
            action_threshold=12.0,
            routing_n_batches=4,
            routing_batch_size=64,
            relative_threshold=True,
            copy_creation_margin=2.5,
        )
        strategy = TSNAffinityStrategy(
            obs_shape=(obs_dim,),
            n_actions=n_actions,
            seq_len=20,
            device=device,
            model_config=ModelConfig(obs_shape=(obs_dim,), n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
            affinity_config=affinity_config,
        )
    elif strategy_name == "tsn_core":
        strategy = TSNCoreStrategy(
            obs_shape=(obs_dim,),
            n_actions=n_actions,
            seq_len=20,
            device=device,
            model_config=ModelConfig(obs_shape=(obs_dim,), n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Collect trajectories and train
    task_trajs = []
    for task_id in range(n_tasks):
        trajs = make_synthetic_trajectories(
            n_trajs=trajs_per_task,
            traj_len=traj_len,
            obs_dim=obs_dim,
            n_actions=n_actions,
            task_id=task_id,
            seed=seed + task_id * 1000,
        )
        task_trajs.append(trajs)

    # Training loop
    start_time = time.time()
    training_times = []

    for task_id in range(n_tasks):
        task_start = time.time()
        strategy.train_task(task_trajs[task_id], steps=train_steps, batch_size=32)
        strategy.after_task(task_trajs[task_id])
        training_times.append(time.time() - task_start)

    total_time = time.time() - start_time

    # Evaluate: compute loss matrix M[i,j] = performance on task j after training on task i
    from tsn_affinity.data.batch_generator import make_minibatches, masked_cross_entropy

    performance_matrix = np.zeros((n_tasks, n_tasks))

    for train_task in range(n_tasks):
        for eval_task in range(n_tasks):
            strategy.set_eval_task(eval_task)
            loader = make_minibatches(
                task_trajs[eval_task],
                strategy.seq_len,
                32,
                strategy.device,
            )
            total_loss = 0.0
            count = 0
            for i, (obs, actions, rtg, ts, mask) in enumerate(loader):
                if i >= 5:
                    break
                with torch.no_grad():
                    logits = strategy.model(obs, actions, rtg, ts, attention_mask=mask)
                    loss = masked_cross_entropy(logits, actions, mask)
                total_loss += float(loss.detach().item())
                count += 1

            avg_loss = total_loss / max(count, 1)
            # Convert loss to performance (lower loss = higher performance)
            # Normalize to [0, 1] where 1 is best (lowest loss)
            performance_matrix[train_task, eval_task] = 1.0 / (1.0 + avg_loss)

        strategy.clear_eval_task()

    # Compute metrics
    metrics = StandardCLMetrics(performance_matrix).compute_all()

    return {
        "strategy": strategy_name,
        "performance_matrix": performance_matrix.tolist(),
        "metrics": metrics,
        "training_times": training_times,
        "total_time": total_time,
    }


def run_benchmark_suite(
    strategies: List[str],
    n_tasks: int,
    trajs_per_task: int,
    traj_len: int,
    obs_dim: int,
    n_actions: int,
    train_steps: int,
    n_runs: int,
    output_dir: str,
    device: str,
) -> Dict:
    """Run benchmark suite with multiple strategies and runs."""
    results = {}

    for strategy_name in strategies:
        print(f"\n{'='*60}")
        print(f"Benchmarking {strategy_name}")
        print(f"{'='*60}")

        run_results = []
        for run_id in range(n_runs):
            print(f"\nRun {run_id + 1}/{n_runs}")
            seed = 42 + run_id * 100

            result = run_single_benchmark(
                strategy_name=strategy_name,
                n_tasks=n_tasks,
                trajs_per_task=trajs_per_task,
                traj_len=traj_len,
                obs_dim=obs_dim,
                n_actions=n_actions,
                train_steps=train_steps,
                device=device,
                seed=seed,
            )

            run_results.append(result)

            print(f"  ACC: {result['metrics']['acc']:.4f}")
            print(f"  BWT: {result['metrics']['bwt']:.4f}")
            print(f"  Forgetting: {result['metrics']['forgetting']:.4f}")
            print(f"  Training time: {result['total_time']:.2f}s")

        # Aggregate results
        accs = [r["metrics"]["acc"] for r in run_results]
        bwts = [r["metrics"]["bwt"] for r in run_results]
        forgettings = [r["metrics"]["forgetting"] for r in run_results]
        fwt = [r["metrics"]["fwt"] for r in run_results]
        times = [r["total_time"] for r in run_results]

        results[strategy_name] = {
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "bwt_mean": float(np.mean(bwts)),
            "bwt_std": float(np.std(bwts)),
            "forgetting_mean": float(np.mean(forgettings)),
            "forgetting_std": float(np.std(forgettings)),
            "fwt_mean": float(np.mean(fwt)),
            "fwt_std": float(np.std(fwt)),
            "time_mean": float(np.mean(times)),
            "time_std": float(np.std(times)),
            "individual_runs": run_results,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run TSN-Affinity benchmarks")
    parser.add_argument("--strategies", nargs="+", default=["tsn_core", "tsn_affinity"],
                        choices=["tsn_core", "tsn_affinity"])
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--trajs-per-task", type=int, default=5)
    parser.add_argument("--traj-len", type=int, default=50)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--n-actions", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--output", type=str, default="runs/benchmark")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    print(f"Running benchmark suite:")
    print(f"  Strategies: {args.strategies}")
    print(f"  Tasks: {args.n_tasks}")
    print(f"  Trajectories per task: {args.trajs_per_task}")
    print(f"  Trajectory length: {args.traj_len}")
    print(f"  Observation dim: {args.obs_dim}")
    print(f"  Action space: {args.n_actions}")
    print(f"  Training steps per task: {args.train_steps}")
    print(f"  Number of runs: {args.n_runs}")

    results = run_benchmark_suite(
        strategies=args.strategies,
        n_tasks=args.n_tasks,
        trajs_per_task=args.trajs_per_task,
        traj_len=args.traj_len,
        obs_dim=args.obs_dim,
        n_actions=args.n_actions,
        train_steps=args.train_steps,
        n_runs=args.n_runs,
        output_dir=args.output,
        device=args.device,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Strategy':<15} {'ACC':<12} {'BWT':<12} {'Forgetting':<12} {'Time':<12}")
    print("-" * 60)

    for name, res in results.items():
        print(
            f"{name:<15} "
            f"{res['acc_mean']:.4f}±{res['acc_std']:.4f} "
            f"{res['bwt_mean']:.4f}±{res['bwt_std']:.4f} "
            f"{res['forgetting_mean']:.4f}±{res['forgetting_std']:.4f} "
            f"{res['time_mean']:.2f}±{res['time_std']:.2f}s"
        )

    # Save results
    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "benchmark_results.json"), "w") as f:
        json.dump({
            "config": vars(args),
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "individual_runs"}
                        for k, v in results.items()},
        }, f, indent=2)

    # Save individual run details
    for strategy_name, res in results.items():
        run_file = os.path.join(args.output, f"{strategy_name}_runs.json")
        with open(run_file, "w") as f:
            json.dump(res["individual_runs"], f, indent=2)

    print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    import torch
    main()
