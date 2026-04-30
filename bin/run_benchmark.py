#!/usr/bin/env python3
"""Benchmark runner for TSN-Affinity strategies.

Reproducible benchmarking with:
- Synthetic trajectory generation with controlled properties
- Multiple strategy comparison (tsn_affinity, tsn_core, tsn_replay_kl)
- Timing measurements per task
- Memory usage tracking
- Performance metrics (ACC, BWT, Forgetting, FWT)
- Statistical validation across multiple runs

Usage:
    python bin/run_benchmark.py --output runs/benchmark_results --n-runs 3
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.strategies.tsn_affinity import AffinityRoutingConfig
from tsn_affinity.data.batch_generator import masked_cross_entropy
from tsn_affinity.data.trajectory import Trajectory, discount_cumsum
from tsn_affinity.run.analysis import compute_final_metrics
from tsn_affinity.routing.affinity_metrics import _AffinityBatchLoader
from tsn_affinity.strategies.tsn_affinity import TSNAffinityStrategy
from tsn_affinity.strategies.tsn_core import TSNCoreStrategy
from tsn_affinity.strategies.tsn_base import TSNBaseStrategy
from tsn_affinity.benchmarks.metrics import StandardCLMetrics


@dataclass
class BenchmarkConfig:
    n_tasks: int = 5
    trajs_per_task: int = 10
    traj_len: int = 100
    obs_shape: tuple = (4, 84, 84)
    n_actions: int = 18
    seq_len: int = 20
    train_steps: int = 200
    eval_steps: int = 50
    batch_size: int = 32
    device: str = "cpu"
    seed: int = 42


@dataclass
class BenchmarkResult:
    strategy: str
    run_id: int
    metrics: Dict[str, float]
    task_times: List[float]
    total_time: float
    memory_peak_mb: float
    performance_matrix: np.ndarray
    task_similarity: Dict[int, Dict]


@dataclass
class BenchmarkSummary:
    strategy: str
    n_runs: int
    acc_mean: float
    acc_std: float
    bwt_mean: float
    bwt_std: float
    forgetting_mean: float
    forgetting_std: float
    fwt_mean: float
    fwt_std: float
    time_per_task_mean: float
    time_per_task_std: float
    memory_peak_mean: float
    memory_peak_std: float


def generate_trajectories(
    n_trajs: int,
    traj_len: int,
    obs_shape: tuple,
    n_actions: int,
    seed: int,
    action_change_prob: float = 0.3,
) -> List[Trajectory]:
    """Generate synthetic trajectories with controlled action patterns.

    Args:
        n_trajs: Number of trajectories to generate.
        traj_len: Length of each trajectory.
        obs_shape: Observation shape.
        n_actions: Number of actions.
        seed: Random seed for reproducibility.
        action_change_prob: Probability of action change per step.

    Returns:
        List of Trajectory objects.
    """
    rng = np.random.default_rng(seed)
    trajs = []

    for t_idx in range(n_trajs):
        T = traj_len

        if len(obs_shape) == 3:
            obs = rng.normal(0.5, 0.25, (T, *obs_shape)).astype(np.float32)
            obs = np.clip(obs, 0, 1)
        else:
            obs = rng.normal(0.5, 0.25, (T, obs_shape[0])).astype(np.float32)

        base_action = rng.integers(0, n_actions)
        actions = np.full(T, base_action, dtype=np.int64)
        for i in range(1, T):
            if rng.random() < action_change_prob:
                actions[i] = (base_action + rng.integers(1, n_actions)) % n_actions
            else:
                actions[i] = actions[i - 1]

        rewards = rng.standard_normal(T).astype(np.float32) * 0.1
        timesteps = np.arange(T, dtype=np.float32)
        returns_to_go = discount_cumsum(rewards, gamma=1.0)

        trajs.append(Trajectory(obs, actions, rewards, timesteps, returns_to_go))

    return trajs


def get_peak_memory_mb() -> float:
    """Get peak memory usage in MB (macOS/Linux compatible)."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def evaluate_strategy(
    strategy,
    task_trajs_all: List[List[Trajectory]],
    config: BenchmarkConfig,
    eval_steps: int,
) -> tuple:
    """Evaluate a strategy on all tasks and return performance matrix.

    Returns:
        Tuple of (performance_matrix, task_similarity, task_times, total_time)
    """
    n_tasks = len(task_trajs_all)
    performance_matrix = np.zeros((n_tasks, n_tasks))
    task_similarity = {}
    task_times = []

    for task_id in range(n_tasks):
        mem_before = get_peak_memory_mb()

        start_time = time.perf_counter()
        strategy.train_task(task_trajs_all[task_id], steps=config.train_steps, batch_size=config.batch_size)
        strategy.after_task(task_trajs_all[task_id])
        task_time = time.perf_counter() - start_time
        task_times.append(task_time)

        mem_after = get_peak_memory_mb()

        for eval_task_id in range(n_tasks):
            strategy.set_eval_task(eval_task_id)
            perf = _evaluate_on_trajectories(strategy, task_trajs_all[eval_task_id], eval_steps)
            performance_matrix[task_id, eval_task_id] = perf

        if hasattr(strategy, "task_similarity"):
            task_similarity[task_id] = strategy.task_similarity.get(task_id, {})

        strategy.clear_eval_task()

        print(f"    Task {task_id}: time={task_time:.2f}s mem_delta={mem_after - mem_before:.1f}MB "
              f"diagonal={performance_matrix[task_id, task_id]:.4f}")

    total_time = sum(task_times)
    return performance_matrix, task_similarity, task_times, total_time


def _evaluate_on_trajectories(strategy, trajectories: List[Trajectory], n_steps: int = 50) -> float:
    """Evaluate strategy on trajectories and return average loss."""
    if not trajectories:
        return 0.0

    strategy.model.eval()
    losses = []

    loader = _EvalBatchLoader(trajectories, strategy.seq_len, 32, "cpu")
    for i, batch in enumerate(loader):
        if i >= n_steps // 32:
            break
        obs, actions, rtg, ts = batch
        mask = (actions != -1).float()
        with torch.no_grad():
            logits = strategy.model(obs, actions, rtg, ts, attention_mask=mask)
        loss = masked_cross_entropy(logits, actions, mask)
        losses.append(float(loss.detach().item()))

    return float(np.mean(losses)) if losses else 0.0


class _EvalBatchLoader:
    """Efficient evaluation batch loader."""

    def __init__(self, trajs: List[Trajectory], seq_len: int, batch_size: int, device: str):
        self.device = device
        self._prebuild_batches(trajs, seq_len, batch_size, device)

    def _prebuild_batches(self, trajs, seq_len, batch_size, device):
        batch_list = []
        for _ in range(batch_size):
            traj = trajs[np.random.randint(len(trajs))]
            T = len(traj.actions)
            start = 0 if T <= seq_len else np.random.randint(0, T - seq_len + 1)
            end = min(start + seq_len, T)

            o = traj.obs[start:end]
            a = traj.actions[start:end]
            r = traj.returns_to_go[start:end]
            t = traj.timesteps[start:end]

            pad = seq_len - len(a)
            if o.ndim == 2:
                o = np.pad(o, ((0, pad), (0, 0)))
            else:
                o = np.pad(o, ((0, pad), (0, 0), (0, 0), (0, 0)))
            a = np.pad(a, (0, pad), constant_values=-1)
            r = np.pad(r, (0, pad))
            t = np.pad(t, (0, pad))

            batch_list.append((o, a, r[:, None], t))

        self.obs = torch.tensor(np.stack([b[0] for b in batch_list]), dtype=torch.float32, device=device)
        self.actions = torch.tensor(np.stack([b[1] for b in batch_list]), dtype=torch.long, device=device)
        self.rtg = torch.tensor(np.stack([b[2] for b in batch_list]), dtype=torch.float32, device=device)
        self.ts = torch.tensor(np.stack([b[3] for b in batch_list]), dtype=torch.long, device=device)
        self.n_batches = 1

    def __iter__(self):
        return self

    def __next__(self):
        return self.obs, self.actions, self.rtg, self.ts


def make_strategy(name: str, obs_shape: tuple, n_actions: int, device: str):
    """Create a strategy by name."""
    if name == "tsn_core":
        return TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=20,
            device=device,
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )
    elif name == "tsn_affinity":
        affinity_config = AffinityRoutingConfig(
            mode="action",
            action_threshold=12.0,
            routing_n_batches=4,
            routing_batch_size=64,
        )
        return TSNAffinityStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=20,
            device=device,
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
            affinity_config=affinity_config,
        )
    elif name == "tsn_replay_kl":
        from tsn_affinity.strategies.tsn_replay_kl import TSNReplayKLStrategy
        return TSNReplayKLStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=20,
            device=device,
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.5),
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")


def run_single_benchmark(
    strategy_name: str,
    config: BenchmarkConfig,
    run_id: int,
    seed_offset: int = 0,
) -> BenchmarkResult:
    """Run a single benchmark iteration.

    Args:
        strategy_name: Name of the strategy to benchmark.
        config: Benchmark configuration.
        run_id: Run identifier for this iteration.
        seed_offset: Additional seed offset for variation.

    Returns:
        BenchmarkResult with all metrics and timing data.
    """
    np.random.seed(config.seed + seed_offset)
    torch.manual_seed(config.seed + seed_offset)

    print(f"\n  [{strategy_name}] Generating data (seed={config.seed + seed_offset})...")
    task_trajs_all = []
    for task_id in range(config.n_tasks):
        trajs = generate_trajectories(
            n_trajs=config.trajs_per_task,
            traj_len=config.traj_len,
            obs_shape=config.obs_shape,
            n_actions=config.n_actions,
            seed=config.seed + seed_offset + task_id * 1000,
        )
        task_trajs_all.append(trajs)

    strategy = make_strategy(strategy_name, config.obs_shape, config.n_actions, config.device)

    print(f"  [{strategy_name}] Training and evaluation...")
    perf_matrix, task_sim, task_times, total_time = evaluate_strategy(
        strategy, task_trajs_all, config, config.eval_steps
    )

    metrics = compute_final_metrics(perf_matrix)
    memory_peak = get_peak_memory_mb()

    return BenchmarkResult(
        strategy=strategy_name,
        run_id=run_id,
        metrics=metrics,
        task_times=task_times,
        total_time=total_time,
        memory_peak_mb=memory_peak,
        performance_matrix=perf_matrix,
        task_similarity=task_sim,
    )


def run_benchmark_suite(
    strategies: List[str],
    config: BenchmarkConfig,
    n_runs: int = 3,
    output_dir: str = "runs/benchmark",
) -> Dict[str, BenchmarkSummary]:
    """Run full benchmark suite across multiple strategies and runs.

    Args:
        strategies: List of strategy names to benchmark.
        config: Benchmark configuration.
        n_runs: Number of runs per strategy for statistical validation.
        output_dir: Directory to save results.

    Returns:
        Dict of strategy_name -> BenchmarkSummary.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results: Dict[str, List[BenchmarkResult]] = {s: [] for s in strategies}

    for run_id in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Benchmark Run {run_id + 1}/{n_runs}")
        print(f"{'='*60}")

        for strategy_name in strategies:
            print(f"\nStrategy: {strategy_name}")
            result = run_single_benchmark(strategy_name, config, run_id, seed_offset=run_id * 100)
            all_results[strategy_name].append(result)

            summary = compute_summary(all_results[strategy_name])
            print(f"  Cumulative ({run_id + 1} runs): ACC={summary.acc_mean:.4f}±{summary.acc_std:.4f} "
                  f"BWT={summary.bwt_mean:.4f}±{summary.bwt_std:.4f}")

    summaries = {}
    for strategy_name in strategies:
        summaries[strategy_name] = compute_summary(all_results[strategy_name])

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for strategy_name, summary in summaries.items():
        print(f"\n{strategy_name}:")
        print(f"  ACC:    {summary.acc_mean:.4f} ± {summary.acc_std:.4f}")
        print(f"  BWT:    {summary.bwt_mean:.4f} ± {summary.bwt_std:.4f}")
        print(f"  Forgetting: {summary.forgetting_mean:.4f} ± {summary.forgetting_std:.4f}")
        print(f"  FWT:    {summary.fwt_mean:.4f} ± {summary.fwt_std:.4f}")
        print(f"  Time/task: {summary.time_per_task_mean:.2f}s ± {summary.time_per_task_std:.2f}s")
        print(f"  Memory:  {summary.memory_peak_mean:.1f}MB ± {summary.memory_peak_std:.1f}MB")

    _save_results(summaries, all_results, output_dir)

    return summaries


def compute_summary(results: List[BenchmarkResult]) -> BenchmarkSummary:
    """Compute summary statistics from benchmark results."""
    acc_values = [r.metrics["acc"] for r in results]
    bwt_values = [r.metrics["bwt"] for r in results]
    forgetting_values = [r.metrics["forgetting"] for r in results]
    fwt_values = [r.metrics["fwt"] for r in results]
    times = [sum(r.task_times) / len(r.task_times) for r in results]
    mems = [r.memory_peak_mb for r in results]

    return BenchmarkSummary(
        strategy=results[0].strategy,
        n_runs=len(results),
        acc_mean=float(np.mean(acc_values)),
        acc_std=float(np.std(acc_values)),
        bwt_mean=float(np.mean(bwt_values)),
        bwt_std=float(np.std(bwt_values)),
        forgetting_mean=float(np.mean(forgetting_values)),
        forgetting_std=float(np.std(forgetting_values)),
        fwt_mean=float(np.mean(fwt_values)),
        fwt_std=float(np.std(fwt_values)),
        time_per_task_mean=float(np.mean(times)),
        time_per_task_std=float(np.std(times)),
        memory_peak_mean=float(np.mean(mems)),
        memory_peak_std=float(np.std(mems)),
    )


def _save_results(
    summaries: Dict[str, BenchmarkSummary],
    all_results: Dict[str, List[BenchmarkResult]],
    output_dir: str,
) -> None:
    """Save benchmark results to output directory."""
    summary_data = {}
    for name, s in summaries.items():
        summary_data[name] = {
            "n_runs": s.n_runs,
            "acc": f"{s.acc_mean:.4f} ± {s.acc_std:.4f}",
            "bwt": f"{s.bwt_mean:.4f} ± {s.bwt_std:.4f}",
            "forgetting": f"{s.forgetting_mean:.4f} ± {s.forgetting_std:.4f}",
            "fwt": f"{s.fwt_mean:.4f} ± {s.fwt_std:.4f}",
            "time_per_task_s": f"{s.time_per_task_mean:.2f} ± {s.time_per_task_std:.2f}",
            "memory_peak_mb": f"{s.memory_peak_mean:.1f} ± {s.memory_peak_std:.1f}",
        }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    for name, results in all_results.items():
        for r in results:
            run_dir = os.path.join(output_dir, name, f"run_{r.run_id}")
            os.makedirs(run_dir, exist_ok=True)
            np.save(os.path.join(run_dir, "performance_matrix.npy"), r.performance_matrix)
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump({k: float(v) for k, v in r.metrics.items()}, f, indent=2)
            with open(os.path.join(run_dir, "task_similarity.json"), "w") as f:
                json.dump({str(k): v for k, v in r.task_similarity.items()}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run TSN-Affinity benchmark suite")
    parser.add_argument("--strategies", nargs="+", default=["tsn_core", "tsn_affinity"],
                        help="Strategies to benchmark")
    parser.add_argument("--n-tasks", type=int, default=5, help="Number of tasks")
    parser.add_argument("--trajs-per-task", type=int, default=10, help="Trajectories per task")
    parser.add_argument("--train-steps", type=int, default=200, help="Training steps per task")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--output", type=str, default="runs/benchmark", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = BenchmarkConfig(
        n_tasks=args.n_tasks,
        trajs_per_task=args.trajs_per_task,
        traj_len=100,
        obs_shape=(4, 84, 84),
        n_actions=18,
        seq_len=20,
        train_steps=args.train_steps,
        eval_steps=50,
        batch_size=32,
        device=args.device,
        seed=args.seed,
    )

    print(f"Starting benchmark suite:")
    print(f"  Strategies: {args.strategies}")
    print(f"  Tasks: {config.n_tasks}")
    print(f"  Trajectories/task: {config.trajs_per_task}")
    print(f"  Training steps/task: {config.train_steps}")
    print(f"  Benchmark runs: {args.n_runs}")

    summaries = run_benchmark_suite(
        strategies=args.strategies,
        config=config,
        n_runs=args.n_runs,
        output_dir=args.output,
    )

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()