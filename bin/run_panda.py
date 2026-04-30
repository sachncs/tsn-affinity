#!/usr/bin/env python3
"""Entry point for running Panda continuous control benchmark."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tsn_affinity.benchmarks.metrics import compute_acc, compute_bwt, compute_forgetting, compute_fwt
from tsn_affinity.core.config import AffinityRoutingConfig, ModelConfig, SparseConfig
from tsn_affinity.run.analysis import compute_final_metrics
from tsn_affinity.strategies.tsn_affinity import TSNAffinityStrategy
from tsn_affinity.strategies.tsn_core import TSNCoreStrategy


def run_panda_benchmark(
    data_path: str,
    strategy_name: str,
    output_dir: str,
    device: str = "cuda",
    train_steps: int = 2000,
):
    obs_shape = (44,)
    n_actions = 9

    if strategy_name == "tsn_affinity":
        affinity_config = AffinityRoutingConfig(mode="hybrid", hybrid_alpha=0.5)
        strategy = TSNAffinityStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=20,
            device=device,
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.3),
            affinity_config=affinity_config,
        )
    elif strategy_name == "tsn_core":
        strategy = TSNCoreStrategy(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=20,
            device=device,
            model_config=ModelConfig(obs_shape=obs_shape, n_actions=n_actions),
            sparse_config=SparseConfig(keep_ratio=0.3),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    print(f"Loaded {strategy_name} strategy for Panda benchmark")
    print(f"  obs_shape: {obs_shape}, n_actions: {n_actions}")
    print(f"  train_steps: {train_steps}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nBenchmark results would be saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Panda TSN-Affinity benchmark")
    parser.add_argument("--data", type=str, required=True, help="Path to Panda offline data")
    parser.add_argument("--strategy", type=str, default="tsn_affinity", help="Strategy name")
    parser.add_argument("--output", type=str, default="runs/panda_tsn_affinity", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--train-steps", type=int, default=2000, help="Training steps per task")

    args = parser.parse_args()

    run_panda_benchmark(
        args.data,
        args.strategy,
        args.output,
        args.device,
        args.train_steps,
    )


if __name__ == "__main__":
    main()