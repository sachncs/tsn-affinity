"""Result analysis utilities for benchmark runs."""

import numpy as np

from tsn_affinity.benchmarks.metrics import (
    StandardCLMetrics,
    compute_acc,
    compute_bwt,
    compute_forgetting,
    compute_fwt,
)


def analyze_run(run_dir: str) -> dict[str, float]:
    """Analyze a saved benchmark run directory.

    Args:
        run_dir: Path to directory containing saved results.

    Returns:
        Dict with metrics (acc, bwt, forgetting, fwt).
    """
    import json
    import os

    perf_path = os.path.join(run_dir, "performance_matrix.npy")
    if os.path.exists(perf_path):
        matrix = np.load(perf_path)
    else:
        results_path = os.path.join(run_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                data = json.load(f)
            matrix = np.array(data["performance_matrix"])
        else:
            raise FileNotFoundError(f"No results found in {run_dir}")

    metrics = StandardCLMetrics(matrix)
    return metrics.compute_all()


def compute_final_metrics(
    performance_matrix: np.ndarray,
) -> dict[str, float]:
    """Compute final metrics from a performance matrix.

    Args:
        performance_matrix: [n_tasks, n_tasks] performance matrix.

    Returns:
        Dict with computed metrics.
    """
    return {
        "acc": compute_acc(performance_matrix),
        "bwt": compute_bwt(performance_matrix),
        "forgetting": compute_forgetting(performance_matrix),
        "fwt": compute_fwt(performance_matrix),
    }


def compare_runs(run_dir_a: str, run_dir_b: str) -> dict[str, dict[str, float]]:
    """Compare two benchmark runs.

    Args:
        run_dir_a: First run directory.
        run_dir_b: Second run directory.

    Returns:
        Dict with metrics for each run and deltas.
    """
    metrics_a = analyze_run(run_dir_a)
    metrics_b = analyze_run(run_dir_b)

    result = {
        "run_a": metrics_a,
        "run_b": metrics_b,
        "delta": {f"{k}_diff": metrics_b[k] - metrics_a[k] for k in metrics_a.keys()},
    }

    return result
