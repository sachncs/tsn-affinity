"""Standard continual learning metric computations."""

from typing import Dict, List

import numpy as np


def compute_acc(performance_matrix: np.ndarray) -> float:
    """Compute average accuracy (ACC) metric.

    ACC = mean of diagonal entries in performance matrix.

    Args:
        performance_matrix: [n_tasks, n_tasks] matrix where M[i,j] is
            performance on task j after training on task i.

    Returns:
        Mean diagonal performance.
    """
    n = performance_matrix.shape[0]
    return float(np.mean([performance_matrix[i, i] for i in range(n)]))


def compute_bwt(performance_matrix: np.ndarray) -> float:
    """Compute backward transfer (BWT) metric.

    BWT = mean_{i<j} (M[j,i] - M[i,i])

    Measures how much learning task j hurts performance on earlier task i.

    Args:
        performance_matrix: [n_tasks, n_tasks] performance matrix.

    Returns:
        BWT score (higher is better).
    """
    n = performance_matrix.shape[0]
    bwt_sum = 0.0
    count = 0
    for j in range(n):
        for i in range(j):
            bwt_sum += performance_matrix[j, i] - performance_matrix[i, i]
            count += 1
    return float(bwt_sum / count) if count > 0 else 0.0


def compute_forgetting(performance_matrix: np.ndarray) -> float:
    """Compute forgetting metric.

    F = mean_{i<j} max_{k<j} (M[k,i] - M[j,i])

    Measures how much performance on task i drops after learning later tasks.

    Args:
        performance_matrix: [n_tasks, n_tasks] performance matrix.

    Returns:
        Forgetting score (lower is better, 0 = no forgetting).
    """
    n = performance_matrix.shape[0]
    f_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            max_before = max(performance_matrix[k, i] for k in range(j))
            f_sum += max_before - performance_matrix[j, i]
            count += 1
    return float(f_sum / count) if count > 0 else 0.0


def compute_fwt(performance_matrix: np.ndarray) -> float:
    """Compute forward transfer (FWT) metric.

    FWT = mean_{i<j} (M[j,i] - M[i,i])

    Measures how much learning task j helps performance on later task i.

    Args:
        performance_matrix: [n_tasks, n_tasks] performance matrix.

    Returns:
        FWT score (higher is better).
    """
    n = performance_matrix.shape[0]
    fwt_sum = 0.0
    count = 0
    for j in range(n):
        for i in range(j):
            fwt_sum += performance_matrix[j, i] - performance_matrix[i, i]
            count += 1
    return float(fwt_sum / count) if count > 0 else 0.0


class StandardCLMetrics:
    """Container for standard continual learning metrics.

    Provides a unified interface to compute all metrics at once.
    """

    def __init__(self, performance_matrix: np.ndarray) -> None:
        self.performance_matrix = performance_matrix
        self.n_tasks = performance_matrix.shape[0]

    def compute_all(self) -> Dict[str, float]:
        return {
            "acc": compute_acc(self.performance_matrix),
            "bwt": compute_bwt(self.performance_matrix),
            "forgetting": compute_forgetting(self.performance_matrix),
            "fwt": compute_fwt(self.performance_matrix),
        }

    def summary(self) -> str:
        metrics = self.compute_all()
        return (
            f"ACC: {metrics['acc']:.4f} | "
            f"BWT: {metrics['bwt']:.4f} | "
            f"Forgetting: {metrics['forgetting']:.4f} | "
            f"FWT: {metrics['fwt']:.4f}"
        )