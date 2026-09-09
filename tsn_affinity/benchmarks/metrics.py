"""Continual-learning metrics for performance-matrix analysis.

A *performance matrix* is a square array ``R[i, j]`` where entry
``i, j`` is the score the model achieved on task ``j`` after being
trained on task ``i``. The metrics computed here follow the definitions
used in the TSN-Affinity paper and the broader continual-learning
literature.

Public functions:
    - ``compute_acc``: Mean of the diagonal (after-task performance).
    - ``compute_bwt``: Backward transfer (forgetting of previous tasks).
    - ``compute_forgetting``: Average drop in score on each task.
    - ``compute_fwt``: Forward transfer (zero-shot transfer to future tasks).

Public class:
    - ``StandardCLMetrics``: Bundles the four metrics and provides
      ``compute_all()`` / ``summary()`` helpers.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_acc",
    "compute_bwt",
    "compute_forgetting",
    "compute_fwt",
    "StandardCLMetrics",
]


def compute_acc(performance_matrix: np.ndarray) -> float:
    """Compute ACC: mean of the diagonal entries.

    ACC measures the model's average score on each task immediately
    after it has been trained on that task.

    Args:
        performance_matrix: ``[n_tasks, n_tasks]`` matrix where ``R[i, j]``
            is the score on task ``j`` after training on task ``i``.

    Returns:
        ACC value in the same units as the matrix.
    """
    n = int(performance_matrix.shape[0])
    if n == 0:
        return 0.0
    return float(np.diag(performance_matrix).mean())


def compute_bwt(performance_matrix: np.ndarray) -> float:
    """Compute BWT: backward transfer (negative when forgetting occurs).

    BWT = (1 / (T - 1)) * sum_{i < T} (R[T-1, i] - R[i, i])

    Args:
        performance_matrix: ``[n_tasks, n_tasks]`` performance matrix.

    Returns:
        BWT value (negative indicates forgetting).
    """
    n = int(performance_matrix.shape[0])
    if n <= 1:
        return 0.0
    diag = np.diag(performance_matrix)
    last_row = performance_matrix[n - 1, : n - 1]
    return float((last_row - diag[: n - 1]).mean())


def compute_forgetting(performance_matrix: np.ndarray) -> float:
    """Compute average forgetting across tasks.

    Forgetting for task ``i`` is the maximum drop from the post-training
    score ``R[i, i]`` to the final evaluation ``R[T-1, i]``. The reported
    value is the mean of the per-task forgetting across tasks ``0..T-2``.

    Args:
        performance_matrix: ``[n_tasks, n_tasks]`` performance matrix.

    Returns:
        Forgetting value (>= 0; 0 means no forgetting).
    """
    n = int(performance_matrix.shape[0])
    if n <= 1:
        return 0.0
    diag = np.diag(performance_matrix)
    final_row = performance_matrix[n - 1, :]
    forgetting = (diag - final_row).clip(min=0.0)[: n - 1]
    return float(forgetting.mean())


def compute_fwt(performance_matrix: np.ndarray) -> float:
    """Compute FWT: forward transfer (positive means transfer helps).

    FWT = (1 / (T - 1)) * sum_{j > i} R[i, j]

    Args:
        performance_matrix: ``[n_tasks, n_tasks]`` performance matrix.

    Returns:
        FWT value (>= 0 means transfer to future tasks was positive).
    """
    n = int(performance_matrix.shape[0])
    if n <= 1:
        return 0.0
    upper = performance_matrix[np.triu_indices(n, k=1)]
    if upper.size == 0:
        return 0.0
    return float(upper.mean())


class StandardCLMetrics:
    """Bundle of standard continual-learning metrics for a performance matrix.

    Use ``compute_all`` for a dictionary view or ``summary`` for a
    human-readable, uppercase-keyed variant suitable for logging.

    Attributes:
        performance_matrix: The ``[n_tasks, n_tasks]`` matrix passed in.
        n_tasks: Number of tasks (matrix dimension).
    """

    def __init__(self, performance_matrix: np.ndarray) -> None:
        """Initialize with a performance matrix.

        Args:
            performance_matrix: ``[n_tasks, n_tasks]`` matrix where
                ``R[i, j]`` is the score on task ``j`` after training on
                task ``i``.
        """
        self.performance_matrix = np.asarray(performance_matrix)
        if self.performance_matrix.ndim != 2:
            raise ValueError(
                f"performance_matrix must be 2D, got {self.performance_matrix.ndim}D"
            )
        if self.performance_matrix.shape[0] != self.performance_matrix.shape[1]:
            raise ValueError(
                "performance_matrix must be square, "
                f"got shape {self.performance_matrix.shape}"
            )
        self.n_tasks = int(self.performance_matrix.shape[0])

    def compute_all(self) -> dict[str, float]:
        """Compute all four standard metrics.

        Returns:
            Dictionary with keys ``acc``, ``bwt``, ``forgetting``, ``fwt``.
        """
        return {
            "acc": compute_acc(self.performance_matrix),
            "bwt": compute_bwt(self.performance_matrix),
            "forgetting": compute_forgetting(self.performance_matrix),
            "fwt": compute_fwt(self.performance_matrix),
        }

    def summary(self) -> dict[str, float]:
        """Return a human-readable summary with uppercase keys.

        Returns:
            Dictionary with keys ``ACC``, ``BWT``, ``Forgetting``,
            ``FWT`` containing the same numeric values as ``compute_all``.
        """
        metrics = self.compute_all()
        return {
            "ACC": metrics["acc"],
            "BWT": metrics["bwt"],
            "Forgetting": metrics["forgetting"],
            "FWT": metrics["fwt"],
        }
