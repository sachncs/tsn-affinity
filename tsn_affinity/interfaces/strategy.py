"""Strategy interface definitions."""

from abc import ABC, abstractmethod

from tsn_affinity.data.trajectory import Trajectory


class StrategyInterface(ABC):
    """Abstract interface for continual learning strategies."""

    @abstractmethod
    def train_task(
        self,
        task_trajs: list[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """Train on a single task's trajectory data.

        Args:
            task_trajs: List of trajectories for the current task.
            steps: Number of training steps.
            batch_size: Batch size for training.

        Returns:
            Dictionary of training metrics (e.g., loss).
        """
        ...

    @abstractmethod
    def after_task(self, task_trajs: list[Trajectory]) -> None:
        """Called after training on a task completes.

        Args:
            task_trajs: List of trajectories for the task that just finished.
        """
        ...

    def set_eval_task(self, task_id: int) -> None:
        """Set the task for evaluation.

        Args:
            task_id: Identifier of the task to evaluate.
        """
        return None

    def clear_eval_task(self) -> None:
        """Clear evaluation task."""
        return None

    def has_task_mask(self, task_id: int) -> bool:
        """Check if a task mask exists.

        Args:
            task_id: Task identifier.

        Returns:
            True if a mask exists for the task.
        """
        return False
