"""High-level training service for continual learning loops.

Encapsulates the train-evaluate loop to keep strategies focused
on algorithmic logic rather than orchestration.
"""

from tsn_affinity.data.trajectory import Trajectory
from tsn_affinity.interfaces.strategy import StrategyInterface


class TrainingService:
    """Orchestrates training across multiple tasks."""

    def __init__(self, strategy: StrategyInterface) -> None:
        """Initialize with a strategy instance.

        Args:
            strategy: Strategy implementing StrategyInterface.
        """
        self.strategy = strategy

    def train_tasks(
        self,
        task_trajectories: list[list[Trajectory]],
        steps: int = 2000,
        batch_size: int = 64,
    ) -> dict[int, dict[str, float]]:
        """Train sequentially on all tasks.

        Args:
            task_trajectories: List of trajectory lists, one per task.
            steps: Training steps per task.
            batch_size: Batch size.

        Returns:
            Dict mapping task_id -> training metrics.
        """
        results: dict[int, dict[str, float]] = {}
        for task_id, trajs in enumerate(task_trajectories):
            metrics = self.strategy.train_task(
                trajs, steps=steps, batch_size=batch_size
            )
            self.strategy.after_task(trajs)
            results[task_id] = metrics
        return results
