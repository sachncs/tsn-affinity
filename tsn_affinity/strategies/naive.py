"""Naive baseline strategy - single model, no CL mechanism."""

import logging

import torch

from tsn_affinity.core.config import ModelConfig
from tsn_affinity.data.batch import make_minibatches, masked_cross_entropy
from tsn_affinity.data.trajectory import Trajectory
from tsn_affinity.strategies.base import BaseStrategy

logger = logging.getLogger("tsn_affinity")


class NaiveStrategy(BaseStrategy):
    """Naive baseline: single Decision Transformer without any continual learning.

    This strategy trains a single model on each task sequentially with no
    protection against forgetting. It serves as the baseline to demonstrate
    catastrophic forgetting in the CL setting.

    This corresponds to the "cumulative" baseline in the paper (single model,
    sequential training, no replay).
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        seq_len: int = 20,
        device: str = "cuda",
        model_config: ModelConfig | None = None,
    ) -> None:
        super().__init__(obs_shape, n_actions, seq_len, device, model_config)
        self.current_task_id = 0

    def train_task(
        self,
        task_trajs: list[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """Train on a single task's trajectory data."""
        self.model.train()
        loader = make_minibatches(task_trajs, self.seq_len, batch_size, self.device)

        total_loss = 0.0
        for iteration in range(int(steps)):
            obs, actions, rtg, ts, mask = next(loader)

            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = masked_cross_entropy(logits, actions, mask)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += float(loss.detach().item())

            if iteration % 500 == 0 or iteration == int(steps) - 1:
                avg_loss = total_loss / (iteration + 1)
                logger.info(
                    "[naive] task=%d iteration=%d loss=%.6f",
                    self.current_task_id,
                    iteration,
                    avg_loss,
                )

        return {"loss": total_loss / max(1, int(steps)), "keep_ratio": 1.0}

    def after_task(self, task_trajs: list[Trajectory]) -> None:
        """Called after training on a task completes."""
        self.current_task_id += 1

    def set_eval_task(self, task_id: int) -> None:
        """Set the task for evaluation."""

    def clear_eval_task(self) -> None:
        """Clear evaluation task."""

    def has_task_mask(self, task_id: int) -> bool:
        """Check if a task mask exists."""
        return False
