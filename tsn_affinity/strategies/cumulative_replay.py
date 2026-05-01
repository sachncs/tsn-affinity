"""Cumulative replay strategy - maintains replay buffer of all previous tasks."""

import logging
from dataclasses import dataclass

import torch

from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.data.batch import make_minibatches, masked_cross_entropy
from tsn_affinity.data.trajectory import Trajectory
from tsn_affinity.strategies.tsn_base import TSNBaseStrategy

logger = logging.getLogger("tsn_affinity")


@dataclass
class CumulativeReplayConfig:
    """Configuration for cumulative replay strategy.

    Attributes:
        replay_capacity: Maximum number of trajectories to store in replay buffer.
        replay_ratio: Fraction of each batch from replay buffer (0.3 = 30% replay).
        keep_ratio: Sparse mask keep ratio.
    """

    replay_capacity: int = 2000
    replay_ratio: float = 0.3
    keep_ratio: float = 0.5


class CumulativeReplayStrategy(TSNBaseStrategy):
    """Cumulative replay strategy for continual learning.

    Maintains a replay buffer containing trajectories from all previously
    learned tasks. When training on a new task, also samples from the
    replay buffer to maintain performance on previous tasks.

    This is the standard cumulative replay baseline described in the paper.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        seq_len: int = 20,
        device: str = "cuda",
        model_config: ModelConfig | None = None,
        sparse_config: SparseConfig | None = None,
        replay_capacity: int = 2000,
        replay_ratio: float = 0.3,
    ) -> None:
        if sparse_config is None:
            sparse_config = SparseConfig()

        super().__init__(
            obs_shape, n_actions, seq_len, device, model_config, sparse_config
        )

        self.replay_capacity = int(replay_capacity)
        self.replay_ratio = float(replay_ratio)
        self.replay_buffer: list[Trajectory] = []

    def train_task(
        self,
        task_trajs: list[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """Train on a single task's trajectory data."""
        self._prepare_current_task()

        self.model.train()
        total_loss = 0.0

        current_steps = int(steps * (1.0 - self.replay_ratio))
        replay_steps = int(steps * self.replay_ratio)

        if current_steps > 0:
            loader = make_minibatches(task_trajs, self.seq_len, batch_size, self.device)
            for iteration in range(current_steps):
                obs, actions, rtg, ts, mask = next(loader)
                logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
                loss = masked_cross_entropy(logits, actions, mask)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total_loss += float(loss.detach().item())

                if iteration % 500 == 0 or iteration == current_steps - 1:
                    avg_loss = total_loss / (iteration + 1)
                    logger.info(
                        "[cumulative-replay] task=%d iteration=%d/%d loss=%.6f",
                        self.current_task_id,
                        iteration,
                        current_steps,
                        avg_loss,
                    )

        if replay_steps > 0 and self.replay_buffer:
            replay_batch_size = max(1, int(batch_size * self.replay_ratio))
            replay_loader = make_minibatches(
                self.replay_buffer, self.seq_len, replay_batch_size, self.device
            )
            for _iteration in range(replay_steps):
                obs, actions, rtg, ts, mask = next(replay_loader)
                logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
                loss = masked_cross_entropy(logits, actions, mask)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total_loss += float(loss.detach().item())

        self._add_to_replay_buffer(task_trajs)

        return {
            "loss": total_loss / max(1, steps),
            "keep_ratio": float(self.current_keep_ratio),
        }

    def _add_to_replay_buffer(self, task_trajs: list[Trajectory]) -> None:
        for traj in task_trajs:
            self.replay_buffer.append(traj)

        total_obs = sum(len(t.obs) for t in self.replay_buffer)
        while total_obs > self.replay_capacity and self.replay_buffer:
            removed = self.replay_buffer.pop(0)
            total_obs -= len(removed.obs)

    def after_task(self, task_trajs: list[Trajectory]) -> None:
        """Called after training on a task completes."""
        task_id = int(self.current_task_id)

        task_masks = self._collect_current_task_masks()
        self.per_task_masks[task_id] = task_masks
        self.task_codebooks[task_id] = self._quantize_new_weights(task_masks)
        self._update_consolidated_masks(task_masks)

        used = 0
        total = 0
        for key, mask in self.consolidated_masks.items():
            if mask is None or not key.endswith(".weight"):
                continue
            used += int(mask.sum().item())
            total += int(mask.numel())
        ratio = float(used / max(1, total))
        logger.info(
            "[cumulative-replay] after task %d: occupied_ratio=%.4f", task_id, ratio
        )

        self.set_eval_task(task_id)
        self.current_task_id += 1
