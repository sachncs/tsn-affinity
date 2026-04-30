"""Benchmark runner for continual learning evaluation."""

from typing import Dict, List

import numpy as np

from tsn_affinity.benchmarks.task_spec import TaskSpec
from tsn_affinity.data.trajectory import Trajectory
from tsn_affinity.strategies.base_strategy import BaseStrategy


class BenchmarkRunner:
    """Runs continual learning benchmark with task sequencing and evaluation.

    Handles:
    - Sequential task training
    - Performance matrix evaluation across all tasks
    - Metric computation

    Attributes:
        tasks: List of task specifications.
        strategy: Continual learning strategy to evaluate.
        eval_steps: Number of evaluation steps per task.
    """

    def __init__(
        self,
        tasks: List[TaskSpec],
        strategy: BaseStrategy,
        eval_steps: int = 2000,
    ) -> None:
        self.tasks = tasks
        self.strategy = strategy
        self.eval_steps = eval_steps

    def run(
        self,
        task_trajectories: List[List[Trajectory]],
    ) -> Dict[str, np.ndarray]:
        """Run the benchmark.

        Args:
            task_trajectories: List of trajectory lists, one per task.

        Returns:
            Dict with keys:
            - "performance_matrix": [n_tasks, n_tasks] matrix of evaluations
            - "task_similarity": Dict of routing decisions per task
        """
        n_tasks = len(self.tasks)
        performance_matrix = np.zeros((n_tasks, n_tasks))

        for task_id in range(n_tasks):
            task_trajs = task_trajectories[task_id]

            self.strategy.train_task(
                task_trajs,
                steps=self.eval_steps,
                batch_size=64,
            )
            self.strategy.after_task(task_trajs)

            for eval_task_id in range(n_tasks):
                if eval_task_id <= task_id:
                    self.strategy.set_eval_task(eval_task_id)
                    performance = self._evaluate_task(eval_task_id, task_trajectories[eval_task_id])
                    performance_matrix[task_id, eval_task_id] = performance

        return {
            "performance_matrix": performance_matrix,
            "task_similarity": getattr(self.strategy, "task_similarity", {}),
        }

    def _evaluate_task(self, task_id: int, trajectories: List[Trajectory]) -> float:
        self.strategy.set_eval_task(task_id)

        losses = []
        self.strategy.model.eval()

        loader = self._make_eval_loader(trajectories, self.strategy.seq_len, 64, "cuda")
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            obs, actions, rtg, ts, mask = batch
            logits = self.strategy.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = self._compute_loss(logits, actions, mask)
            losses.append(loss)

        return float(np.mean(losses)) if losses else 0.0

    def _make_eval_loader(self, trajectories, seq_len, batch_size, device):
        while True:
            batch_obs = []
            batch_actions = []
            batch_rtg = []
            batch_ts = []

            for _ in range(batch_size):
                traj = trajectories[np.random.randint(len(trajectories))]
                T = len(traj.actions)
                start = 0 if T <= seq_len else np.random.randint(0, T - seq_len + 1)
                end = min(start + seq_len, T)

                obs = traj.obs[start:end]
                act = traj.actions[start:end]
                rtg = traj.returns_to_go[start:end]
                ts = traj.timesteps[start:end]

                pad = seq_len - len(act)
                if obs.ndim == 2:
                    obs = np.pad(obs, ((0, pad), (0, 0)))
                else:
                    obs = np.pad(obs, ((0, pad), (0, 0), (0, 0), (0, 0)))
                act = np.pad(act, (0, pad), constant_values=-1)
                rtg = np.pad(rtg, (0, pad))
                ts = np.pad(ts, (0, pad))

                batch_obs.append(obs)
                batch_actions.append(act)
                batch_rtg.append(rtg[:, None])
                batch_ts.append(ts)

            import torch
            obs_t = torch.tensor(np.stack(batch_obs), dtype=torch.float32, device=device)
            act_t = torch.tensor(np.stack(batch_actions), dtype=torch.long, device=device)
            rtg_t = torch.tensor(np.stack(batch_rtg), dtype=torch.float32, device=device)
            ts_t = torch.tensor(np.stack(batch_ts), dtype=torch.long, device=device)

            yield obs_t, act_t, rtg_t, ts_t

    def _compute_loss(self, logits, actions, mask):
        import torch.nn.functional as F
        B, L, D = logits.shape
        logits_flat = logits.reshape(B * L, D)
        actions_flat = actions.reshape(B * L)
        mask_flat = mask.reshape(B * L)
        loss = F.cross_entropy(logits_flat, actions_flat, reduction="none", ignore_index=-1)
        loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
        return float(loss.detach().cpu().item())