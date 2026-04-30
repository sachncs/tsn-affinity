"""Base strategy interface for continual learning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tsn_affinity.core.config import ModelConfig
from tsn_affinity.data.trajectory import Trajectory


class BaseStrategy(ABC):
    """Abstract base class for continual learning strategies.

    All strategies must implement train_task and after_task methods.

    Attributes:
        device: Device to run computations on.
        seq_len: Maximum context length for Decision Transformer.
        grad_clip: Gradient clipping norm.
        model_hparams: Model hyperparameters dict.
        model: The Decision Transformer model.
        optimizer: The optimizer.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        n_actions: int,
        seq_len: int = 20,
        device: str = "cuda",
        model_config: ModelConfig | None = None,
    ) -> None:
        self.device = device
        self.seq_len = int(seq_len)
        self.grad_clip = 1.0

        if model_config is None:
            model_config = ModelConfig(obs_shape=obs_shape, n_actions=n_actions, seq_len=seq_len)

        self.model_config = model_config
        self.model_hparams = {
            "obs_shape": tuple(obs_shape),
            "n_actions": int(n_actions),
            "seq_len": int(seq_len),
            "d_model": int(model_config.d_model),
            "n_layers": int(model_config.n_layers),
            "n_heads": int(model_config.n_heads),
            "p_drop": float(model_config.p_drop),
            "max_ep_len": int(model_config.max_ep_len),
            "rtg_scale": float(model_config.rtg_scale),
            "lr": float(model_config.lr),
            "weight_decay": float(model_config.weight_decay),
            "grad_clip": float(model_config.grad_clip),
        }

        self._build_model()

    def _build_model(self) -> None:
        from tsn_affinity.core.decision_transformer import DecisionTransformer

        self.model = DecisionTransformer(
            obs_shape=self.model_config.obs_shape,
            n_actions=self.model_config.n_actions,
            d_model=self.model_config.d_model,
            n_layers=self.model_config.n_layers,
            n_heads=self.model_config.n_heads,
            seq_len=self.model_config.seq_len,
            p_drop=self.model_config.p_drop,
            max_ep_len=self.model_config.max_ep_len,
            rtg_scale=self.model_config.rtg_scale,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.model_config.lr,
            weight_decay=self.model_config.weight_decay,
        )

    @abstractmethod
    def train_task(
        self,
        task_trajs: List[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Train on a single task's trajectory data.

        Args:
            task_trajs: List of trajectories for the current task.
            steps: Number of training steps.
            batch_size: Batch size.

        Returns:
            Dict with training metrics (e.g., loss, keep_ratio).
        """
        pass

    @abstractmethod
    def after_task(self, task_trajs: List[Trajectory]) -> None:
        """Called after training on a task completes.

        Args:
            task_trajs: Trajectories from the just-completed task.
        """
        pass

    def set_eval_task(self, task_id: int) -> None:
        """Set the task for evaluation (activates task-specific masks).

        Args:
            task_id: Task ID to activate.
        """
        pass

    def clear_eval_task(self) -> None:
        """Clear evaluation task (deactivate task-specific masks)."""
        pass

    def has_task_mask(self, task_id: int) -> bool:
        """Check if a task mask exists.

        Args:
            task_id: Task ID to check.

        Returns:
            True if mask exists.
        """
        return False