"""Copy manager for lifecycle management of model copies."""

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from tsn_affinity.core.config import ModelConfig
from tsn_affinity.core.decision_transformer import DecisionTransformer
from tsn_affinity.sparse.module_converter import SparseConversionConfig, convert_to_sparse, rebuild_optimizer
from tsn_affinity.strategies.model_copy import ModelCopy


class CopyManager:
    """Manages lifecycle of model copies for multi-copy TSN strategies.

    Handles creation, activation, syncing, and state management of
    model copies used for different tasks.

    Attributes:
        device: Device to place models on.
        model_config: Model architecture configuration.
        sparse_config: Sparse layer configuration.
    """

    def __init__(
        self,
        device: str,
        model_config: ModelConfig,
        sparse_config: SparseConversionConfig,
    ) -> None:
        self.device = device
        self.model_config = model_config
        self.sparse_config = sparse_config
        self.copies: List[ModelCopy] = []
        self.active_copy_id: int = 0

    def create_initial_copy(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> int:
        """Create the first model copy from an existing model/optimizer.

        Converts the model to sparse layers and stores as copy 0.

        Args:
            model: Initial Decision Transformer model.
            optimizer: Initial optimizer.

        Returns:
            Copy ID (0).
        """
        convert_to_sparse(model, self.sparse_config)
        model.to(self.device)
        new_opt = rebuild_optimizer(optimizer, model.parameters())

        copy = ModelCopy(
            model=model,
            optimizer=new_opt,
            per_task_masks={},
            consolidated_masks={},
            task_codebooks={},
            task_keep_ratios={},
        )
        self.copies.append(copy)
        self.active_copy_id = 0
        return 0

    def create_new_copy(
        self,
        model_factory: Callable[[], nn.Module],
        optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    ) -> int:
        """Create a fresh model copy.

        Args:
            model_factory: Function that creates a new DecisionTransformer.
            optimizer_factory: Function that creates an optimizer for a model.

        Returns:
            New copy ID.
        """
        model = model_factory()
        convert_to_sparse(model, self.sparse_config)
        model.to(self.device)
        optimizer = optimizer_factory(model)

        copy = ModelCopy(
            model=model,
            optimizer=optimizer,
            per_task_masks={},
            consolidated_masks={},
            task_codebooks={},
            task_keep_ratios={},
        )
        self.copies.append(copy)
        return len(self.copies) - 1

    def activate_copy(self, copy_id: int) -> nn.Module:
        """Activate a specific copy as the current active model.

        Args:
            copy_id: ID of copy to activate.

        Returns:
            The activated model.
        """
        self.active_copy_id = int(copy_id)
        return self.copies[self.active_copy_id].model

    def sync_public_state_to_active_copy(self) -> None:
        """Sync current public model/optimizer state to active copy.

        Called before switching to a different copy to preserve training progress.
        """
        if not hasattr(self, "copies") or not self.copies:
            return
        if self.active_copy_id < 0 or self.active_copy_id >= len(self.copies):
            return

        active = self.copies[self.active_copy_id]
        if hasattr(self, "public_model"):
            active.model = self.public_model
        if hasattr(self, "public_optimizer"):
            active.optimizer = self.public_optimizer

    def get_active_state(self) -> ModelCopy:
        """Get the currently active copy state.

        Returns:
            Active ModelCopy.
        """
        return self.copies[self.active_copy_id]

    def get_copy_id_for_task(self, task_id: int) -> Optional[int]:
        """Get copy ID assigned to a task.

        Args:
            task_id: Task ID.

        Returns:
            Copy ID or None if task not registered.
        """
        for copy_id, copy in enumerate(self.copies):
            if task_id in copy.per_task_masks:
                return copy_id
        return None