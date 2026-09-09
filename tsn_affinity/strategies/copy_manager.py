"""Lifecycle management for the per-task model copies.

Multi-copy continual-learning strategies (TSN-Affinity, TSN-ReplayKL)
need a small registry to keep track of the per-task ``ModelCopy``
objects, the active copy, and the helper methods that create new
copies from a model factory. :class:`CopyManager` is the single place
that implements those concerns; the strategies own the public state
(``model``, ``optimizer``, mask dictionaries) and call into the
manager for bookkeeping.

Typical use:

    cm = CopyManager(device, model_config, sparse_config)
    cm.create_initial_copy(initial_model, initial_optimizer)
    new_id = cm.create_new_copy(model_factory, optimizer_factory)
    model = cm.activate_copy(new_id)
    active_state = cm.get_active_state()

The class previously held its own copy of the lifecycle logic that the
strategies duplicated; the duplication has been removed and the
strategies now defer to :class:`CopyManager` (see :class:`TSNAffinityStrategy`).
The class is kept public so downstream code can compose it with new
strategies or external trackers.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from tsn_affinity.core.config import ModelConfig
from tsn_affinity.sparse.converter import (
    SparseConversionConfig,
    convert_to_sparse,
    rebuild_optimizer,
)
from tsn_affinity.strategies.model_copy import ModelCopy

__all__ = ["CopyManager"]


class CopyManager:
    """Manages lifecycle of model copies for multi-copy TSN strategies.

    Handles creation, activation, syncing, and state management of
    model copies used for different tasks.

    Attributes:
        device: Device to place models on.
        model_config: Model architecture configuration.
        sparse_config: Sparse layer configuration.
        copies: List of registered :class:`ModelCopy` objects.
        active_copy_id: ID of the currently active copy.
    """

    device: str
    model_config: ModelConfig
    sparse_config: SparseConversionConfig
    copies: list[ModelCopy]
    active_copy_id: int

    def __init__(
        self,
        device: str,
        model_config: ModelConfig,
        sparse_config: SparseConversionConfig,
    ) -> None:
        """Initialize the copy manager.

        Args:
            device: Torch device used for all copies.
            model_config: Model architecture configuration.
            sparse_config: Sparse layer configuration used to convert
                new copies to sparse form.
        """
        self.device = device
        self.model_config = model_config
        self.sparse_config = sparse_config
        self.copies: list[ModelCopy] = []
        self.active_copy_id = 0

    def create_initial_copy(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> int:
        """Create the first model copy from an existing model/optimizer.

        Converts the model to sparse layers and stores it as copy 0.

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
            model_factory: Callable that returns a fresh
                :class:`DecisionTransformer` (or any nn.Module).
            optimizer_factory: Callable that takes the created model
                and returns a configured optimizer.

        Returns:
            ID of the newly created copy.
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
        """No-op: CopyManager does not hold public state.

        Strategies that need public-state syncing implement this logic
        directly within the strategy class.
        """
        return None

    def get_active_state(self) -> ModelCopy:
        """Return the currently active :class:`ModelCopy`.

        Returns:
            Active :class:`ModelCopy`.
        """
        return self.copies[self.active_copy_id]

    def get_copy_id_for_task(self, task_id: int) -> int | None:
        """Return the copy ID that owns ``task_id``'s masks, if any.

        Args:
            task_id: Task ID to look up.

        Returns:
            Copy ID, or ``None`` when the task has not been registered.
        """
        for copy_id, copy in enumerate(self.copies):
            if task_id in copy.per_task_masks:
                return copy_id
        return None
