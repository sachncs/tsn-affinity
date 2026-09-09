"""Model copy state container for multi-copy TSN strategies."""

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ModelCopy:
    """Immutable-ish container for one model copy's state.

    Attributes:
        model: The PyTorch model for this copy.
        optimizer: The PyTorch optimizer for this copy.
        per_task_masks: Dict of task_id -> {param_name -> mask_tensor}.
        consolidated_masks: Dict of param_name -> mask_tensor (union of all tasks).
        task_codebooks: Dict of task_id -> {param_name -> centroids}.
        task_keep_ratios: Dict of task_id -> keep_ratio float.
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    per_task_masks: dict[int, dict[str, torch.Tensor | None]] = field(
        default_factory=dict
    )
    consolidated_masks: dict[str, torch.Tensor | None] = field(default_factory=dict)
    task_codebooks: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    task_keep_ratios: dict[int, float] = field(default_factory=dict)
