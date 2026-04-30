"""Training utilities for gradient handling and state management."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class TrainingSnapshot:
    """Snapshot of frozen parameter values for restoration after training.

    Attributes:
        name_to_value_and_mask: Dict of param_name -> (value_tensor, mask_tensor).
        Mask is None for non-frozen params that need full restoration.
    """

    name_to_value_and_mask: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]


def verify_frozen_gradient_zeroing(
    model: torch.nn.Module,
    consolidated_masks: Dict[str, torch.Tensor],
    name_to_module: Dict[str, torch.nn.Module],
) -> Tuple[bool, str]:
    """Verify that gradients on frozen parameters are properly zeroed.

    Args:
        model: PyTorch model.
        consolidated_masks: Dict of param_name -> frozen mask.
        name_to_module: Dict mapping module name to module.

    Returns:
        Tuple of (is_valid, message).
    """
    issues = []
    for key, old_mask in consolidated_masks.items():
        if old_mask is None:
            continue
        module_name, _, attr = key.rpartition(".")
        module = name_to_module.get(module_name, None)
        if module is None:
            issues.append(f"Module not found for key: {key}")
            continue
        param = getattr(module, attr, None)
        if param is None:
            issues.append(f"Param not found: {key}")
            continue
        if param.grad is None:
            continue
        mask = old_mask.to(device=param.grad.device, dtype=torch.bool)
        frozen_grad = param.grad.masked_select(mask)
        if frozen_grad.abs().sum() > 1e-12:
            issues.append(
                f"Non-zero frozen gradient at {key}: {frozen_grad.abs().sum():.6e}"
            )
    if issues:
        return False, "; ".join(issues)
    return True, "OK"


def snapshot_frozen_parameters(
    model: torch.nn.Module,
    consolidated_masks: Dict[str, torch.Tensor],
    maskable_param_names: set,
    score_param_names: set,
    freeze_non_maskable_params: bool = True,
    current_task_id: int = 0,
) -> TrainingSnapshot:
    """Take a snapshot of frozen parameter values before optimizer step.

    Args:
        model: PyTorch model.
        consolidated_masks: Dict of param_name -> frozen mask.
        maskable_param_names: Set of maskable parameter names.
        score_param_names: Set of score parameter names.
        freeze_non_maskable_params: Whether to freeze non-maskable params after task 0.
        current_task_id: Current task ID.

    Returns:
        TrainingSnapshot with frozen parameter values.
    """
    snap: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
    if current_task_id == 0:
        return TrainingSnapshot(name_to_value_and_mask=snap)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        old_mask = consolidated_masks.get(name, None)
        if old_mask is not None:
            snap[name] = (
                param.detach().clone(),
                old_mask.to(device=param.device, dtype=torch.bool),
            )
            continue

        if freeze_non_maskable_params:
            if name not in maskable_param_names and name not in score_param_names:
                snap[name] = (param.detach().clone(), None)

    return TrainingSnapshot(name_to_value_and_mask=snap)


def restore_frozen_parameters(
    model: torch.nn.Module,
    snapshot: TrainingSnapshot,
) -> None:
    """Restore frozen parameter values from snapshot after optimizer step.

    Args:
        model: PyTorch model.
        snapshot: TrainingSnapshot with frozen parameter values.
    """
    if not snapshot.name_to_value_and_mask:
        return

    name_to_param = dict(model.named_parameters())
    with torch.no_grad():
        for name, (old_value, restore_mask) in snapshot.name_to_value_and_mask.items():
            param = name_to_param.get(name, None)
            if param is None:
                continue
            if restore_mask is None:
                param.copy_(old_value)
            else:
                param.copy_(torch.where(restore_mask, old_value, param))


def zero_gradients_for_frozen_params(
    model: torch.nn.Module,
    consolidated_masks: Dict[str, torch.Tensor],
) -> None:
    """Zero gradients for parameters covered by frozen masks.

    Args:
        model: PyTorch model.
        consolidated_masks: Dict of param_name -> frozen mask.
    """
    if not consolidated_masks:
        return
    name_to_module = dict(model.named_modules())
    for key, old_mask in consolidated_masks.items():
        if old_mask is None:
            continue
        module_name, _, attr = key.rpartition(".")
        module = name_to_module.get(module_name, None)
        if module is None:
            continue
        param = getattr(module, attr, None)
        if param is None or param.grad is None:
            continue
        mask = old_mask.to(device=param.grad.device, dtype=torch.bool, non_blocking=True)
        param.grad.masked_fill_(mask, 0.0)


def zero_gradients_for_non_maskable_params(
    model: torch.nn.Module,
    maskable_param_names: set,
    score_param_names: set,
    freeze_non_maskable_after_first: bool = True,
    current_task_id: int = 0,
) -> None:
    """Zero gradients for non-maskable parameters.

    Args:
        model: PyTorch model.
        maskable_param_names: Set of maskable parameter names.
        score_param_names: Set of score parameter names.
        freeze_non_maskable_after_first: Whether to freeze non-maskable after task 0.
        current_task_id: Current task ID.
    """
    if not freeze_non_maskable_after_first or current_task_id == 0:
        return
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name in maskable_param_names or name in score_param_names:
            continue
        param.grad.zero_()