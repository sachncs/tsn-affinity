"""Mask warm-starting from source task to accelerate new task learning."""

import torch

from tsn_affinity.sparse.module_converter import iter_sparse_modules


class MaskWarmstarter:
    """Warm-starts mask scores from source task to accelerate new task learning.

    When a new task is routed to a copy with a source task, the mask scores
    of the new task are initialized from the source task's masks rather than
    random initialization. This provides a good starting point for the new
    task's sparse mask.

    Attributes:
        warmstart: Whether to apply warm-starting.
        strength: Scaling factor for source scores.
        noise_std: Standard deviation of noise to add to source scores.
    """

    def __init__(
        self,
        warmstart: bool = True,
        strength: float = 2.0,
        noise_std: float = 0.02,
    ) -> None:
        self.warmstart = bool(warmstart)
        self.strength = float(strength)
        self.noise_std = float(noise_std)

    def apply(
        self,
        model: torch.nn.Module,
        source_task_id: int,
        task_to_copy: dict,
        copy_states: list,
    ) -> None:
        """Apply warm-start to model scores from source task.

        Args:
            model: The model whose scores to warm-start.
            source_task_id: ID of the source task.
            task_to_copy: Dict mapping task_id -> copy_id.
            copy_states: List of model copy states.
        """
        if not self.warmstart or source_task_id is None:
            return

        source_copy_id = task_to_copy.get(source_task_id, None)
        if source_copy_id is None:
            return

        src_state = copy_states[int(source_copy_id)]
        src_masks = src_state.per_task_masks.get(source_task_id, None)
        if src_masks is None:
            return

        with torch.no_grad():
            for name, mod in iter_sparse_modules(model):
                w_key = f"{name}.weight"
                src_w = src_masks.get(w_key, None)
                if src_w is not None:
                    src_w = src_w.to(
                        device=mod.score.device, dtype=mod.score.dtype
                    )
                    mod.score.normal_(mean=0.0, std=self.noise_std)
                    mod.score.add_(self.strength * src_w)

                if getattr(mod, "bias_score", None) is not None:
                    b_key = f"{name}.bias"
                    src_b = src_masks.get(b_key, None)
                    if src_b is not None:
                        src_b = src_b.to(
                            device=mod.bias_score.device, dtype=mod.bias_score.dtype
                        )
                        mod.bias_score.normal_(mean=0.0, std=self.noise_std)
                        mod.bias_score.add_(self.strength * src_b)

    def apply_to_new_copy(
        self,
        model: torch.nn.Module,
        source_task_id: int,
        task_to_copy: dict,
        copy_states: list,
    ) -> None:
        """Apply warm-start even when creating a new copy.

        Args:
            model: The model whose scores to warm-start.
            source_task_id: ID of the source task.
            task_to_copy: Dict mapping task_id -> copy_id.
            copy_states: List of model copy states.
        """
        if not self.warmstart or source_task_id is None:
            return

        self.apply(model, source_task_id, task_to_copy, copy_states)