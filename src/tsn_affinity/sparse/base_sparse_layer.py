"""Base mixin for TSN sparse layers.

Provides mask management interface for sparse layers.
"""

import torch


class TSNNMaskMixin:
    """Mixin providing mask management for TSN sparse layers.

    Subclasses must implement weight and bias properties, and provide
    mask management via keep_ratio, allow_weight_reuse, and related attributes.

    Attributes:
        keep_ratio: Fraction of weights to keep.
        allow_weight_reuse: Whether to allow overlapping with occupied weights.
        occupied_weight_mask: Mask of weights occupied by previous tasks.
        occupied_bias_mask: Mask of biases occupied by previous tasks.
        active_weight_mask: Current eval mask for weights.
        active_bias_mask: Current eval mask for biases.
        weight_mask: Last computed forward-pass weight mask.
        bias_mask: Last computed forward-pass bias mask.
    """

    keep_ratio: float
    allow_weight_reuse: bool

    def clear_active_masks(self) -> None:
        self.active_weight_mask = None
        self.active_bias_mask = None

    def clear_occupied_masks(self) -> None:
        self.occupied_weight_mask = None
        self.occupied_bias_mask = None

    def get_free_weight_mask(self) -> torch.Tensor | None:
        if self.allow_weight_reuse or self.occupied_weight_mask is None:
            return None
        return ~self.occupied_weight_mask.to(
            device=self.weight.device, dtype=torch.bool
        )

    def get_free_bias_mask(self) -> torch.Tensor | None:
        if getattr(self, "bias", None) is None:
            return None
        if self.allow_weight_reuse or self.occupied_bias_mask is None:
            return None
        return ~self.occupied_bias_mask.to(
            device=self.bias.device, dtype=torch.bool
        )