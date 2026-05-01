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
    active_weight_mask: torch.Tensor | None = None
    active_bias_mask: torch.Tensor | None = None
    occupied_weight_mask: torch.Tensor | None = None
    occupied_bias_mask: torch.Tensor | None = None
    weight_mask: torch.Tensor | None = None
    bias_mask: torch.Tensor | None = None

    def clear_active_masks(self) -> None:
        """Clear active evaluation masks."""
        self.active_weight_mask = None
        self.active_bias_mask = None

    def clear_occupied_masks(self) -> None:
        """Clear occupied masks from previous tasks."""
        self.occupied_weight_mask = None
        self.occupied_bias_mask = None

    def get_free_weight_mask(self) -> torch.Tensor | None:
        """Get mask of unoccupied free weights."""
        if self.allow_weight_reuse or self.occupied_weight_mask is None:
            return None
        weight_device = self.weight.device  # type: ignore[attr-defined]
        return ~self.occupied_weight_mask.to(
            device=weight_device,
            dtype=torch.bool,
        )

    def get_free_bias_mask(self) -> torch.Tensor | None:
        """Get mask of unoccupied free biases."""
        if getattr(self, "bias", None) is None:
            return None
        if self.allow_weight_reuse or self.occupied_bias_mask is None:
            return None
        bias_device = self.bias.device  # type: ignore[attr-defined]
        return ~self.occupied_bias_mask.to(
            device=bias_device,
            dtype=torch.bool,
        )
