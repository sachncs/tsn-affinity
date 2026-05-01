"""Top-K Mask Straight-Through Estimator autograd function."""

import torch


class TopKMaskSTE(torch.autograd.Function):
    """Straight-through estimator for top-k binary mask selection.

    Forward: Selects the top-k elements by magnitude from scores, returns binary mask.
    Backward: Passes gradients unchanged (STE).

    Attributes:
        None (stateless function).

    Usage:
        mask = TopKMaskSTE.apply(scores, keep_ratio, free_mask)
    """

    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        keep_ratio: float,
        free_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute top-k binary mask from scores."""
        flat_scores = scores.reshape(-1)
        n_all = flat_scores.numel()
        if n_all == 0:
            return torch.zeros_like(scores)

        keep_ratio_clamped = max(0.0, min(1.0, keep_ratio))

        if free_mask is not None:
            flat_free = free_mask.reshape(-1).to(
                device=flat_scores.device, dtype=torch.bool
            )
            n_free = flat_free.sum().item()
            if n_free == 0:
                return torch.zeros_like(scores)

            k_keep = max(1, int(round(keep_ratio_clamped * n_free)))
            if k_keep >= n_free:
                return flat_free.view_as(scores).to(dtype=scores.dtype)

            work = flat_scores.clone()
            work[~flat_free] = -torch.inf
            topk_vals = work.topk(k=k_keep, largest=True, sorted=True).values
            threshold = topk_vals[-1]
            mask = (work >= threshold) & flat_free
            return mask.view_as(scores).to(dtype=scores.dtype)

        k_keep = max(1, int(round(keep_ratio_clamped * n_all)))
        if k_keep >= n_all:
            return torch.ones_like(scores)
        topk_vals = flat_scores.topk(k=k_keep, largest=True, sorted=True).values
        threshold = topk_vals[-1]
        return (flat_scores >= threshold).view_as(scores).to(dtype=scores.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through backward (passes grad unchanged)."""
        return grad_output, None, None
