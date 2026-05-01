"""Sparse embedding layer with learnable score-based mask selection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsn_affinity.sparse.base import TSNNMaskMixin
from tsn_affinity.sparse.topk import TopKMaskSTE


class TSNSparseEmbedding(nn.Embedding, TSNNMaskMixin):
    """Sparse embedding with learnable score-based mask selection.

    Each weight element has a corresponding learnable score. During forward
    pass, top-k elements by score magnitude are kept (其余置零).
    The k is determined by keep_ratio * n_free_elements.

    Uses straight-through gradient estimation for the mask.

    Attributes:
        keep_ratio: Fraction of weights to keep per task.
        allow_weight_reuse: Whether to allow overlapping with occupied weights.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        keep_ratio: float = 0.5,
        allow_weight_reuse: bool = False,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        self.keep_ratio = float(keep_ratio)
        self.allow_weight_reuse = bool(allow_weight_reuse)

        self.score = nn.Parameter(torch.empty_like(self.weight))
        self.register_parameter("bias_score", None)

        self.weight_mask: torch.Tensor | None = None
        self.bias_mask: torch.Tensor | None = None
        self.active_weight_mask: torch.Tensor | None = None
        self.active_bias_mask: torch.Tensor | None = None
        self.occupied_weight_mask: torch.Tensor | None = None
        self.occupied_bias_mask: torch.Tensor | None = None

        self.reset_scores()

    def reset_scores(self) -> None:
        """Reset score parameters."""
        torch.nn.init.normal_(self.score, mean=0.0, std=0.02)

    def current_weight_mask(self) -> torch.Tensor:
        """Compute current weight mask from scores."""
        if self.active_weight_mask is not None:
            mask = self.active_weight_mask.to(
                device=self.weight.device, dtype=self.weight.dtype
            )
        else:
            mask = TopKMaskSTE.apply(
                self.score.abs(),
                float(self.keep_ratio),
                self.get_free_weight_mask(),
            )
        self.weight_mask = mask.detach().to(dtype=torch.uint8)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply masked embedding lookup to input."""
        wm = self.current_weight_mask()
        w = self.weight * wm
        return F.embedding(
            x,
            w,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
