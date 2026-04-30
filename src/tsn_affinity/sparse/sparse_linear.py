"""Sparse linear layer with learnable score-based mask selection."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsn_affinity.sparse.base_sparse_layer import TSNNMaskMixin
from tsn_affinity.sparse.topk_ste import TopKMaskSTE


class TSNSparseLinear(nn.Linear, TSNNMaskMixin):
    """Sparse linear layer with learnable score-based mask selection.

    Each weight element has a corresponding learnable score. During forward
    pass, top-k elements by score magnitude are kept (其余置零).
    The k is determined by keep_ratio * n_free_elements.

    Uses straight-through gradient estimation for the mask (gradient flows
    through even when elements are masked out).

    Attributes:
        keep_ratio: Fraction of weights to keep per task.
        allow_weight_reuse: Whether to allow overlapping with occupied weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        keep_ratio: float = 0.5,
        allow_weight_reuse: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.keep_ratio = float(keep_ratio)
        self.allow_weight_reuse = bool(allow_weight_reuse)

        self.score = nn.Parameter(torch.empty_like(self.weight))
        if bias:
            self.bias_score = nn.Parameter(torch.empty_like(self.bias))
        else:
            self.register_parameter("bias_score", None)

        self.weight_mask: Optional[torch.Tensor] = None
        self.bias_mask: Optional[torch.Tensor] = None
        self.active_weight_mask: Optional[torch.Tensor] = None
        self.active_bias_mask: Optional[torch.Tensor] = None
        self.occupied_weight_mask: Optional[torch.Tensor] = None
        self.occupied_bias_mask: Optional[torch.Tensor] = None

        self.reset_scores()

    def reset_scores(self) -> None:
        nn.init.kaiming_uniform_(self.score, a=np.sqrt(5))
        if self.bias_score is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(max(1, fan_in))
            nn.init.uniform_(self.bias_score, -bound, bound)

    def current_weight_mask(self) -> torch.Tensor:
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

    def current_bias_mask(self) -> Optional[torch.Tensor]:
        if getattr(self, "bias", None) is None:
            self.bias_mask = None
            return None
        if self.active_bias_mask is not None:
            mask = self.active_bias_mask.to(
                device=self.bias.device, dtype=self.bias.dtype
            )
        else:
            mask = TopKMaskSTE.apply(
                self.bias_score.abs(),
                float(self.keep_ratio),
                self.get_free_bias_mask(),
            )
        self.bias_mask = mask.detach().to(dtype=torch.uint8)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wm = self.current_weight_mask()
        w = self.weight * wm
        if self.bias is None:
            b = None
        else:
            bm = self.current_bias_mask()
            b = self.bias * bm if bm is not None else self.bias
        return F.linear(x, w, b)