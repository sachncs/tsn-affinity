"""Transformer attention components and building blocks."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer normalization with optional bias.

    Operates on the last dimension of the input tensor.

    Attributes:
        normalized_shape: Shape of the dimension to normalize.
        bias: Whether to include a learnable bias.
    """

    def __init__(self, normalized_shape: int, bias: bool = True) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias)


class MLP(nn.Module):
    """Feedforward network with GELU activation.

    Expands from d_model to 4*d_model then projects back.

    Attributes:
        d_model: Input and output dimension.
        expanded_dim: Intermediate dimension (4 * d_model).
        dropout: Dropout probability.
    """

    def __init__(
        self, d_model: int, expanded_dim: int | None = None, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if expanded_dim is None:
            expanded_dim = d_model * 4
        self.d_model = d_model
        self.expanded_dim = expanded_dim
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(d_model, expanded_dim)
        self.fc2 = nn.Linear(expanded_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    """GPT-style causal self-attention.

    Supports both PyTorch's scaled_dot_product_attention and a manual
    fallback implementation.

    Attributes:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.
            attention_mask: Optional attention mask.

        Returns:
            Output tensor.
        """
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        use_pytorch_sdpa = hasattr(F, "scaled_dot_product_attention")

        # Build causal mask (True = future positions to mask out)
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        if use_pytorch_sdpa:
            attn_mask = torch.zeros(B, self.n_heads, T, T, device=x.device)
            attn_mask = attn_mask.masked_fill(
                causal.unsqueeze(0).unsqueeze(0), float("-inf")
            )
            if attention_mask is not None:
                padding_mask = attention_mask.view(B, 1, 1, T).expand(
                    -1, self.n_heads, T, -1
                )
                attn_mask = attn_mask.masked_fill(padding_mask == 0, float("-inf"))
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            attn_scores = attn_scores.masked_fill(
                causal.unsqueeze(0).unsqueeze(0), float("-inf")
            )

            if attention_mask is not None:
                mask = attention_mask.view(B, 1, 1, T).expand(-1, self.n_heads, T, -1)
                attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out  # type: ignore[no-any-return]


class Block(nn.Module):
    """Transformer block with causal attention and MLP.

    Applies pre-norm residual connections: x = x + Attention(LayerNorm(x))

    Attributes:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(d_model, bias=True)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = LayerNorm(d_model, bias=True)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.
            attention_mask: Optional attention mask.

        Returns:
            Output tensor.
        """
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x  # type: ignore[no-any-return]
