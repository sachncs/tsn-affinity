"""Tests for attention components."""

import torch

from tsn_affinity.core.attention import MLP, Block, CausalSelfAttention, LayerNorm


class TestLayerNorm:
    def test_forward(self):
        ln = LayerNorm(128)
        x = torch.randn(4, 10, 128)
        out = ln(x)
        assert out.shape == x.shape

    def test_with_bias(self):
        ln = LayerNorm(64, bias=True)
        assert ln.bias is not None


class TestMLP:
    def test_forward(self):
        mlp = MLP(128)
        x = torch.randn(4, 10, 128)
        out = mlp(x)
        assert out.shape == x.shape

    def test_expanded_dim(self):
        mlp = MLP(128, expanded_dim=512)
        assert mlp.expanded_dim == 512


class TestCausalSelfAttention:
    def test_forward(self):
        attn = CausalSelfAttention(128, n_heads=4)
        x = torch.randn(2, 10, 128)
        out = attn(x)
        assert out.shape == x.shape

    def test_with_attention_mask(self):
        attn = CausalSelfAttention(128, n_heads=4)
        x = torch.randn(2, 10, 128)
        mask = torch.ones(2, 10)
        mask[:, 5:] = 0
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape


class TestBlock:
    def test_forward(self):
        block = Block(128, n_heads=4)
        x = torch.randn(2, 10, 128)
        out = block(x)
        assert out.shape == x.shape

    def test_with_attention_mask(self):
        block = Block(128, n_heads=4)
        x = torch.randn(2, 10, 128)
        mask = torch.ones(2, 10)
        out = block(x, attention_mask=mask)
        assert out.shape == x.shape
