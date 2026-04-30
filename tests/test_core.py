"""Tests for Decision Transformer core components."""

import torch
import pytest

from tsn_affinity.core.attention import CausalSelfAttention, MLP, Block, LayerNorm
from tsn_affinity.core.obs_encoder import ObsEncoder
from tsn_affinity.core.decision_transformer import DecisionTransformer, DTBackbone


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


class TestObsEncoder:
    def test_mlp_encoder(self):
        encoder = ObsEncoder(obs_shape=(10,), d_model=128, encoder_type="mlp")
        x = torch.randn(4, 10)
        out = encoder(x)
        assert out.shape == (4, 128)

    def test_cnn_encoder(self):
        encoder = ObsEncoder(obs_shape=(4, 84, 84), d_model=128, encoder_type="cnn")
        x = torch.randn(4, 4, 84, 84)
        out = encoder(x)
        assert out.shape == (4, 128)


class TestDecisionTransformer:
    def test_forward(self):
        dt = DecisionTransformer(obs_shape=(4,), n_actions=2, d_model=64, n_layers=2, n_heads=2)
        B, K = 2, 10
        obs = torch.randn(B, K, 4)
        actions = torch.randint(0, 2, (B, K))
        rtg = torch.randn(B, K, 1)
        ts = torch.arange(K).unsqueeze(0).repeat(B, 1)
        logits = dt(obs, actions, rtg, ts)
        assert logits.shape == (B, K, 2)

    def test_reset_history(self):
        dt = DecisionTransformer(obs_shape=(4,), n_actions=2)
        dt.reset_history()
        assert hasattr(dt, "_obs_history")
        assert len(dt._obs_history) == 0