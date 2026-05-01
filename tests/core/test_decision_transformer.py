"""Tests for Decision Transformer."""

import torch

from tsn_affinity.core.decision_transformer import DecisionTransformer


class TestDecisionTransformer:
    def test_forward(self):
        dt = DecisionTransformer(
            obs_shape=(4,), n_actions=2, d_model=64, n_layers=2, n_heads=2
        )
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

    def test_act_produces_valid_action(self):
        dt = DecisionTransformer(
            obs_shape=(4,), n_actions=2, d_model=64, n_layers=2, n_heads=2
        )
        dt.reset_history()
        obs = torch.randn(4)
        action = dt.act(obs, returns_to_go=0.0, deterministic=True)
        assert isinstance(action, int)
        assert 0 <= action < 2

    def test_act_accumulates_history(self):
        dt = DecisionTransformer(
            obs_shape=(4,), n_actions=2, d_model=64, n_layers=2, n_heads=2
        )
        dt.reset_history()
        obs = torch.randn(4)
        dt.act(obs, returns_to_go=0.0, deterministic=True)
        dt.act(obs, returns_to_go=0.0, deterministic=True)
        assert len(dt._obs_history) == 2
        assert len(dt._act_history) == 2
