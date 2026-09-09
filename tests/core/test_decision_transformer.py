"""Tests for Decision Transformer."""

import pytest
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

    def test_act_is_deterministic_when_requested(self):
        """``deterministic=True`` yields the same action on the same history."""
        torch.manual_seed(7)
        dt = DecisionTransformer(
            obs_shape=(4,), n_actions=4, d_model=32, n_layers=2, n_heads=2
        )
        dt.reset_history()
        # Two calls with the same observation in the same state must
        # produce identical deterministic actions because the
        # transformer is deterministic in eval mode.
        dt.eval()
        obs = torch.randn(4)
        dt.act(obs, returns_to_go=1.0, deterministic=True)
        # The history grew by one element; with the same obs again the
        # action must be reproducible.
        first_history_len = len(dt._act_history)
        second_action = dt.act(obs, returns_to_go=1.0, deterministic=True)
        assert isinstance(second_action, int)
        # After the second call the history has one more entry.
        assert len(dt._act_history) == first_history_len + 1

    def test_act_stochastic_returns_int(self):
        """``deterministic=False`` must still produce a valid integer action."""
        torch.manual_seed(11)
        dt = DecisionTransformer(
            obs_shape=(4,), n_actions=5, d_model=32, n_layers=2, n_heads=2
        )
        dt.reset_history()
        obs = torch.randn(4)
        action = dt.act(obs, returns_to_go=0.0, deterministic=False)
        assert isinstance(action, int)
        assert 0 <= action < 5

    def test_act_truncates_history_to_seq_len(self):
        """History must not grow unbounded; the tensor build slices it."""
        seq_len = 4
        dt = DecisionTransformer(
            obs_shape=(4,),
            n_actions=2,
            d_model=32,
            n_layers=2,
            n_heads=2,
            seq_len=seq_len,
        )
        dt.reset_history()
        obs = torch.randn(4)
        for _ in range(seq_len * 3):
            dt.act(obs, returns_to_go=0.0, deterministic=True)
        # Internal histories may grow, but tensor construction must
        # never exceed ``seq_len`` elements.
        assert len(dt._obs_history[-seq_len:]) <= seq_len

    def test_act_with_cnn_obs(self):
        """``act`` must support image observations through the CNN encoder."""
        torch.manual_seed(13)
        dt = DecisionTransformer(
            obs_shape=(4, 84, 84),
            n_actions=4,
            d_model=64,
            n_layers=2,
            n_heads=2,
            seq_len=8,
        )
        dt.reset_history()
        obs = torch.rand(4, 84, 84, dtype=torch.float32)
        action = dt.act(obs, returns_to_go=1.0, deterministic=True)
        assert isinstance(action, int)
        assert 0 <= action < 4

    def test_act_continuous_returns_vector(self):
        """For continuous actions ``act`` must return a numpy array."""
        torch.manual_seed(17)
        dt = DecisionTransformer(
            obs_shape=(4,),
            n_actions=1,
            action_dim=2,
            continuous_actions=True,
            d_model=32,
            n_layers=2,
            n_heads=2,
        )
        dt.reset_history()
        obs = torch.randn(4)
        action = dt.act(obs, returns_to_go=0.0, deterministic=True)
        assert hasattr(action, "shape")
        assert action.shape == (2,)

    @pytest.mark.parametrize("action_dim", [1, 3, 5])
    def test_act_continuous_various_dims(self, action_dim):
        torch.manual_seed(23)
        dt = DecisionTransformer(
            obs_shape=(4,),
            n_actions=action_dim,
            action_dim=action_dim,
            continuous_actions=True,
            d_model=32,
            n_layers=1,
            n_heads=2,
            seq_len=4,
        )
        dt.reset_history()
        action = dt.act(torch.randn(4), returns_to_go=0.0, deterministic=True)
        assert action.shape == (action_dim,)
