"""Tests for environment adapters."""

import pytest

from tsn_affinity.benchmarks.adapters.atari import AtariAdapter
from tsn_affinity.benchmarks.adapters.base import BaseEnvAdapter
from tsn_affinity.benchmarks.adapters.panda import PandaAdapter
from tsn_affinity.benchmarks.spec import TaskSpec


class TestAtariAdapter:
    def test_is_compatible_with_atari_spec(self):
        adapter = AtariAdapter()
        spec = TaskSpec(
            name="Breakout",
            env_id="ALE/Breakout-v5",
            seed=0,
            params={"game": "ALE/Breakout-v5"},
        )
        assert adapter.is_compatible(spec)

    def test_not_compatible_with_panda_spec(self):
        adapter = AtariAdapter()
        spec = TaskSpec(name="reach", env_id=None, seed=0, params={})
        assert not adapter.is_compatible(spec)


class TestPandaAdapter:
    def test_is_compatible_with_panda_spec(self):
        adapter = PandaAdapter()
        spec = TaskSpec(name="panda_reach", env_id=None, seed=0, params={})
        assert adapter.is_compatible(spec)

    def test_create_env_raises(self):
        adapter = PandaAdapter()
        spec = TaskSpec(name="panda_reach", env_id=None, seed=0, params={})
        with pytest.raises(NotImplementedError):
            adapter.create_env(spec)


class TestBaseEnvAdapter:
    def test_is_protocol(self):
        assert hasattr(BaseEnvAdapter, "create_env")
        assert hasattr(BaseEnvAdapter, "describe")
        assert hasattr(BaseEnvAdapter, "is_compatible")


class TestAtariAdapterDescribe:
    def test_describe(self):
        adapter = AtariAdapter()
        mock_env = type("MockEnv", (), {})()
        mock_env.observation_space = type("ObsSpace", (), {"shape": (4, 84, 84)})()
        mock_env.action_space = type("ActSpace", (), {"n": 18})()
        info = adapter.describe(mock_env)
        assert info["obs_shape"] == (4, 84, 84)
        assert info["n_actions"] == 18
        assert info["env_type"] == "atari"

    def test_describe_with_shape_action_space(self):
        adapter = AtariAdapter()
        mock_env = type("MockEnv", (), {})()
        mock_env.observation_space = type("ObsSpace", (), {"shape": (4, 84, 84)})()
        mock_env.action_space = type("ActSpace", (), {"shape": [4]})()
        info = adapter.describe(mock_env)
        assert info["n_actions"] == 4

    def test_create_env(self, monkeypatch):
        adapter = AtariAdapter()
        spec = TaskSpec(
            name="Breakout",
            env_id="ALE/Breakout-v5",
            seed=0,
            params={"game": "ALE/Breakout-v5"},
        )
        mock_env = type("MockEnv", (), {})()
        monkeypatch.setattr("gymnasium.make", lambda *a, **k: mock_env)
        env = adapter.create_env(spec)
        assert env is mock_env


class TestPandaAdapterDescribe:
    def test_describe(self):
        adapter = PandaAdapter()
        mock_env = type("MockEnv", (), {})()
        mock_env.observation_space = type("ObsSpace", (), {"shape": (7,)})()
        mock_env.action_space = type("ActSpace", (), {"shape": [4]})()
        info = adapter.describe(mock_env)
        assert info["obs_shape"] == (7,)
        assert info["n_actions"] == 4
        assert info["action_dim"] == 4
        assert info["env_type"] == "panda"
