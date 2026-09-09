"""Tests for configuration dataclasses."""

from tsn_affinity.core.config import (
    ModelConfig,
    RoutingConfig,
    SparseConfig,
    TSNAffinityConfig,
)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.d_model == 128
        assert cfg.n_layers == 3

    def test_custom_values(self):
        cfg = ModelConfig(d_model=64, n_layers=2)
        assert cfg.d_model == 64
        assert cfg.n_layers == 2


class TestSparseConfig:
    def test_defaults(self):
        cfg = SparseConfig()
        assert cfg.keep_ratio == 0.5

    def test_custom_keep_ratio(self):
        cfg = SparseConfig(keep_ratio=0.3)
        assert cfg.keep_ratio == 0.3


class TestRoutingConfig:
    def test_defaults(self):
        cfg = RoutingConfig()
        assert cfg.mode == "action"


class TestTSNAffinityConfig:
    def test_composition(self):
        cfg = TSNAffinityConfig()
        assert cfg.sparse is not None
        assert cfg.routing is not None
