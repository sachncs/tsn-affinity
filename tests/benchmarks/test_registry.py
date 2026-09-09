"""Tests for TaskRegistry."""

import pytest

from tsn_affinity.benchmarks.adapters.atari import AtariAdapter
from tsn_affinity.benchmarks.registry import TaskRegistry


class TestTaskRegistry:
    def test_singleton(self):
        r1 = TaskRegistry()
        r2 = TaskRegistry()
        assert r1 is r2

    def test_register_and_get(self):
        registry = TaskRegistry()
        adapter = AtariAdapter()
        registry.register("atari", adapter)
        assert registry.get("atari") is adapter

    def test_get_missing_raises(self):
        registry = TaskRegistry()
        with pytest.raises(KeyError):
            registry.get("missing")

    def test_list_adapters(self):
        registry = TaskRegistry()
        adapter = AtariAdapter()
        registry.register("atari", adapter)
        adapters = registry.list_adapters()
        assert "atari" in adapters
