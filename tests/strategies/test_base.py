"""Tests for base strategy interface."""

import pytest

from tsn_affinity.strategies.base import BaseStrategy


class TestBaseStrategy:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseStrategy(obs_shape=(4,), n_actions=2)
