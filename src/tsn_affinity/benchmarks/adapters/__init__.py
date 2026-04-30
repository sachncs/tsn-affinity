"""Environment adapters for different task families."""
from tsn_affinity.benchmarks.adapters.atari_adapter import AtariAdapter
from tsn_affinity.benchmarks.adapters.panda_adapter import PandaAdapter

__all__ = ["AtariAdapter", "PandaAdapter"]