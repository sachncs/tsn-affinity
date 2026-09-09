"""Environment adapter protocols.

Adapters wrap gymnasium-style environments so the rest of the package
can discover observation/action spaces without depending on a specific
backend. The package exposes:

    - ``base.BaseEnvAdapter``: ``Protocol`` describing the contract.
    - ``atari.AtariAdapter``: Real ALE environments via gymnasium.
    - ``panda.PandaAdapter``: Placeholder for Panda robotics tasks.

Public attributes are also re-exported from this package's
``__init__`` for convenience.
"""

from __future__ import annotations

from tsn_affinity.benchmarks.adapters.atari import AtariAdapter
from tsn_affinity.benchmarks.adapters.base import BaseEnvAdapter
from tsn_affinity.benchmarks.adapters.panda import PandaAdapter

__all__ = ["AtariAdapter", "BaseEnvAdapter", "PandaAdapter"]
