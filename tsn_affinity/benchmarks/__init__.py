"""Benchmark adapters, registries, and metrics for TSN-Affinity.

This package groups environment adapters (Atari, Panda, synthetic), the
shared task registry, and the continual-learning metrics that downstream
analysis relies on. The package was previously referenced from the CLI,
``tsn_affinity.run.analysis``, and the test suite but its source files
were missing from the published wheel.

Public API:
    - ``adapters``: ``AtariAdapter``, ``PandaAdapter``, ``BaseEnvAdapter``.
    - ``metrics``: ``compute_acc``, ``compute_bwt``, ``compute_forgetting``,
      ``compute_fwt``, ``StandardCLMetrics``.
    - ``registry``: ``TaskRegistry``.
    - ``spec``: ``TaskSpec``.
    - ``atari_baselines``: ``ATARI_GAMES``, ``ATARI_BASELINES``,
      ``get_atari_baselines``.
"""

from tsn_affinity.benchmarks import adapters, atari_baselines, metrics, registry, spec
from tsn_affinity.benchmarks.adapters import (
    AtariAdapter,
    BaseEnvAdapter,
    PandaAdapter,
)
from tsn_affinity.benchmarks.atari_baselines import (
    ATARI_BASELINES,
    ATARI_GAMES,
    AtariBaselines,
    get_atari_baselines,
)
from tsn_affinity.benchmarks.metrics import (
    StandardCLMetrics,
    compute_acc,
    compute_bwt,
    compute_forgetting,
    compute_fwt,
)
from tsn_affinity.benchmarks.registry import TaskRegistry
from tsn_affinity.benchmarks.spec import TaskSpec

__all__ = [
    "adapters",
    "atari_baselines",
    "metrics",
    "registry",
    "spec",
    "AtariAdapter",
    "AtariBaselines",
    "BaseEnvAdapter",
    "PandaAdapter",
    "StandardCLMetrics",
    "ATARI_BASELINES",
    "ATARI_GAMES",
    "compute_acc",
    "compute_bwt",
    "compute_forgetting",
    "compute_fwt",
    "get_atari_baselines",
    "TaskRegistry",
    "TaskSpec",
]
