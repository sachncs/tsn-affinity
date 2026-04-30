"""Benchmark framework for continual learning evaluation."""
from tsn_affinity.benchmarks.adapters.base import BaseEnvAdapter
from tsn_affinity.benchmarks.metrics import (
    compute_acc,
    compute_bwt,
    compute_forgetting,
    compute_fwt,
)
from tsn_affinity.benchmarks.task_registry import TaskRegistry
from tsn_affinity.benchmarks.task_spec import TaskSpec

__all__ = [
    "BaseEnvAdapter",
    "compute_acc",
    "compute_bwt",
    "compute_forgetting",
    "compute_fwt",
    "TaskRegistry",
    "TaskSpec",
]