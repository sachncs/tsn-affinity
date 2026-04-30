"""Benchmark execution and result analysis."""
from tsn_affinity.run.analysis import analyze_run, compute_final_metrics
from tsn_affinity.run.benchmark_runner import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "analyze_run",
    "compute_final_metrics",
]