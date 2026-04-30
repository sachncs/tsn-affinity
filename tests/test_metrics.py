"""Tests for metrics module."""

import numpy as np
import pytest

from tsn_affinity.benchmarks.metrics import (
    compute_acc,
    compute_bwt,
    compute_forgetting,
    compute_fwt,
    StandardCLMetrics,
)


class TestComputeAcc:
    def test_perfect_performance(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        acc = compute_acc(matrix)
        assert abs(acc - 1.0) < 1e-6

    def test_partial_performance(self):
        matrix = np.array([[0.8, 0.3], [0.4, 0.9]])
        acc = compute_acc(matrix)
        expected = (0.8 + 0.9) / 2
        assert abs(acc - expected) < 1e-6


class TestComputeBWT:
    def test_negative_when_performance_drops(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        bwt = compute_bwt(matrix)
        assert bwt < 0.0

    def test_positive_forgetting(self):
        matrix = np.array([[1.0, 0.5], [0.3, 0.9]])
        bwt = compute_bwt(matrix)
        assert bwt < 0.0


class TestComputeForgetting:
    def test_zero_forgetting_when_never_drops(self):
        matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
        f = compute_forgetting(matrix)
        assert abs(f - 0.0) < 1e-6

    def test_has_forgetting(self):
        matrix = np.array([[1.0, 0.5], [0.3, 0.9]])
        f = compute_forgetting(matrix)
        assert f > 0.0


class TestStandardCLMetrics:
    def test_all_metrics(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        metrics = StandardCLMetrics(matrix)
        results = metrics.compute_all()
        assert "acc" in results
        assert "bwt" in results
        assert "forgetting" in results
        assert "fwt" in results

    def test_summary(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        metrics = StandardCLMetrics(matrix)
        summary = metrics.summary()
        assert "ACC" in summary
        assert "BWT" in summary