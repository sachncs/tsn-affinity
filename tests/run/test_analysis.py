"""Tests for run analysis utilities."""

import json

import numpy as np
import pytest

from tsn_affinity.run.analysis import analyze_run, compute_final_metrics


class TestComputeFinalMetrics:
    def test_computes_all_metrics(self):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        metrics = compute_final_metrics(matrix)
        assert "acc" in metrics
        assert "bwt" in metrics
        assert "forgetting" in metrics
        assert "fwt" in metrics


class TestAnalyzeRun:
    def test_analyzes_npy_file(self, tmp_path):
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        np.save(tmp_path / "performance_matrix.npy", matrix)
        metrics = analyze_run(str(tmp_path))
        assert "acc" in metrics

    def test_analyzes_json_file(self, tmp_path):
        matrix = [[1.0, 0.5], [0.5, 1.0]]
        with open(tmp_path / "results.json", "w") as f:
            json.dump({"performance_matrix": matrix}, f)
        metrics = analyze_run(str(tmp_path))
        assert "acc" in metrics

    def test_missing_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            analyze_run(str(tmp_path))
