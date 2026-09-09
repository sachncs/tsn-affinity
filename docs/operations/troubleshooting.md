# Troubleshooting

This page lists the most common errors you may encounter and how to
fix them. If your problem is not listed, open an issue with the full
traceback and the output of `python -c "import tsn_affinity; print(tsn_affinity.__version__)"`.

## Installation

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: tsn_affinity` | Installed in a different virtualenv. | Run `which python && python -m pip list` to verify. |
| `ImportError: gymnasium` | Atari extra not installed. | `pip install 'tsn-affinity[atari]'`. |
| `pip` cannot find a compatible wheel | Python version older than 3.10. | Upgrade to Python 3.10 or later. |

## Runtime

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `RuntimeError: Frozen gradient zeroing failed` | A new sparse layer was added but consolidated masks were not refreshed. | Call `strategy._sync_occupied_masks_into_modules()` after adding the layer, then retry. |
| `StrategyError: Failed to prepare the current task because mask capacity was exhausted` | Single-copy strategy ran out of capacity. | Lower `keep_ratio`, switch to `TSNAffinityStrategy`, or raise `d_model`. |
| `RoutingError: Non-finite affinity score` | Numerical instability during routing. | Lower `routing_n_batches` or `routing_batch_size`, or enable gradient clipping. |
| `CUDA out of memory` | Sequence length too long for the available memory. | Lower `seq_len`, reduce `--batch-size`, or enable mixed precision. |

## Training quality

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Loss becomes NaN after a few hundred steps | Learning rate too high. | Lower `lr` in `ModelConfig` (try `1e-4` instead of `3e-4`). |
| Previous-task scores drop after every new task | `keep_ratio` too high, mask capacity exhausted. | Switch to a multi-copy strategy or lower `keep_ratio`. |
| `tsn_affinity` ACC lower than `tsn_core` | Routing overhead without commensurate benefit on small task suites. | Increase the number of tasks, increase model width, or use `hybrid` routing. |
| Atari scores close to random | Insufficient training or trajectories. | Increase `--train-steps` and `--n-trajectories`. |

## Benchmarking

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: ale_py` | ALE bindings missing. | `pip install ale-py`. |
| NaN values in `performance_matrix.npy` | One of the greedy rollouts diverged. | Lower `--max-steps`, increase `--eval-rollouts`, or set `deterministic=True` (already the default). |
| Saved matrices look like all ones | The CLI shortcut that produced them bypassed real evaluation. | This was a known bug in versions 0.3.0 and earlier; the runs/ directory no longer ships placeholder results. |

## Tests

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ImportError` on `tsn_affinity.benchmarks.metrics` | Older version missing the benchmarks package. | Upgrade to 0.4.0 or later. |
| `Failed: Required test coverage of 70%` | New code not covered by tests. | Add tests under `tests/` mirroring the package layout. |
| `pytest` hangs on first run | Cache from a previous install interfering. | Delete `.pytest_cache/` and re-run. |

## Where to ask

- Open an issue:
  [github.com/sachncs/tsn-affinity/issues](https://github.com/sachncs/tsn-affinity/issues)
- Read the [FAQ](../faq.md) for conceptual questions.
- See [Contributing](../contributing.md) for development setup.
