# FAQ

## General

### What is TSN-Affinity?

TSN-Affinity is a continual offline reinforcement learning algorithm that uses sparse subnetwork allocation (TinySubNetworks) and dynamic task routing to learn multiple tasks sequentially without catastrophic forgetting.

### How does it differ from standard offline RL?

Standard offline RL trains on a fixed dataset. TSN-Affinity handles multiple sequential tasks, where each new task builds on knowledge from previous tasks while maintaining performance on learned tasks.

### What tasks does it support?

- **Atari games** (discrete actions): Via the ALE benchmark suite
- **Panda robotic manipulation** (continuous actions): Via PandaPush/PandaReach
- **Synthetic tasks**: For controlled experimentation

## Installation

### Why do I need Python >= 3.10?

The codebase uses modern Python features like `match` statements and enhanced type hints that require Python 3.10+.

### Do I need a GPU?

No. TSN-Affinity runs on CPU, but GPU acceleration is recommended for larger models and benchmarks. Install with `pip install -e .` for CPU-only usage.

### How do I install Atari support?

```bash
pip install -e ".[atari]"
```

This installs the `gymnasium[accept-license-requests]` extra needed for ALE environments.

## Usage

### Which strategy should I use?

| Strategy | Use Case |
|----------|----------|
| `tsn_core` | Baseline comparisons, single-task evaluation |
| `tsn_affinity` | Production use, diverse task sets |
| `tsn_replay_kl` | When you have replay buffers |

### How do I add new tasks?

Register tasks via the `TaskRegistry`:

```python
from tsn_affinity.core.config import ModelConfig
from tsn_affinity.strategies import TSNAffinityStrategy

# Register a custom task
registry.register("my_task", task_spec)
```

### What are the key hyperparameters?

- `keep_ratio`: Fraction of weights to keep (0.0-1.0). Lower = sparser.
- `action_threshold`: Affinity threshold for action-based routing.
- `latent_threshold`: Affinity threshold for latent-based routing.
- `d_model`: Model dimension (64 for small, 128 for full).
- `n_layers`: Number of transformer layers.

## Development

### How do I run tests?

```bash
pytest                              # Run all tests
pytest tests/core/ -v               # Run specific module
pytest -m "not slow"                # Skip slow tests
pytest --cov=tsn_affinity           # With coverage
```

### How do I lint and format?

```bash
ruff check tsn_affinity/ tests/     # Lint
ruff format tsn_affinity/ tests/    # Format
mypy tsn_affinity/                  # Type check
```

### Where do I report bugs?

Open an issue on [GitHub](https://github.com/sachncs/tsn-affinity/issues) using the bug report template.
