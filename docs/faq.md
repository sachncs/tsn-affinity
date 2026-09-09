# Frequently asked questions

## General

### What is TSN-Affinity?

TSN-Affinity is a continual offline reinforcement-learning algorithm
that uses sparse subnetwork allocation (TinySubNetworks) and dynamic
task routing to learn multiple tasks sequentially without
catastrophic forgetting.

### How does it differ from standard offline RL?

Standard offline RL trains on a fixed dataset. TSN-Affinity handles
multiple sequential tasks, where each new task builds on knowledge
from previous tasks while maintaining performance on learned tasks.

### What tasks does it support?

- **Synthetic tasks** — fully reproducible, used for CI and quick
  sanity checks.
- **Atari games** (discrete actions) — via the ALE benchmark suite.
- **Panda robotic manipulation** (continuous actions) — via
  PandaReach / PandaPush when `panda-gym` is installed.

## Installation

### Why do I need Python >= 3.10?

The codebase uses modern Python features like `match` statements,
PEP 604 union types (`X | None`), and improved type-hint syntax that
require Python 3.10+.

### Do I need a GPU?

No. TSN-Affinity runs on CPU, but GPU acceleration is recommended
for larger models and benchmarks. Set `TORCH_DEVICE=cuda` or pass
`--device cuda` to the CLIs.

### How do I install Atari support?

```bash
pip install "tsn-affinity[atari]"
```

This pulls in `gymnasium[accept-license-requests]` plus the ALE
bindings.

## Usage

### Which strategy should I use?

| Strategy | Use case |
| --- | --- |
| `tsn_core` | Single-copy baseline, no routing. Good for small task suites. |
| `tsn_affinity` | Action / latent / hybrid routing. Recommended for most multi-task setups. |
| `tsn_replay_kl` | Replay-memory KL routing. Use when you want a fast, model-free similarity signal. |
| `cumulative_replay` | Replay-buffer baseline. Use to compare against naive replay strategies. |
| `naive` | No continual-learning mechanism. Lower bound for catastrophic forgetting. |

### How do I register a new environment?

Implement the `BaseEnvAdapter` protocol and register it with
`TaskRegistry`:

```python
from tsn_affinity.benchmarks import TaskRegistry, TaskSpec

class MyAdapter:
    def is_compatible(self, spec: TaskSpec) -> bool:
        return "myenv" in spec.name

    def create_env(self, spec):
        ...

    def describe(self, env):
        ...

TaskRegistry().register("myenv", MyAdapter())
```

See [Add a new environment](guides/new-environment.md) for the full
recipe.

### What are the key hyperparameters?

| Hyperparameter | Where | Effect |
| --- | --- | --- |
| `keep_ratio` | `SparseConfig` | Fraction of weights to keep per task (0.0–1.0). Lower = sparser. |
| `action_threshold` | `RoutingConfig` | Cross-entropy threshold for action-based routing. |
| `latent_threshold` | `RoutingConfig` | KL divergence threshold for latent-based routing. |
| `kl_threshold` | `RoutingConfig` | KL threshold for replay-memory routing. |
| `d_model` | `ModelConfig` | Transformer embedding dimension (64 for small, 128 for full). |
| `n_layers` | `ModelConfig` | Number of transformer layers. |
| `n_heads` | `ModelConfig` | Number of attention heads. |

## Development

### How do I run tests?

```bash
pytest                              # Run all tests
pytest tests/core/ -v               # Run a specific package
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

Open an issue on
[GitHub](https://github.com/sachncs/tsn-affinity/issues/new?template=bug.yml).

### Where do I get help?

- [GitHub Discussions](https://github.com/sachncs/tsn-affinity/discussions)
- [Documentation](https://sachncs.github.io/tsn-affinity/)
- [Support page](support.md)
