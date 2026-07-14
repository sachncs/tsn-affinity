<p align="center">
  <h1 align="center">TSN-Affinity</h1>
  <p align="center">Training-Aware Sparse Networks with Affinity Routing for Continual Offline Reinforcement Learning.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/tsn-affinity/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/tsn-affinity/ci.yml?branch=master" alt="CI"></a>
    <a href="https://pypi.org/project/tsn-affinity/"><img src="https://img.shields.io/pypi/v/tsn-affinity" alt="PyPI"></a>
    <a href="https://github.com/sachncs/tsn-affinity/stargazers"><img src="https://img.shields.io/github/stars/sachncs/tsn-affinity" alt="Stars"></a>
  </p>
</p>

**Modular re-implementation of Training-Aware Sparse Networks with Affinity Routing for Continual Offline Reinforcement Learning.**

TSN-Affinity combines sparse mask-based subnetwork allocation (TinySubNetworks) with dynamic affinity routing to learn multiple tasks sequentially without catastrophic forgetting. The Decision-Transformer backbone processes trajectories; per-task sparse subnetworks isolate learned weights; the affinity router chooses which copy (and where to spawn new ones) based on action affinity, latent affinity, or hybrid similarity.

## Features

- **Sparse subnetwork allocation** — Each task gets a dedicated sparse subnetwork via learnable weight scores and straight-through estimation
- **Affinity routing** — Dynamic model-copy selection / spawning based on action affinity, latent affinity, or hybrid similarity
- **Frozen weight protection** — Previously learned weights are protected from interference
- **Decision Transformer backbone** — Leverages transformer-based sequence modeling for offline RL
- **Multiple strategies** — `tsn_core`, `tsn_affinity`, and `tsn_replay_kl` for different use cases
- **CLI tools** — Ready-to-use commands for synthetic and Atari benchmarks
- **Google Python Style** — Type hints, docstrings, and ruff linting throughout

## Installation

### From PyPI

```bash
pip install tsn-affinity
pip install "tsn-affinity[atari]"   # gymnasium accept-license-requests
```

### From source

```bash
git clone https://github.com/sachncs/tsn-affinity.git
cd tsn-affinity
pip install -e .
pip install -e ".[dev]"        # dev tooling
pip install -e ".[atari]"      # Atari benchmarks
```

## Quick Start

### CLI

Run a synthetic benchmark to verify everything works:

```bash
tsn-benchmark \
    --strategies tsn_core tsn_affinity \
    --n-tasks 5 \
    --trajs-per-task 10 \
    --train-steps 200 \
    --n-runs 3 \
    --output runs/benchmark

tsn-atari --strategy tsn_affinity --output runs/atari
tsn-panda --strategy tsn_affinity --output runs/panda
```

### Python API

```python
from tsn_affinity.core.config import ModelConfig, SparseConfig, RoutingConfig
from tsn_affinity.strategies import TSNAffinityStrategy

config = ModelConfig(d_model=128, n_layers=3)
sparse_config = SparseConfig(keep_ratio=0.3)
routing_config = RoutingConfig(mode="hybrid")  # 'action' | 'latent' | 'hybrid'

strategy = TSNAffinityStrategy(
    model_config=config,
    sparse_config=sparse_config,
    routing_config=routing_config,
)
strategy.train(trajectories, n_tasks=5)
```

## Available Strategies

| Strategy | Description |
|----------|-------------|
| `tsn_affinity` | Full TSN-Affinity with action / latent / hybrid routing |
| `tsn_core` | Single-copy TSN baseline (no routing) |
| `tsn_replay_kl` | TSN with replay-memory KL routing |

## Configuration

Configuration is managed through dataclasses. See `configs/` for examples:

| File | Purpose |
|------|---------|
| `configs/base.py` | Default configurations |
| `configs/development.py` | Small model for development |
| `configs/production.py` | Full-size model |

Environment variables are documented in [`.env.example`](.env.example).

| Setting | Mechanism | Default | Description |
|---------|-----------|---------|-------------|
| Routing mode | `RoutingConfig.mode` | `"hybrid"` | `action` / `latent` / `hybrid` |
| Sparsity | `SparseConfig.keep_ratio` | `1.0` | Fraction of weights kept per task |
| Model size | `ModelConfig.d_model`, `n_layers` | `128`, `3` | Transformer backbone |

## API

| Symbol | Type | Description |
|--------|------|-------------|
| `tsn_affinity.strategies.TSNAffinityStrategy` | class | Full TSN-Affinity strategy (action / latent / hybrid routing) |
| `tsn_affinity.strategies.TSNCoreStrategy` | class | Single-copy TSN baseline (no routing) |
| `tsn_affinity.strategies.TSNReplayKLStrategy` | class | TSN with replay-memory KL routing |
| `tsn_affinity.core.config.ModelConfig` | dataclass | Transformer / model hyperparameters |
| `tsn_affinity.core.config.SparseConfig` | dataclass | Sparsity / masking |
| `tsn_affinity.core.config.RoutingConfig` | dataclass | Affinity routing mode |
| `tsn_affinity.core.DecisionTransformer` | class | Backbone sequence model |
| `tsn_affinity.sparse` | module | TSN layers (Linear, Conv2d, Embedding, TopK STE) |
| `tsn_affinity.routing` | module | Affinity metrics, router, warm-starter |
| `tsn_affinity.cli.benchmark` | CLI | `tsn-benchmark` |
| `tsn_affinity.cli.atari` | CLI | `tsn-atari` |
| `tsn_affinity.cli.panda` | CLI | `tsn-panda` |

## Examples

Affinity routing modes:

```python
action_only = RoutingConfig(mode="action")      # choose copy by action-similarity
latent_only = RoutingConfig(mode="latent")      # choose copy by latent-similarity
hybrid      = RoutingConfig(mode="hybrid")      # blend action + latent similarity
```

Sparsity sweep via `SparseConfig`:

```python
SparseConfig(keep_ratio=0.1)   # very sparse — strongest capacity isolation
SparseConfig(keep_ratio=0.3)   # default
SparseConfig(keep_ratio=0.5)   # more capacity per copy
```

## Benchmark Results

Synthetic benchmark (5 tasks, 10 trajs/task, 200 train steps, 3 runs):

| Strategy | ACC | BWT | Forgetting | FWT | Time/Task |
|----------|-----|-----|------------|-----|-----------|
| `tsn_core` | 0.5388 ± 0.0028 | -0.0012 ± 0.0015 | 0.0028 ± 0.0007 | -0.0012 ± 0.0015 | 70.58s ± 3.74s |
| `tsn_affinity` | 0.4017 ± 0.0041 | -0.0008 ± 0.0016 | 0.0026 ± 0.0016 | -0.0008 ± 0.0016 | 106.35s ± 4.18s |

## Project Structure

```
tsn_affinity/
├── core/           # DecisionTransformer, attention, encoder, config
├── sparse/         # TSN layers (Linear, Conv2d, Embedding, TopK STE)
├── routing/        # Affinity metrics, router, warmstarter
├── strategies/     # Training strategies and copy management
├── data/           # Trajectory handling, batch generation
├── interfaces/     # Abstract protocols and type definitions
├── services/       # Training orchestration service
├── run/            # Benchmark runner and analysis utilities
├── cli/            # Command-line entry points (benchmark, atari, panda)
└── configs/        # base / development / production dataclass configs
```

## Development

```bash
pip install -e ".[dev]"
pytest --cov=tsn_affinity --cov-report=term-missing
ruff check .
ruff format .
mypy tsn_affinity/
```

## Testing

```bash
pytest                                     # full suite
pytest -m "not slow"                        # skip slow tests
pytest -m gpu                               # CUDA-only tests
pytest --cov=tsn_affinity --cov-fail-under=70   # coverage gate (70%)
```

## Build

```bash
pip install build
python -m build
```

## Release

```bash
pytest && ruff check . && mypy tsn_affinity/
git tag vX.Y.Z && git push origin vX.Y.Z
# .github/workflows/publish.yml publishes to PyPI via trusted publishing
```

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python ≥ 3.10 |
| ML framework | PyTorch ≥ 2.0 |
| RL environment | Gymnasium |
| Numerical | NumPy, scikit-learn |
| Linting | ruff |
| Type checking | mypy |
| Testing | pytest + pytest-cov |
| Hooks | pre-commit |
| Docs | mkdocs + mkdocstrings |

## Roadmap

- **v0.3.x** — Current: sparse subnetwork allocation, affinity routing (action / latent / hybrid), Decision Transformer backbone, synthetic + Atari + Panda benchmarks.
- **v0.4.0** — Planned: adaptive routing thresholds via meta-learning, dynamic task-similarity metrics with learned weights, memory-efficient affinity computation with caching.
- **v0.5.0** — Planned: GPU-optimized batched evaluation, expert trajectory support (DQN replay buffer), continuous-control benchmarks (DMControl suite).
- **v1.0.0** — Planned: transformer-based encoders (ViT replacement for CNN).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).

## Security

Report vulnerabilities to **sachncs@gmail.com** — see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2026 Sachin

## Acknowledgments

Based on the TSN-Affinity paper for continual offline reinforcement learning.
