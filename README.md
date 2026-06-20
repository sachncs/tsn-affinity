# TSN-Affinity

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/sachn-cs/tsn-affinity/actions/workflows/ci.yml/badge.svg)](https://github.com/sachn-cs/tsn-affinity/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)

A modular re-implementation of **Training-Aware Sparse Networks with Affinity Routing** for Continual Offline Reinforcement Learning.

TSN-Affinity combines sparse mask-based subnetwork allocation (TinySubNetworks) with dynamic affinity routing to learn multiple tasks sequentially without catastrophic forgetting.

## Features

- **Sparse Subnetwork Allocation**: Each task gets a dedicated sparse subnetwork via learnable weight scores and straight-through estimation
- **Affinity Routing**: Dynamic model copy selection/spawning based on action affinity, latent affinity, or hybrid similarity
- **Frozen Weight Protection**: Previously learned weights are protected from interference
- **Decision Transformer Backbone**: Leverages transformer-based sequence modeling for offline RL
- **Multiple Strategies**: `tsn_core`, `tsn_affinity`, and `tsn_replay_kl` for different use cases
- **CLI Tools**: Ready-to-use commands for synthetic and Atari benchmarks
- **Google Python Style**: Type hints, docstrings, and linting throughout

## Installation

```bash
git clone https://github.com/sachn-cs/tsn-affinity.git
cd tsn-affinity
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

For Atari benchmarks:

```bash
pip install -e ".[atari]"
```

## Quick Start

Run a synthetic benchmark to verify everything works:

```bash
tsn-benchmark \
    --strategies tsn_core tsn_affinity \
    --n-tasks 5 \
    --trajs-per-task 10 \
    --train-steps 200 \
    --n-runs 3 \
    --output runs/benchmark
```

## Usage

### Available Strategies

| Strategy | Description |
|----------|-------------|
| `tsn_affinity` | Full TSN-Affinity with action/latent/hybrid routing |
| `tsn_core` | Single-copy TSN baseline (no routing) |
| `tsn_replay_kl` | TSN with replay-memory KL routing |

### CLI Commands

```bash
# Synthetic benchmark
tsn-benchmark --strategies tsn_core tsn_affinity --n-tasks 5

# Atari benchmark
tsn-atari --strategy tsn_affinity --output runs/atari

# Panda benchmark
tsn-panda --strategy tsn_affinity --output runs/panda
```

### Python API

```python
from tsn_affinity.core.config import ModelConfig, SparseConfig, RoutingConfig
from tsn_affinity.strategies import TSNAffinityStrategy

config = ModelConfig(d_model=128, n_layers=3)
sparse_config = SparseConfig(keep_ratio=0.3)
routing_config = RoutingConfig(mode="hybrid")

strategy = TSNAffinityStrategy(
    model_config=config,
    sparse_config=sparse_config,
    routing_config=routing_config,
)
```

## Configuration

Configuration is managed through dataclasses. See `configs/` for examples:

- `configs/base.py` - Default configurations
- `configs/development.py` - Small model for development
- `configs/production.py` - Full-size model

Environment variables are documented in [`.env.example`](.env.example).

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
└── cli/            # Command-line entry points
```

## Benchmark Results

### Synthetic Benchmark (5 tasks, 10 trajs/task, 200 train steps, 3 runs)

| Strategy | ACC | BWT | Forgetting | FWT | Time/Task |
|----------|-----|-----|------------|-----|-----------|
| tsn_core | 0.5388 ± 0.0028 | -0.0012 ± 0.0015 | 0.0028 ± 0.0007 | -0.0012 ± 0.0015 | 70.58s ± 3.74s |
| tsn_affinity | 0.4017 ± 0.0041 | -0.0008 ± 0.0016 | 0.0026 ± 0.0016 | -0.0008 ± 0.0016 | 106.35s ± 4.18s |

## Tech Stack

- **Language**: Python >= 3.10
- **ML Framework**: PyTorch >= 2.0
- **RL Environment**: Gymnasium
- **Numerical**: NumPy, scikit-learn
- **Linting**: ruff
- **Type Checking**: mypy
- **Testing**: pytest + pytest-cov

## Roadmap

- [ ] Adaptive routing thresholds via meta-learning
- [ ] Dynamic task similarity metrics with learned weights
- [ ] Memory-efficient affinity computation with caching
- [ ] GPU-optimized batched evaluation
- [ ] Expert trajectory support (DQN replay buffer)
- [ ] Continuous control benchmarks (DMControl suite)
- [ ] Transformer-based encoders (ViT replacement for CNN)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Security

For security vulnerabilities, please see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Based on the TSN-Affinity paper for continual offline reinforcement learning.
