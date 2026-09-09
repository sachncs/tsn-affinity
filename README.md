<p align="center">
  <h1 align="center">TSN-Affinity</h1>
  <p align="center">Continual offline reinforcement learning with training-aware sparse networks and affinity routing.</p>
  <p align="center">
    <a href="https://pypi.org/project/tsn-affinity/"><img src="https://img.shields.io/pypi/v/tsn-affinity" alt="PyPI"></a>
    <a href="https://github.com/sachncs/tsn-affinity/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/tsn-affinity/ci.yml?branch=master" alt="CI"></a>
    <a href="https://sachncs.github.io/tsn-affinity/"><img src="https://img.shields.io/badge/docs-tsn--affinity-blue" alt="Docs"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
    <a href="https://github.com/sachncs/tsn-affinity/stargazers"><img src="https://img.shields.io/github/stars/sachncs/tsn-affinity" alt="Stars"></a>
  </p>
</p>

**TSN-Affinity** is a modular PyTorch library for continual offline
reinforcement learning. It combines sparse subnetwork allocation
(TinySubNetworks) with affinity routing to learn multiple tasks
sequentially without catastrophic forgetting.

## Highlights

- **Sparse subnetwork allocation** — every task gets a dedicated
  sparse subnetwork via learnable weight scores and straight-through
  estimation.
- **Affinity routing** — dynamic model-copy selection / spawning
  based on action affinity, latent affinity, or hybrid similarity.
- **Decision Transformer backbone** — leverages transformer-based
  sequence modeling for offline RL.
- **Multiple strategies** — `tsn_core`, `tsn_affinity`, `tsn_replay_kl`,
  `cumulative_replay`, and `naive` for ablation studies.
- **CLI tooling** — `tsn-benchmark`, `tsn-atari`, `tsn-atari-collect`,
  and `tsn-panda` for reproducible experiments.
- **Google Python Style** — type hints, docstrings, and ruff linting
  throughout.

## Quick start

### Install

```bash
pip install tsn-affinity
pip install "tsn-affinity[atari]"     # optional: gymnasium + ALE
pip install "tsn-affinity[dev]"       # optional: testing + docs
```

### Run a benchmark

```bash
tsn-benchmark \
    --strategies tsn_core tsn_affinity \
    --n-tasks 5 \
    --trajs-per-task 10 \
    --train-steps 200 \
    --n-runs 3 \
    --output runs/benchmark
```

### Train your own model

```python
from tsn_affinity import (
    RoutingConfig,
    SparseConfig,
    ModelConfig,
    TSNAffinityStrategy,
)

strategy = TSNAffinityStrategy(
    obs_shape=(4,),
    n_actions=2,
    seq_len=10,
    device="cpu",
    model_config=ModelConfig(d_model=64, n_layers=2, n_heads=2),
    sparse_config=SparseConfig(keep_ratio=0.3),
    affinity_config=RoutingConfig(mode="hybrid"),
    seed=0,
)

for trajectories in (task_one, task_two):
    strategy.train_task(trajectories, steps=200, batch_size=16)
    strategy.after_task(trajectories)
```

## Documentation

The full documentation lives at
[sachncs.github.io/tsn-affinity](https://sachncs.github.io/tsn-affinity/):

- [Getting started](https://sachncs.github.io/tsn-affinity/getting-started/)
- [How-to guides](https://sachncs.github.io/tsn-affinity/guides/)
- [Architecture overview](https://sachncs.github.io/tsn-affinity/architecture/)
- [API reference](https://sachncs.github.io/tsn-affinity/api/)
- [Operations & deployment](https://sachncs.github.io/tsn-affinity/deployment/)
- [FAQ](https://sachncs.github.io/tsn-affinity/faq/)

## Project structure

```text
tsn_affinity/
├── core/           # Decision Transformer, attention, encoder, configs
├── sparse/         # TSN layers (Linear, Conv2d, Embedding, TopK STE)
├── routing/        # Affinity metrics, router, warmstarter
├── strategies/     # Continual learning strategies + copy manager
├── data/           # Trajectory handling, batch generation, Panda utilities
├── interfaces/     # Abstract protocols
├── services/       # Training orchestration service
├── run/            # Result analysis utilities
├── benchmarks/     # Adapters, registry, metrics, Atari baselines
└── cli/            # Command-line entry points
```

## Available strategies

| Strategy | Description |
| --- | --- |
| `tsn_affinity` | Full TSN-Affinity with action / latent / hybrid routing. |
| `tsn_core` | Single-copy TSN baseline (no routing). |
| `tsn_replay_kl` | TSN with replay-memory KL routing. |
| `cumulative_replay` | Cumulative replay-buffer baseline. |
| `naive` | Plain Decision Transformer with no continual-learning mechanism. |

## Configuration

Configuration is managed through dataclasses. See `configs/` for
examples:

| File | Purpose |
| --- | --- |
| `configs/base.py` | Default configurations. |
| `configs/development.py` | Small model for development. |
| `configs/production.py` | Full-size model. |

Environment variables are documented in [`.env.example`](.env.example).

| Setting | Default | Description |
| --- | --- | --- |
| Routing mode | `action` | `action` / `latent` / `hybrid` / `replay_kl` |
| Sparsity (`keep_ratio`) | `0.5` | Fraction of weights kept per task |
| Model size | `d_model=128, n_layers=3` | Transformer backbone |

## API at a glance

| Symbol | Description |
| --- | --- |
| `tsn_affinity.strategies.TSNAffinityStrategy` | Full TSN-Affinity strategy. |
| `tsn_affinity.strategies.TSNCoreStrategy` | Single-copy baseline. |
| `tsn_affinity.strategies.TSNReplayKLStrategy` | Replay-KL strategy. |
| `tsn_affinity.core.DecisionTransformer` | Backbone sequence model. |
| `tsn_affinity.sparse` | TSN layers (Linear, Conv2d, Embedding, TopK STE). |
| `tsn_affinity.routing` | Affinity metrics, router, warm-starter. |
| `tsn_affinity.cli.benchmark` | `tsn-benchmark` CLI. |
| `tsn_affinity.cli.atari` | `tsn-atari` CLI. |
| `tsn_affinity.cli.panda` | `tsn-panda` CLI. |

## Benchmark results

The headline numbers below come from the synthetic benchmark bundled
with the repository (`tsn-benchmark` with 5 tasks, 10 trajs per task,
200 train steps, 3 runs):

| Strategy | ACC | BWT | Forgetting | FWT | Time/Task |
| --- | --- | --- | --- | --- | --- |
| `tsn_core` | 0.5388 ± 0.0028 | -0.0012 ± 0.0015 | 0.0028 ± 0.0007 | -0.0012 ± 0.0015 | 70.58s ± 3.74s |
| `tsn_affinity` | 0.4017 ± 0.0041 | -0.0008 ± 0.0016 | 0.0026 ± 0.0016 | -0.0008 ± 0.0016 | 106.35s ± 4.18s |

!!! note "Interpreting the headline table"
    `tsn_core` ACC is higher than `tsn_affinity` ACC on the synthetic
    benchmark because the small task suite does not benefit from the
    affinity router's overhead. `tsn_affinity` shines on longer task
    sequences and visually distinct environments where the router's
    copy-reuse meaningfully amortises training. See the
    [routing modes guide](https://sachncs.github.io/tsn-affinity/guides/routing-modes/)
    for guidance on picking the right strategy.

To regenerate the numbers above:

```bash
tsn-benchmark \
    --strategies tsn_core tsn_affinity \
    --n-tasks 5 \
    --trajs-per-task 10 \
    --train-steps 200 \
    --n-runs 3 \
    --output runs/benchmark
```

The CLI writes `runs/benchmark/summary.json` with full metrics and
standard deviations.

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

## Tech stack

| Category | Technology |
| --- | --- |
| Language | Python ≥ 3.10 |
| ML framework | PyTorch ≥ 2.0 |
| RL environment | Gymnasium |
| Numerical | NumPy, scikit-learn |
| Linting | ruff |
| Type checking | mypy |
| Testing | pytest + pytest-cov |
| Hooks | pre-commit |
| Docs | mkdocs + mkdocstrings + mkdocs-material |

## Roadmap

- **0.4.x** — Site redesign, unified routing config, restored
  benchmarks package, expanded documentation.
- **0.5.x** — Adaptive routing thresholds via meta-learning, dynamic
  task-similarity metrics with learned weights, memory-efficient
  affinity computation with caching.
- **0.6.x** — GPU-optimised batched evaluation, expert trajectory
  support (DQN replay buffer), continuous-control benchmarks
  (DMControl suite).
- **1.0.0** — Transformer-based encoders (ViT replacement for CNN).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Code of conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).

## Security

Report vulnerabilities to **sachncs@gmail.com** — see
[SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2026 Sachin.

## Acknowledgments

Based on the TSN-Affinity paper for continual offline reinforcement
learning.
