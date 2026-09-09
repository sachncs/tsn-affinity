# Deployment

This page covers deployment patterns for TSN-Affinity, from running
single-machine benchmarks to scaling to multi-GPU training servers.
For day-to-day local use, follow the [getting started guide](getting-started.md).

## Local development

Install the package in editable mode and run the test suite:

```bash
pip install -e ".[dev]"
pytest
ruff check tsn_affinity/ tests/
mypy tsn_affinity/
```

The `Makefile` exposes shortcuts for each step:

```bash
make dev        # Install dev dependencies
make test       # Run the full test suite
make lint       # Run ruff
make format     # Auto-format code
make typecheck  # Run mypy
make docs       # Serve docs locally at http://localhost:8000
```

## Running benchmarks

### Synthetic benchmark

```bash
tsn-benchmark \
    --strategies tsn_core tsn_affinity \
    --n-tasks 5 \
    --trajs-per-task 10 \
    --train-steps 200 \
    --n-runs 3 \
    --output runs/benchmark
```

Outputs land in `runs/benchmark/` with `summary.json`, per-strategy
performance matrices, and per-run metrics.

### Atari benchmark

Requires the Atari extra:

```bash
pip install "tsn-affinity[atari]"
tsn-atari --strategy tsn_affinity --output runs/atari
```

See the [Atari guide](guides/atari.md) for the full workflow
including data collection with `tsn-atari-collect`.

### Panda benchmark

The Panda adapter is a stub. To run a real benchmark install
`panda-gym` and provide your own trajectory pickle:

```bash
pip install panda-gym
tsn-panda --data data/panda_tasks.pkl --output runs/panda
```

The pickle must contain a list of dictionaries with `obs`, `actions`,
and `rewards` arrays. See `tsn_affinity.data.loaders.panda_loader`.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `TORCH_DEVICE` | `cpu` | Default device used by the strategies. |
| `TORCH_ENABLE_FLASH_ATTENTION` | `0` | Set to `1` to enable flash attention on Ampere+ GPUs. |
| `LOG_LEVEL` | `INFO` | Logging verbosity for the `tsn_affinity` logger. |
| `RESULTS_DIR` | `runs/` | Default output directory for CLI runs. |
| `DATA_DIR` | `data/` | Default data directory. |

Copy `.env.example` to `.env` and source it before running the CLIs.

## Docker

A multi-stage `Dockerfile` is provided. Build and run it with:

```bash
docker build -t tsn-affinity:latest .
docker run --rm -it tsn-affinity:latest
```

## Continuous integration

GitHub Actions runs on every push and pull request:

- Lint with `ruff check` and `ruff format --check`.
- Type-check with `mypy`.
- Run the test suite on Python 3.10, 3.11, and 3.12.
- Build and deploy the documentation site.

See `.github/workflows/ci.yml` and `.github/workflows/pages.yml` for
the workflow definitions.

## Release process

1. Bump the version in `pyproject.toml`.
2. Update the `## [Unreleased]` section of `CHANGELOG.md`.
3. Open a pull request, get review, and merge to `master`.
4. Tag the release: `git tag vX.Y.Z && git push origin vX.Y.Z`.
5. The `publish.yml` workflow uploads to PyPI via trusted publishing.
