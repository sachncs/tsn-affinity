# Deployment

## Local Development

```bash
pip install -e ".[dev]"
scripts/setup.sh
```

## Running Benchmarks

### Synthetic Benchmark

```bash
tsn-benchmark \
    --strategies tsn_core tsn_affinity \
    --n-tasks 5 \
    --trajs-per-task 10 \
    --train-steps 200 \
    --n-runs 3 \
    --output runs/benchmark
```

### Atari Benchmark

```bash
tsn-atari \
    --strategy tsn_affinity \
    --output runs/atari \
    --n-trajectories 10 \
    --max-steps 2000 \
    --train-steps 2000
```

## Output Structure

Results are saved to the specified output directory:

```
runs/benchmark/
├── results.json           # Per-strategy aggregate metrics
├── <strategy>_<run>/      # Individual run results
│   ├── results.json
│   └── performance_matrix.npy
└── ...
```

## Environment Variables

See [`.env.example`](../.env.example) for all configurable options.

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCH_DEVICE` | `cpu` | Training device |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `RESULTS_DIR` | `runs/` | Output directory |

## GPU Configuration

For GPU acceleration:

```bash
export TORCH_DEVICE=cuda
# or for specific GPU:
export TORCH_DEVICE=cuda:0
```

## CI/CD

The project includes GitHub Actions workflows for:

- **Linting**: `ruff check` and `ruff format --check`
- **Type checking**: `mypy`
- **Testing**: `pytest` across Python 3.10, 3.11, 3.12

See `.github/workflows/ci.yml` for details.
