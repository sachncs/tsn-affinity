# Getting Started

This guide walks you through installing TSN-Affinity and running your first continual learning benchmark.

## Prerequisites

- Python >= 3.10
- pip
- (Optional) NVIDIA GPU with CUDA support for GPU acceleration

## Installation

### 1. Clone and install

```bash
git clone https://github.com/sachn-cs/tsn-affinity.git
cd tsn-affinity

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install core package
pip install -e .
```

### 2. Install optional dependencies

```bash
# For development (testing, linting, type checking)
pip install -e ".[dev]"

# For Atari benchmarks
pip install -e ".[atari]"
```

### 3. Verify installation

```bash
python -c "import tsn_affinity; print(tsn_affinity.__version__)"
pytest
```

## First Benchmark

Run the synthetic benchmark to verify everything works:

```bash
tsn-benchmark \
    --strategies tsn_core tsn_affinity \
    --n-tasks 5 \
    --trajs-per-task 10 \
    --train-steps 200 \
    --n-runs 1 \
    --output runs/quick-test
```

## Next Steps

- Read the [Architecture Guide](architecture.md) to understand how TSN-Affinity works
- Check the [Deployment Guide](deployment.md) for production setup
- See [FAQ](faq.md) for common questions
