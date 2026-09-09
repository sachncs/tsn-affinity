# Getting started

This guide walks you through installing TSN-Affinity, verifying the
install, and running your first continual-learning benchmark. By the
end you will have a working environment and an intuition for how the
public API fits together.

## Prerequisites

TSN-Affinity requires **Python 3.10 or later** and supports CPU and
CUDA execution. We test against Python 3.10, 3.11, and 3.12 on Linux
and macOS.

!!! tip "Optional GPU"
    All examples in this guide run on CPU. Set `TORCH_DEVICE=cuda`
    or pass `--device cuda` to the CLIs to use a CUDA-capable GPU
    for training.

## Install

=== "From PyPI"

    ```bash
    pip install tsn-affinity
    ```

=== "With Atari support"

    ```bash
    pip install "tsn-affinity[atari]"
    ```

=== "Editable from source"

    ```bash
    git clone https://github.com/sachncs/tsn-affinity.git
    cd tsn-affinity
    pip install -e ".[dev]"
    ```

The `[dev]` extra pulls in the testing, linting, type-checking, and
documentation toolchain used by the project's own CI.

## Verify the install

```bash
python -c "import tsn_affinity; print(tsn_affinity.__version__)"
```

You should see a version string such as `0.4.0`.

Then run a smoke-test benchmark that takes a few seconds on a laptop:

```bash
tsn-benchmark \
    --strategies tsn_core \
    --n-tasks 3 \
    --trajs-per-task 5 \
    --train-steps 50 \
    --n-runs 1 \
    --output runs/smoke
```

The CLI prints a `final summary` block with the computed `ACC`, `BWT`,
`forgetting`, and `FWT` metrics and writes `runs/smoke/summary.json`.

## Train your first model

The Python API is intentionally small. The minimum code to train a
two-task benchmark is:

```python
from tsn_affinity import (
    RoutingConfig,
    SparseConfig,
    ModelConfig,
    TSNAffinityStrategy,
    Trajectory,
)
from tsn_affinity.data.loaders.batch_loader import make_minibatches
import numpy as np

# 1. Build synthetic trajectories ----------------------------------
def make_traj(seed: int, T: int = 50) -> Trajectory:
    rng = np.random.default_rng(seed)
    obs = rng.normal(0.0, 1.0, (T, 4)).astype(np.float32)
    actions = rng.integers(0, 2, T).astype(np.int64)
    rewards = rng.normal(0.0, 0.1, T).astype(np.float32)
    timesteps = np.arange(T, dtype=np.float32)
    rtg = rewards.copy()
    return Trajectory(obs, actions, rewards, timesteps, rtg)

task_one = [make_traj(seed=1) for _ in range(8)]
task_two = [make_traj(seed=2) for _ in range(8)]

# 2. Configure the strategy ---------------------------------------
strategy = TSNAffinityStrategy(
    obs_shape=(4,),
    n_actions=2,
    seq_len=10,
    device="cpu",
    model_config=ModelConfig(d_model=64, n_layers=2, n_heads=2),
    sparse_config=SparseConfig(keep_ratio=0.3),
    affinity_config=RoutingConfig(mode="action"),
    seed=0,
)

# 3. Train and finalise each task --------------------------------
for trajectories in (task_one, task_two):
    strategy.train_task(trajectories, steps=200, batch_size=16)
    strategy.after_task(trajectories)

# 4. Switch back to task 0 and verify it still works ------------
strategy.set_eval_task(0)
loader = make_minibatches(task_one, seq_len=10, batch_size=16, device="cpu")
obs, actions, rtg, ts, mask = next(loader)
logits = strategy.model(obs, actions, rtg, ts, attention_mask=mask)
print("Diagonal ACC proxy:", float(logits.argmax(-1).eq(actions).float().mean()))
```

The trained strategy now exposes both task-specific masks through
`strategy.task_to_copy` and a frozen consolidation mask that protects
the previously trained weights.

## Where to go next

- [Concepts](guides/concepts.md) — vocabulary used throughout the docs.
- [Routing modes](guides/routing-modes.md) — pick the right mode for
  your task distribution.
- [Atari experiments](guides/atari.md) — collect trajectories and
  reproduce the headline benchmark.
- [API reference](api.md) — every public symbol, auto-generated.

If you hit a snag, see the [troubleshooting guide](operations/troubleshooting.md)
or [open an issue](https://github.com/sachncs/tsn-affinity/issues/new).
