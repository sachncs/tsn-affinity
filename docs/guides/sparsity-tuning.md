# Tune sparsity

Sparsity is the lever that decides how much capacity each task gets
inside the shared model. This guide explains the knobs and how to
choose values that fit your task distribution.

## The `keep_ratio` parameter

`SparseConfig.keep_ratio` controls the fraction of weights that stay
active per task. The base strategy divides this ratio by the current
**free ratio** (the fraction of weights not yet consolidated into any
previous task):

```text
effective_keep_ratio = keep_ratio / (1 - occupied_ratio)
```

When `effective_keep_ratio` would exceed 1.0 the strategy raises a
`ConfigurationError`. You have three options:

1. **Lower the base `keep_ratio`.** Halving the ratio doubles the
   number of tasks you can fit before saturation.
2. **Switch to a multi-copy strategy** (`TSNAffinityStrategy` or
   `TSNReplayKLStrategy`) so the model allocates a fresh copy when
   capacity runs out.
3. **Increase the model width** by raising `ModelConfig.d_model` so
   each task has more parameters to play with.

## Embeddings

Set `SparseConfig.include_embeddings=False` if your action space has
many tokens (large `n_actions`). TSN's sparse embedding layer slows
down by `O(n_embeddings)` per step, and most continual RL tasks do
not benefit from sparse token tables.

## Quantization

`SparseConfig.quantize_after_task=True` runs k-means quantization over
the newly active weights after each task and stores the codebook in
`strategy.task_codebooks`. This compresses memory but slightly hurts
fine-grained accuracy. Turn it off for debugging and on for large
runs that need to fit in memory.

## Skip modules

`SparseConfig.skip_module_names` is a tuple of module name prefixes
that should never be converted. The default `("dt.te",)` excludes
timestep embeddings, which are shared across tasks and should not be
sparse.

```python
SparseConfig(
    keep_ratio=0.3,
    include_embeddings=False,
    allow_weight_reuse=False,
    skip_module_names=("dt.te", "encoder"),
    quantize_after_task=True,
    quant_clusters=16,
)
```

## Verifying the configuration

A quick sanity check before launching a long run:

```python
from tsn_affinity import RoutingConfig, SparseConfig
from tsn_affinity.strategies.tsn_base import TSNBaseStrategy

config = SparseConfig(keep_ratio=0.3)
strategy = TSNBaseStrategy(
    obs_shape=(4,),
    n_actions=2,
    seq_len=10,
    device="cpu",
    sparse_config=config,
)

# How many tasks fit before saturation?
free = 1.0
tasks = 0
while True:
    try:
        # Pretend we've consumed `free` of the capacity.
        consumed = 1.0 - free
        ratio = config.keep_ratio / free if free > 0 else float("inf")
        if ratio >= 1.0:
            break
        free -= consumed * ratio
        tasks += 1
        if tasks > 50:
            break
    except ZeroDivisionError:
        break

print(f"Approx tasks before saturation: {tasks}")
```

Multi-copy strategies do not have this limitation because they spawn
new copies rather than squeezing more tasks into existing ones.
