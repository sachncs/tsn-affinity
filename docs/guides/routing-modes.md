# Choose a routing mode

The affinity router decides whether to spawn a new model copy for an
incoming task or to reuse an existing one. TSN-Affinity ships three
modes; pick the one that best matches the structure of your task
sequence.

## `action` (cross-entropy)

Action affinity measures the average cross-entropy between each
existing copy's predicted actions and the demonstrated actions of
the new task. Use it when tasks are best distinguished by their
action distributions, for example discrete-control environments like
Atari or CartPole.

```python
from tsn_affinity import RoutingConfig

RoutingConfig(
    mode="action",
    action_threshold=12.0,
    routing_n_batches=4,
    routing_batch_size=64,
)
```

A higher `action_threshold` makes the router more tolerant of
divergent actions before spawning a new copy. Lower values are more
aggressive about creating copies.

## `latent` (KL divergence)

Latent affinity compares the diagonal Gaussian fits of observation
latents across copies. Use it when the action distribution is shared
but the visual or proprioceptive distribution differs — for example
when the same policy is evaluated on visually distinct environments.

```python
RoutingConfig(
    mode="latent",
    latent_threshold=25.0,
)
```

## `hybrid`

Hybrid mode weights the action and latent metrics. It is the safest
default when you are unsure which signal is more informative.

```python
RoutingConfig(
    mode="hybrid",
    hybrid_alpha=0.7,            # weight for action component
    normalize_scores=True,       # min-max normalize before mixing
    routing_n_batches=4,
)
```

`hybrid_alpha` close to 1.0 leans on action affinity; close to 0.0
leans on latent affinity.

## `replay_kl`

`replay_kl` is the routing mode used by `TSNReplayKLStrategy`. It
matches tasks by the KL divergence between observation replay
memories and does not need a learned model. Pick it when you want a
fast, model-free similarity signal.

```python
RoutingConfig(
    mode="replay_kl",
    kl_threshold=0.25,
    memory_size=256,
    max_copies=8,
)
```

## Absolute vs relative thresholds

`RoutingConfig.relative_threshold=True` switches the copy-creation
heuristic from "best score > absolute threshold" to "best score
sufficiently close to the second-best score". Use the relative form
when your task distribution is heterogeneous and a single threshold
cannot capture both similar and dissimilar pairs.

## Maximum model copies

`RoutingConfig.max_model_copies` caps the total number of copies the
strategy may spawn. When the cap is reached, the router falls back to
the most similar existing copy and never raises. Use it to bound
memory when scaling to very long task sequences.

## Decision flow

```text
                  ┌─────────────────────┐
                  │  affinity scores    │
                  │  per existing copy  │
                  └─────────┬───────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
        action           latent          hybrid
            │               │               │
            ▼               ▼               ▼
        argmin           argmin          argmin
            │               │               │
            └───────┬───────┴───────┬───────┘
                    ▼               ▼
        relative_threshold=True  relative_threshold=False
                    │               │
                    ▼               ▼
        best_score close enough?  best_score > threshold?
                    │               │
                    ▼               ▼
            reuse copy            spawn copy
```

See [Affinity routing](../architecture/affinity-routing.md) for the
implementation details.
