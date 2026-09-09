# Affinity routing

Affinity routing is the decision layer that decides whether an
incoming task is similar enough to an existing model copy to reuse it,
or whether to spawn a fresh copy.

## Modes

The router supports three modes:

- **action** — measures the average cross-entropy between each
  existing copy's predicted actions and the demonstrated actions of
  the new task.
- **latent** — measures the symmetric KL divergence between
  diagonal Gaussian fits of observation latents.
- **hybrid** — weighted combination of the two metrics.

The `RoutingConfig` dataclass exposes a single `mode` field plus the
thresholds and routing-batch hyperparameters that each mode uses.

## Score computation

Action and latent metrics are implemented in
`tsn_affinity.routing.metrics`:

- `compute_action_affinity` returns a scalar mean cross-entropy for
  one model.
- `compute_action_affinity_batch` evaluates a list of models against
  the same trajectories in a single batched forward pass.
- `compute_latent_affinity` returns a symmetric KL score for one
  stored statistics pair.
- `compute_latent_affinity_batch` is the multi-copy equivalent.

The hybrid mode is a weighted combination that is normalised when
there are at least two candidates:

```text
final_score[t] = alpha * normalized_action[t] + (1 - alpha) * normalized_latent[t]
```

When only one previous task exists the router falls back to
`compute_hybrid_affinity`, which uses an absolute ratio comparison
that does not require min-max normalisation.

## Copy creation

After computing the affinity scores, the router picks the existing
copy with the lowest score and decides whether to reuse or spawn a
new copy:

- **Absolute threshold** — spawn a new copy when the best score
  exceeds the configured threshold.
- **Relative threshold** — spawn a new copy when the best score is
  too close to the second-best score (controlled by
  `copy_creation_margin`).

If a new copy is required the strategy calls `_make_fresh_copy`,
which builds a fresh `DecisionTransformer` and converts it to sparse
form. The optimizer is reconstructed from the existing hyperparameters.

## Warm-starting masks

When the router reuses an existing copy, the new task's mask scores
are warm-started from the source task's mask:

```text
new_score = noise(N(0, noise_std)) + strength * source_mask
```

This gives the optimizer a sensible starting point and speeds up
convergence on tasks that share structure. Tune
`RoutingConfig.warmstart_strength` and `RoutingConfig.warmstart_noise_std`
for your task distribution; set `warmstart=False` to disable it.

## Validation

The router validates that every score is finite before making a
decision. If a non-finite score (typically NaN from a divergent
training run) sneaks in, `RoutingError` is raised instead of silently
routing to the first copy.

## Where to read next

- [Routing modes how-to](../guides/routing-modes.md) — pick the right
  mode for your task distribution.
- [Sparse layers](sparse-layers.md) — what masks the router
  manipulates.
- API reference: [Routing](../api/routing.md).
