# API — Strategies

The `strategies` package exposes every continual-learning strategy
and the supporting utilities.

## Modules

- `tsn_affinity.strategies.base` — `BaseStrategy` (abstract).
- `tsn_affinity.strategies.tsn_base` — `TSNBaseStrategy` (sparse
  mask management + quantization).
- `tsn_affinity.strategies.tsn_core` — `TSNCoreStrategy` (single
  copy, no routing).
- `tsn_affinity.strategies.tsn_affinity` — `TSNAffinityStrategy`
  (action / latent / hybrid routing).
- `tsn_affinity.strategies.tsn_replay_kl` — `TSNReplayKLStrategy`
  (replay-memory KL routing).
- `tsn_affinity.strategies.cumulative_replay` —
  `CumulativeReplayStrategy` (cumulative replay buffer).
- `tsn_affinity.strategies.naive` — `NaiveStrategy` (no continual
  learning mechanism).
- `tsn_affinity.strategies.copy_manager` — `CopyManager`.
- `tsn_affinity.strategies.model_copy` — `ModelCopy`.
- `tsn_affinity.strategies.training_utils` — frozen-parameter
  utilities.
- `tsn_affinity.services.training_service` — `TrainingService`.

## Auto-generated reference

### `tsn_affinity.strategies.base`

::: tsn_affinity.strategies.base

### `tsn_affinity.strategies.tsn_base`

::: tsn_affinity.strategies.tsn_base

### `tsn_affinity.strategies.tsn_core`

::: tsn_affinity.strategies.tsn_core

### `tsn_affinity.strategies.tsn_affinity`

::: tsn_affinity.strategies.tsn_affinity

### `tsn_affinity.strategies.tsn_replay_kl`

::: tsn_affinity.strategies.tsn_replay_kl

### `tsn_affinity.strategies.cumulative_replay`

::: tsn_affinity.strategies.cumulative_replay

### `tsn_affinity.strategies.naive`

::: tsn_affinity.strategies.naive

### `tsn_affinity.strategies.copy_manager`

::: tsn_affinity.strategies.copy_manager

### `tsn_affinity.strategies.model_copy`

::: tsn_affinity.strategies.model_copy

### `tsn_affinity.strategies.training_utils`

::: tsn_affinity.strategies.training_utils

### `tsn_affinity.services.training_service`

::: tsn_affinity.services.training_service
