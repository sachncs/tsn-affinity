# API — Core

The `core` package hosts the model architecture, configuration
dataclasses, exception hierarchy, and logging helpers.

## Modules

- `tsn_affinity.core.config` — `ModelConfig`, `SparseConfig`,
  `RoutingConfig`, `TSNAffinityConfig`, `SparseConversionConfig`.
- `tsn_affinity.core.attention` — `LayerNorm`, `MLP`,
  `CausalSelfAttention`, `Block`.
- `tsn_affinity.core.decision_transformer` — `DTBackbone`,
  `DecisionTransformer`.
- `tsn_affinity.core.encoder` — `ObsEncoder`.
- `tsn_affinity.core.exceptions` — `TSNAffinityError`, `RoutingError`,
  `MaskError`, `ConfigurationError`, `DataError`, `StrategyError`,
  `BenchmarkError`.
- `tsn_affinity.core.logging_config` — `setup_logging`.

## Auto-generated reference

### `tsn_affinity.core.config`

::: tsn_affinity.core.config

### `tsn_affinity.core.attention`

::: tsn_affinity.core.attention

### `tsn_affinity.core.decision_transformer`

::: tsn_affinity.core.decision_transformer

### `tsn_affinity.core.encoder`

::: tsn_affinity.core.encoder

### `tsn_affinity.core.exceptions`

::: tsn_affinity.core.exceptions

### `tsn_affinity.core.logging_config`

::: tsn_affinity.core.logging_config
