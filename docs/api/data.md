# API — Data

The `data` package hosts trajectory schemas, batch loaders, and the
Panda-specific helpers.

## Modules

- `tsn_affinity.data.schemas.trajectory` — `Trajectory`,
  `discount_cumsum`.
- `tsn_affinity.data.loaders.batch_loader` — `make_minibatches`,
  `masked_cross_entropy`, `masked_mse`, `unpack_batch_discrete`.
- `tsn_affinity.data.loaders.panda_loader` — `load_panda_offline_pkl`,
  `make_minibatches_panda`, `unpack_batch_continuous`.

The legacy modules `tsn_affinity.data.trajectory`,
`tsn_affinity.data.batch`, and `tsn_affinity.data.panda` re-export the
above for backward compatibility.

## Auto-generated reference

### `tsn_affinity.data.schemas.trajectory`

::: tsn_affinity.data.schemas.trajectory

### `tsn_affinity.data.loaders.batch_loader`

::: tsn_affinity.data.loaders.batch_loader

### `tsn_affinity.data.loaders.panda_loader`

::: tsn_affinity.data.loaders.panda_loader
