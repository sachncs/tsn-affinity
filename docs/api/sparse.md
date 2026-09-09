# API — Sparse Layers

The `sparse` package hosts the masked-layer implementations and the
conversion utilities that swap dense layers for sparse equivalents.

## Modules

- `tsn_affinity.sparse.base` — `TSNNMaskMixin`.
- `tsn_affinity.sparse.topk` — `TopKMaskSTE`.
- `tsn_affinity.sparse.linear` — `TSNSparseLinear`.
- `tsn_affinity.sparse.conv2d` — `TSNSparseConv2d`.
- `tsn_affinity.sparse.embedding` — `TSNSparseEmbedding`.
- `tsn_affinity.sparse.converter` — `SparseConversionConfig`,
  `convert_to_sparse`, `iter_sparse_modules`, `kmeans_quantize`,
  `rebuild_optimizer`.

## Auto-generated reference

### `tsn_affinity.sparse.base`

::: tsn_affinity.sparse.base

### `tsn_affinity.sparse.topk`

::: tsn_affinity.sparse.topk

### `tsn_affinity.sparse.linear`

::: tsn_affinity.sparse.linear

### `tsn_affinity.sparse.conv2d`

::: tsn_affinity.sparse.conv2d

### `tsn_affinity.sparse.embedding`

::: tsn_affinity.sparse.embedding

### `tsn_affinity.sparse.converter`

::: tsn_affinity.sparse.converter
