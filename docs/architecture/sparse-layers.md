# Sparse layers

TSN's sparse layers are drop-in replacements for `nn.Linear`,
`nn.Conv2d`, and `nn.Embedding`. Each layer carries a learned **score
tensor** with the same shape as its weights and selects the top-k
elements by magnitude during the forward pass.

## Top-k straight-through estimator

`tsn_affinity.sparse.topk.TopKMaskSTE` is a custom autograd function
that produces a binary mask in the forward pass and passes gradients
through unchanged in the backward pass.

```python
from tsn_affinity.sparse.topk import TopKMaskSTE
import torch

scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
mask = TopKMaskSTE.apply(scores, keep_ratio=0.4, free_mask=None)
loss = mask.sum()
loss.backward()
assert scores.grad is not None
```

The `free_mask` argument reserves a subset of positions (typically
weights that are still free, i.e. not consolidated into a previous
task). When provided, the top-k selection only considers free
positions.

## Linear, Conv2d, Embedding

All three layers live in `tsn_affinity.sparse`. They inherit from
both the corresponding PyTorch layer and the
[`TSNNMaskMixin`](../api/sparse.md#tsn_affinity.sparse.base.TSNNMaskMixin)
that handles mask bookkeeping.

| Class | Base | Use it for |
| --- | --- | --- |
| `TSNSparseLinear` | `nn.Linear` | Dense projections inside the Decision Transformer. |
| `TSNSparseConv2d` | `nn.Conv2d` | Convolutional encoders for image observations. |
| `TSNSparseEmbedding` | `nn.Embedding` | Token tables for discrete action spaces. |

Each layer accepts a `keep_ratio` argument that controls how much of
its weight matrix stays active. Setting `allow_weight_reuse=True`
makes the layer ignore the `occupied_*_mask` attributes; leaving it
`False` (the default) enforces strict mask isolation.

## Converting an existing model

`convert_to_sparse` walks a model and swaps its dense layers for
sparse equivalents in place. Use it once at the start of training:

```python
from tsn_affinity.sparse.converter import (
    SparseConversionConfig,
    convert_to_sparse,
)

config = SparseConversionConfig(
    keep_ratio=0.3,
    include_embeddings=True,
    skip_module_names=("dt.te",),
)
convert_to_sparse(model, config)
```

The `SparseConversionConfig` mirrors `SparseConfig` so you can pass
either; the CLI accepts the dataclass form for ergonomics.

## Mask bookkeeping

Every sparse layer exposes five mask tensors:

- `weight_mask`, `bias_mask` — last forward-pass mask.
- `active_weight_mask`, `active_bias_mask` — evaluation masks
  activated by `set_eval_task(task_id)`.
- `occupied_weight_mask`, `occupied_bias_mask` — union of every task's
  mask so far, used to enforce isolation.

`TSNBaseStrategy._sync_occupied_masks_into_modules` copies the
strategy's `consolidated_masks` dictionary into the occupied slots
of every sparse module. After that, gradients flowing through the
occupied positions are zeroed by
`zero_gradients_for_frozen_params`.

## Where to read next

- [Affinity routing](affinity-routing.md) — what to do
  when capacity runs out.
- [Continual learning loop](training-loop.md) — how
  masks interact with the optimizer step.
- API reference: [Sparse layers](../api/sparse.md).
