# Decision Transformer

The Decision Transformer (DT) is the sequence-modeling backbone that
predicts actions from `(return-to-go, observation, timestep)`
sequences. TSN-Affinity uses the architecture from Chen et al. (2021)
with a few small adaptations for sparse continual learning.

## Token sequence

The DT treats an episode as a sequence of token triplets:

```text
[R_0, s_0, a_0, R_1, s_1, a_1, …, R_T-1, s_T-1, a_T-1]
```

where `R_t` is the return-to-go, `s_t` is the observation embedding,
and `a_t` is the action embedding. Tokens are interleaved before
being fed to the transformer so the attention mask enforces causality
without any extra positional encoding beyond the timestep embedding.

## Components

The `DecisionTransformer` class composes three submodules:

| Component | File | Purpose |
| --- | --- | --- |
| `ObsEncoder` | `tsn_affinity.core.encoder` | Maps raw observations to `d_model` embeddings. |
| `DTBackbone` | `tsn_affinity.core.decision_transformer` | Transformer blocks + token embeddings + final head. |
| `action_head` | inside `DTBackbone` | Maps the state-position logits to action predictions. |

`ObsEncoder` supports two modes selected at construction time:

- **mlp** — two-layer MLP for 1-D vector observations.
- **cnn** — three convolutional layers for image observations
  (`(C, 84, 84)` is the default Atari size).

## Online inference

`DecisionTransformer.act(obs, returns_to_go, deterministic=True)` is
the autoregressive inference method. It maintains a rolling history of
observations, actions, return-to-go, and timesteps and selects the next
action from the last state-position logit.

```python
import torch
from tsn_affinity import DecisionTransformer

dt = DecisionTransformer(
    obs_shape=(4,),
    n_actions=2,
    d_model=64,
    n_layers=2,
    n_heads=2,
    seq_len=20,
)

dt.reset_history()
for _ in range(50):
    obs = torch.randn(4)
    action = dt.act(obs, returns_to_go=1.0, deterministic=True)
    print(action)
```

For continuous actions `act` returns a NumPy array shaped
`(action_dim,)`.

## Mask interaction

When the parent strategy calls `set_eval_task(task_id)` the strategy
activates the task's mask on every sparse module. The DT's attention
block still sees the same observations, but the per-layer masks
constrain which weights actually contribute. Activating and clearing
masks happens via
[`TSNNMaskMixin`](../api/sparse.md#tsn_affinity.sparse.base.TSNNMaskMixin).

## Where to read next

- [Sparse layers](sparse-layers.md) — the mask system that protects
  task-specific weights.
- [Continual learning loop](training-loop.md) — how the DT is trained
  in the context of TSN-Affinity.
- API reference: [Core](../api/core.md).
