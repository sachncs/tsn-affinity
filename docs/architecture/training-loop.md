# Continual learning loop

This page walks through one optimizer step of `TSNAffinityStrategy`
(or any TSN strategy). Understanding the gradient flow is essential
for diagnosing "why does my model forget?" issues.

## One training step

```text
              ┌─────────────────────────────┐
              │       Sampled batch         │
              │  (obs, actions, rtg, ts)    │
              └─────────────┬───────────────┘
                            ▼
              ┌─────────────────────────────┐
              │  Forward pass through DT    │
              │  (sparse masks active)      │
              └─────────────┬───────────────┘
                            ▼
              ┌─────────────────────────────┐
              │  Loss: masked cross-entropy │
              └─────────────┬───────────────┘
                            ▼
              ┌─────────────────────────────┐
              │  Backward pass              │
              └─────────────┬───────────────┘
                            ▼
   ┌────────────────────────┴─────────────────────────┐
   ▼                                                  ▼
┌───────────────────────┐                ┌────────────────────────┐
│ Zero frozen gradients │                │ Zero non-maskable grads │
│ (consolidated masks)  │                │ (after task 0)         │
└─────────┬─────────────┘                └──────────┬─────────────┘
          ▼                                            ▼
┌───────────────────────┐                ┌────────────────────────┐
│ Snapshot frozen       │                │ Snapshot frozen params │
│ parameter values      │                │ for non-maskable layers│
└─────────┬─────────────┘                └──────────┬─────────────┘
          ▼                                            ▼
              ┌─────────────────────────────┐
              │  Optimizer step              │
              └─────────────┬───────────────┘
                            ▼
              ┌─────────────────────────────┐
              │  Restore frozen parameters   │
              └─────────────────────────────┘
```

## Why we zero gradients

Backward through the sparse layers produces gradients on **every**
parameter, even parameters that are masked out for the current task.
Without intervention the optimizer would happily update frozen
weights. We prevent that with three coordinated steps:

1. **Zero frozen gradients.** Walk every parameter covered by the
   `consolidated_masks` dictionary and zero its gradient in place.
   `verify_frozen_gradient_zeroing` asserts the step succeeded.
2. **Zero non-maskable gradients.** Parameters that are neither in
   `maskable_param_names` nor in `score_param_names` are frozen after
   the first task; their gradients are zeroed too.
3. **Restore parameter values.** Even with zeroed gradients, a
   non-zero `param.grad` would still be added by AdamW's adaptive
   update. We snapshot the frozen values before the optimizer step
   and restore them afterwards.

## Training utilities

The building blocks live in `tsn_affinity.strategies.training_utils`:

- `verify_frozen_gradient_zeroing` — returns `(True, "OK")` or
  `(False, "<error message>")` so the strategy can fail loudly.
- `snapshot_frozen_parameters` — produces a `TrainingSnapshot`
  dataclass covering every mask-protected parameter.
- `restore_frozen_parameters` — applies the snapshot after the
  optimizer step.
- `zero_gradients_for_frozen_params` — performs the in-place
  zeroing using the consolidated masks.
- `zero_gradients_for_non_maskable_params` — performs the
  in-place zeroing for non-maskable parameters after task 0.

## Failure modes

| Symptom | Likely cause |
| --- | --- |
| `RuntimeError: Frozen gradient zeroing failed` | A new sparse layer was added but the consolidated-masks dictionary was not refreshed. |
| Loss NaNs after a few hundred steps | LR too high or sequence length too long for the available memory. |
| Previous-task scores drop after every new task | `keep_ratio` too high — switch to a multi-copy strategy. |

## Where to read next

- [Sparse layers](sparse-layers.md) — what masks protect.
- [Affinity routing](affinity-routing.md) — what happens when
  capacity runs out.
- API reference: [Strategies](../api/strategies.md).
