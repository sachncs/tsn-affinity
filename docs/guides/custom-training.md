# Train your own model

This guide shows how to wire TSN-Affinity into a custom training loop
when you need finer control than the bundled CLIs provide.

## Strategy lifecycle

Every strategy follows the same lifecycle:

1. Construct the strategy with the desired configuration.
2. Call `train_task(trajectories, steps, batch_size)` once per task.
3. Call `after_task(trajectories)` to freeze the masks and finalise
   bookkeeping.
4. Optionally call `set_eval_task(task_id)` to switch between masks
   at evaluation time.

```python
from tsn_affinity import (
    RoutingConfig,
    SparseConfig,
    ModelConfig,
    TSNAffinityStrategy,
)

strategy = TSNAffinityStrategy(
    obs_shape=obs_shape,
    n_actions=n_actions,
    seq_len=seq_len,
    device=device,
    model_config=ModelConfig(d_model=128, n_layers=3, n_heads=4),
    sparse_config=SparseConfig(keep_ratio=0.3),
    affinity_config=RoutingConfig(mode="hybrid"),
    seed=seed,
)

for task_id, trajectories in enumerate(task_trajectories):
    metrics = strategy.train_task(trajectories, steps=2000, batch_size=64)
    strategy.after_task(trajectories)
    print(f"Task {task_id}: {metrics}")
```

## Custom evaluation

Each strategy exposes a `set_eval_task(task_id)` method that activates
the task-specific mask. Use it before running inference:

```python
strategy.set_eval_task(0)
logits = strategy.model(obs, actions, rtg, ts, attention_mask=mask)
strategy.clear_eval_task()
```

`clear_eval_task()` deactivates all masks so you can train the next
task without leaking the previous evaluation state.

## Plug in a custom optimizer

By default every strategy uses AdamW. To swap the optimizer, build the
strategy, then replace `strategy.optimizer`:

```python
strategy = TSNCoreStrategy(
    obs_shape=(4,),
    n_actions=2,
    seq_len=10,
    device="cpu",
    sparse_config=SparseConfig(keep_ratio=0.3),
    seed=0,
)
strategy.optimizer = torch.optim.SGD(
    strategy.model.parameters(),
    lr=1e-3,
    momentum=0.9,
)
```

When you spawn new copies (`TSNAffinityStrategy.train_task` with a
new affinity copy), the optimizer factory in `_make_fresh_copy`
mirrors the current optimizer configuration. If you swap optimizers
mid-training, re-spawn the copy to propagate the change.

## Resume training

To resume from a checkpoint, save and restore the strategy's full
state:

```python
import torch

# Save
torch.save({
    "model": strategy.model.state_dict(),
    "optimizer": strategy.optimizer.state_dict(),
    "per_task_masks": strategy.per_task_masks,
    "consolidated_masks": strategy.consolidated_masks,
    "task_codebooks": strategy.task_codebooks,
    "task_keep_ratios": strategy.task_keep_ratios,
    "current_task_id": strategy.current_task_id,
}, "checkpoint.pt")

# Restore
state = torch.load("checkpoint.pt")
strategy.model.load_state_dict(state["model"])
strategy.optimizer.load_state_dict(state["optimizer"])
strategy.per_task_masks = state["per_task_masks"]
strategy.consolidated_masks = state["consolidated_masks"]
strategy.task_codebooks = state["task_codebooks"]
strategy.task_keep_ratios = state["task_keep_ratios"]
strategy.current_task_id = state["current_task_id"]
```

`TSNAffinityStrategy` adds two more fields you may want to persist:
`copy_states` (list of `ModelCopy` objects) and `task_to_copy`
(dictionary mapping task IDs to copy IDs).

## Logging

Use `setup_logging` to configure structured logging for the
`tsn_affinity` logger:

```python
from tsn_affinity.core.logging_config import setup_logging
import logging

setup_logging(level=logging.INFO)
```

Every strategy logs its training progress under the `tsn_affinity`
logger name; adjust the format string to integrate with your
existing observability stack.
