# Concepts

This page is a vocabulary cheat sheet. Each term links to the module
or guide that explains it in depth.

## Continual reinforcement learning

**Continual RL** trains an agent on a sequence of tasks over time.
Unlike classical RL, the agent cannot freely revisit old data, so it
must balance plasticity (learning new tasks) with stability
(retaining old tasks). TSN-Affinity uses sparse subnetworks and
affinity routing to keep both ends in reach.

**Offline RL** learns from a fixed dataset rather than environment
interaction. TSN-Affinity assumes you provide a trajectory buffer per
task and never reaches back into the environment during training.

## Task, trajectory, batch

A **task** is one labelled distribution you want the agent to master.
The TSN-Affinity CLI exposes five Atari games plus three synthetic
or Panda task splits.

A **trajectory** is a single episode: an observation, an action, a
reward, and a timestep per timestep. TSN-Affinity stores trajectories
in a small `Trajectory` dataclass defined in
`tsn_affinity.data.schemas.trajectory`.

A **batch** is a stack of trajectory slices shaped
`(batch_size, seq_len, ...)`. The `make_minibatches` generator emits
batches of `(obs, actions, returns_to_go, timesteps, mask)`.

## Subnetworks and masks

A **TSN subnetwork** is a sparse subset of the model's weights that
is reserved for a single task. Each layer keeps a learned **score**
tensor; the top-k scores are activated via straight-through estimation
(see [`TopKMaskSTE`](../api/sparse.md#tsn_affinity.sparse.topk.TopKMaskSTE)).

A **consolidated mask** is the union of all per-task masks. The base
strategy uses it to zero out gradients on weights that already encode
another task, preventing interference.

## Routing and copies

A **copy** is a snapshot of the model together with its optimizer and
per-task mask dictionaries. The `CopyManager` class keeps a list of
copies and exposes helpers to create and activate them.

The **affinity router** decides whether to reuse an existing copy or
spawn a new one. Three modes are supported:

- **action** — average cross-entropy between the source copy's action
  predictions and the new task's demonstrated actions.
- **latent** — symmetric KL divergence between observation-latent
  diagonal Gaussian fits.
- **hybrid** — weighted combination of the two metrics.

The full design lives in [Affinity routing](../architecture/affinity-routing.md).

## Metrics

TSN-Affinity reports four standard continual-learning metrics,
implemented in `tsn_affinity.benchmarks.metrics`:

- **ACC** — average score on the diagonal of the performance matrix
  (score after training on each task).
- **BWT** — backward transfer; negative when the model forgets.
- **Forgetting** — average per-task drop from the post-training score
  to the final evaluation.
- **FWT** — forward transfer; zero-shot transfer to future tasks.

See [`StandardCLMetrics`](../api/benchmarks.md#tsn_affinity.benchmarks.metrics.StandardCLMetrics)
for the implementation.
