# How-to guides

The guides on this page walk you through task-oriented recipes. Pick
the one closest to what you want to accomplish, then drill in.

## Recipes

<div class="grid cards" markdown>

-   :material-tune:{ .lg .middle } **[Tune sparsity](sparsity-tuning.md)**

    ---

    Choose `keep_ratio`, opt embeddings in or out, decide whether to
    quantize weights between tasks.

-   :material-vector-combine:{ .lg .middle } **[Pick a routing mode](routing-modes.md)**

    ---

    Switch between `action`, `latent`, and `hybrid` affinity routing
    and tune the threshold or margin.

-   :material-gamepad-variant:{ .lg .middle } **[Run Atari experiments](atari.md)**

    ---

    Collect random-policy trajectories, train a TSN strategy on them,
    and compute human-normalized scores.

-   :material-robot:{ .lg .middle } **[Add a new environment](new-environment.md)**

    ---

    Register a custom environment with `TaskRegistry` and plug it into
    the affinity router.

-   :material-tools:{ .lg .middle } **[Train your own model](custom-training.md)**

    ---

    Wire the public API into a custom training loop with your own data.

</div>

## Conventions used in the guides

- Code blocks assume you have already installed TSN-Affinity.
- Examples are CPU-only by default. Add `--device cuda` (CLI) or pass
  `device="cuda"` (Python API) when you have a GPU.
- All examples set `seed=...` so they are reproducible.
