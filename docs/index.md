---
hide:
  - navigation
  - toc
---

# TSN-Affinity

<p class="hero-subtitle">Continual offline reinforcement learning with training-aware sparse subnetworks and affinity-based copy routing.</p>

<div class="hero-cta" markdown>

[Install](getting-started.md#install){ .md-button .md-button--primary }
[Quick start](getting-started.md){ .md-button }
[Browse API](api.md){ .md-button }

</div>

<div class="hero-meta" markdown>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/sachncs/tsn-affinity/blob/master/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/sachncs/tsn-affinity/ci.yml?branch=master)](https://github.com/sachncs/tsn-affinity/actions)

</div>

## Why TSN-Affinity

Continual reinforcement learning collapses the moment a new task overwrites the weights that mattered for the previous one. TSN-Affinity stops that collapse by combining two ideas:

1. **Training-Aware Sparse Networks (TSN)** allocate a dedicated sparse subnetwork per task inside a shared Decision Transformer, so each task's weights stay reachable.
2. **Affinity Routing** decides whether a new task fits an existing subnetwork or warrants a fresh copy, using action similarity, latent similarity, or a hybrid of both.

The result is a single PyTorch package you can install with `pip` and use to train continual offline RL agents on synthetic, Atari, or Panda tasks.

## What you can do with it

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Train in one line**

    ---

    The `tsn-benchmark` CLI runs the canonical synthetic benchmark so you can confirm your install works before writing code.

    ```bash
    tsn-benchmark \
      --strategies tsn_core tsn_affinity \
      --n-tasks 5 --trajs-per-task 10 \
      --train-steps 200 --n-runs 3 \
      --output runs/benchmark
    ```

-   :material-graph:{ .lg .middle } **Mix and match strategies**

    ---

    `TSNCoreStrategy`, `TSNAffinityStrategy`, `TSNReplayKLStrategy`, `CumulativeReplayStrategy`, and `NaiveStrategy` all share the same interface, so you can swap them with a single argument.

-   :material-vector-triangle:{ .lg .middle } **Control the sparsity**

    ---

    Set the per-task keep ratio, decide whether embeddings participate in masking, opt into k-means quantization, or supply your own conversion via `convert_to_sparse`.

-   :material-gamepad-variant:{ .lg .middle } **Evaluate on Atari**

    ---

    `tsn-atari` collects random-policy trajectories, trains a TSN strategy on them, and reports human-normalized scores using the ALE baselines.

-   :material-robot:{ .lg .middle } **Extend to new environments**

    ---

    Register a new environment with the `TaskRegistry`, plug it into the affinity router, and the rest of the pipeline keeps working.

-   :material-test-tube:{ .lg .middle } **Compose with PyTorch**

    ---

    All sparse layers subclass the standard `nn.Linear`, `nn.Conv2d`, and `nn.Embedding`, so you can drop them into any PyTorch model.

</div>

## How it works

```text
Task 1 ──┐
         │
Task 2 ──┤──► AffinityRouter ──► Strategy.train_task ──► TSN Mask Allocator
         │            │                      │                     │
Task N ──┘            │                      │                     ▼
                      ▼                      ▼              Frozen weights for
              Existing copy or       TSN copies share         previous tasks
              new model copy?        a single backbone
```

The router measures action affinity, latent affinity, or a hybrid of the two. If the new task is similar enough to an existing copy, that copy is reused and its sparse mask is warm-started. Otherwise a fresh copy is spawned with isolated weights.

Read the [architecture overview](architecture.md) for the full design.

## Install

Install the core package:

```bash
pip install tsn-affinity
```

Add Atari support:

```bash
pip install "tsn-affinity[atari]"
```

Install the development toolchain:

```bash
pip install "tsn-affinity[dev]"
```

Verify everything works:

```bash
python -c "import tsn_affinity; print(tsn_affinity.__version__)"
tsn-benchmark --strategies tsn_core --n-tasks 3 --train-steps 50 --n-runs 1 --output runs/smoke
```

See [Installation](guides/installation.md) for troubleshooting and platform-specific notes.

## Next steps

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **[Getting started](getting-started.md)**

    ---

    Install, configure, and run your first continual-learning benchmark end to end.

-   :material-tools:{ .lg .middle } **[How-to guides](guides/index.md)**

    ---

    Task-oriented recipes for tuning sparsity, switching routing modes, and adding new environments.

-   :material-api:{ .lg .middle } **[API reference](api.md)**

    ---

    Auto-generated reference for every public class and function.

-   :material-account-group:{ .lg .middle } **[Contributing](contributing.md)**

    ---

    Development setup, testing, and code-style conventions.

</div>
