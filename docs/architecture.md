# Architecture

TSN-Affinity is a continual offline reinforcement-learning algorithm
built from three composable layers:

1. **Sparse subnetworks** (`tsn_affinity.sparse`) — drop-in
   replacements for the standard PyTorch layers, each carrying a
   learned score tensor and a top-k straight-through estimator.
2. **Affinity routing** (`tsn_affinity.routing`) — a router that picks
   between existing model copies based on task similarity.
3. **Strategies** (`tsn_affinity.strategies`) — orchestration code
   that ties everything together, freezes previous-task weights, and
   exposes a uniform `train_task` / `after_task` interface.

This page sketches the data flow. Each layer has a dedicated page:

- [Sparse layers](architecture/sparse-layers.md)
- [Affinity routing](architecture/affinity-routing.md)
- [Decision Transformer](architecture/decision-transformer.md)
- [Continual learning loop](architecture/training-loop.md)

## High-level diagram

```text
                      ┌──────────────────────┐
   Trajectories  ───► │   make_minibatches    │
                      └─────────┬────────────┘
                                ▼
                      ┌──────────────────────┐
                      │  Affinity router     │
                      │  (action / latent /  │
                      │   hybrid / replay_kl)│
                      └─────────┬────────────┘
                                ▼
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
   Reuse existing copy                      Spawn new copy
   (warm-start masks)                      (convert_to_sparse)
            │                                         │
            └─────────────────┬───────────────────────┘
                              ▼
              ┌────────────────────────────┐
              │  TSN strategy + Decision   │
              │  Transformer + sparse      │
              │  subnetworks                │
              └─────────────┬──────────────┘
                            ▼
              ┌────────────────────────────┐
              │  Frozen weight protection   │
              │  + per-task mask tracking  │
              └────────────────────────────┘
```

## Package layout

```text
tsn_affinity/
├── core/            # Decision Transformer, attention, configs, exceptions
├── sparse/          # TSN layers (Linear, Conv2d, Embedding, TopK STE)
├── routing/         # Affinity metrics, router, warm-starter
├── strategies/      # Continual learning strategies + copy management
├── data/            # Trajectory handling, batch generation
├── interfaces/      # Abstract protocols and type definitions
├── services/        # Training orchestration service
├── run/             # Benchmark runner and analysis utilities
├── benchmarks/      # Environment adapters, registry, metrics, baselines
└── cli/             # Command-line entry points (benchmark, atari, panda)
```

Each package has a `__init__.py` that re-exports its public API and a
`README.md` with a one-paragraph summary.

## Where to read next

- [Sparse layers](architecture/sparse-layers.md) — how masks are
  learned, applied, and protected.
- [Affinity routing](architecture/affinity-routing.md) — how new
  tasks are matched against existing copies.
- [Decision Transformer](architecture/decision-transformer.md) —
  the backbone architecture and its input format.
- [Continual learning loop](architecture/training-loop.md) — the
  per-step gradient flow inside a single training task.
