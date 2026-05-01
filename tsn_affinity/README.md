# TSN-Affinity Package

Core library for Training Sparse Networks with Affinity-based routing.

## Subpackages

- `benchmarks/`: Environment adapters and task registries.
- `core/`: Model definitions, configurations, and exceptions.
- `data/`: Trajectory schemas, loaders, and batch generators.
- `interfaces/`: Strategy and adapter protocols.
- `routing/`: Affinity metrics, warm-starting, and router logic.
- `run/`: Analysis and execution utilities.
- `services/`: High-level orchestration (e.g., TrainingService).
- `sparse/`: Sparse layers with score-based masking.
- `strategies/`: Continual learning strategies.
