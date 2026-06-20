# Architecture

TSN-Affinity implements a continual offline reinforcement learning system that combines sparse subnetwork allocation with dynamic task routing.

## Core Components

### Decision Transformer Backbone

The shared model is a Decision Transformer (`tsn_affinity/core/decision_transformer.py`) that predicts actions from (observation, return-to-go, timestep) sequences. All components use sparse layers from `tsn_affinity/sparse/`.

### TinySubNetworks (TSN)

Each weight matrix has learnable "score" parameters. During training, only the top-k scores are kept active via straight-through estimation (`TopKMaskSTE`). This allows multiple tasks to share a single model while each using a sparse subset of weights.

Key files:
- `tsn_affinity/sparse/linear.py` - Sparse linear layer
- `tsn_affinity/sparse/conv2d.py` - Sparse convolutional layer
- `tsn_affinity/sparse/embedding.py` - Sparse embedding layer
- `tsn_affinity/sparse/topk.py` - Top-K mask autograd function
- `tsn_affinity/sparse/base.py` - Mask management mixin

### Affinity Routing

When a new task arrives, the router decides whether to spawn a new model copy or reuse an existing one based on task similarity.

Routing modes:
- **Action affinity**: Cross-entropy between predicted actions and new task data
- **Latent affinity**: KL divergence between observation latent distributions
- **Hybrid**: Weighted combination

Key files:
- `tsn_affinity/routing/metrics.py` - Affinity computation
- `tsn_affinity/routing/warmstarter.py` - Mask initialization

### Strategy Pattern

Training strategies encapsulate the full learning loop:

| Strategy | Description |
|----------|-------------|
| `TSNCoreStrategy` | Single-copy baseline (no routing) |
| `TSNAffinityStrategy` | Full affinity routing with action/latent/hybrid modes |
| `TSNReplayKLStrategy` | Replay-memory based KL routing |

Key files:
- `tsn_affinity/strategies/base.py` - Abstract interface
- `tsn_affinity/strategies/tsn_base.py` - Common mask management
- `tsn_affinity/strategies/copy_manager.py` - Model copy lifecycle

## Data Flow

```
Task 1 arrives
    |
    v
AffinityRouter evaluates similarity
    |
    +--> Low similarity --> Spawn new model copy
    |
    +--> High similarity --> Reuse existing copy
    |
    v
Strategy trains on task data
    |
    v
TSN allocates sparse subnetwork
    |
    v
Weights frozen for this task
    |
    v
Next task arrives...
```

## Project Structure

```
tsn_affinity/
├── core/           # DecisionTransformer, attention, encoder, config
├── sparse/         # TSN layers (Linear, Conv2d, Embedding, TopK STE)
├── routing/        # Affinity metrics, router, warmstarter
├── strategies/     # Training strategies and copy management
├── data/           # Trajectory handling, batch generation
├── interfaces/     # Abstract protocols and type definitions
├── services/       # Training orchestration service
├── run/            # Benchmark runner and analysis utilities
└── cli/            # Command-line entry points
```
