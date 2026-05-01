# Data Module

Trajectory containers, batch generation, and environment-specific loaders.

## Subpackages

- `loaders/`: Minibatch generators for discrete and continuous actions
- `transformers/`: Trajectory preprocessing utilities
- `schemas/`: Data structures (Trajectory dataclass)

## Inputs/Outputs

- **Input**: Lists of `Trajectory` objects
- **Output**: Batched tensors `(obs, actions, rtg, timesteps, mask)`

## Extension Points

Add new loaders in `loaders/` for custom action spaces.
