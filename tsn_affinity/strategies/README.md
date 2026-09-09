# Strategies Module

Continual learning strategy implementations.

## Components

- `base.py`: Abstract `BaseStrategy` interface
- `tsn_base.py`: TSN mask management and quantization
- `tsn_core.py`: Single-copy TSN baseline
- `tsn_affinity.py`: TSN-Affinity with routing and warm-starting
- `tsn_replay_kl.py`: TSN with replay-memory KL routing
- `naive.py`: Naive baseline (no continual learning)
- `cumulative_replay.py`: Cumulative replay with sparse masks
- `copy_manager.py`: Lifecycle manager for model copies
- `model_copy.py`: Immutable dataclass for copy state
- `training_utils.py`: Gradient handling and snapshot utilities

## Extension Points

All strategies implement `BaseStrategy`. Subclass `TSNBaseStrategy` for new TSN variants.
