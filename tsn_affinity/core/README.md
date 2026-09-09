# Core Module

Decision Transformer backbone, attention layers, configuration dataclasses, exceptions, and logging.

## Components

- `attention.py`: CausalSelfAttention, MLP, Block, LayerNorm
- `config.py`: ModelConfig, SparseConfig, RoutingConfig, TSNAffinityConfig
- `decision_transformer.py`: DecisionTransformer, DTBackbone
- `encoder.py`: ObsEncoder (CNN and MLP modes)
- `exceptions.py`: Domain-specific exception hierarchy
- `logging_config.py`: Structured logging setup

## Extension Points

Add new encoder types in `encoder.py` and register them in `ObsEncoder`.
