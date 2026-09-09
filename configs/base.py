"""Base configuration shared across environments."""

from tsn_affinity.core.config import ModelConfig, RoutingConfig, SparseConfig

BASE_MODEL_CONFIG = ModelConfig()
BASE_SPARSE_CONFIG = SparseConfig()
BASE_ROUTING_CONFIG = RoutingConfig()
