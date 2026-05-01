"""Production environment overrides."""

from tsn_affinity.core.config import ModelConfig, SparseConfig

PROD_MODEL_CONFIG = ModelConfig(
    d_model=128,
    n_layers=3,
    n_heads=4,
    lr=3e-4,
)

PROD_SPARSE_CONFIG = SparseConfig(
    keep_ratio=0.3,
    quantize_after_task=True,
)
