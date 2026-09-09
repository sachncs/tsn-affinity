"""Development environment overrides."""

from tsn_affinity.core.config import ModelConfig, SparseConfig

DEV_MODEL_CONFIG = ModelConfig(
    d_model=64,
    n_layers=2,
    n_heads=2,
    lr=1e-3,
)

DEV_SPARSE_CONFIG = SparseConfig(
    keep_ratio=0.5,
    quantize_after_task=False,
)
