"""TSN-Affinity: Training-Aware Sparse Networks with Affinity Routing.

This package groups the model implementations, continual-learning
strategies, sparse layers, and supporting utilities needed to train
and evaluate TSN-Affinity.

The ``__version__`` attribute is populated from the installed
distribution metadata when available. The most commonly used classes
are re-exported here so callers can write::

    from tsn_affinity import (
        DecisionTransformer,
        RoutingConfig,
        SparseConfig,
        TSNAffinityStrategy,
        TSNCoreStrategy,
        TSNReplayKLStrategy,
    )

without navigating the internal package layout.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tsn-affinity")
except PackageNotFoundError:
    __version__ = "0.0.0"

from tsn_affinity.core.config import (
    ModelConfig,
    RoutingConfig,
    SparseConfig,
    TSNAffinityConfig,
)
from tsn_affinity.core.decision_transformer import (
    DecisionTransformer,
    DTBackbone,
)
from tsn_affinity.core.encoder import ObsEncoder
from tsn_affinity.core.exceptions import (
    BenchmarkError,
    ConfigurationError,
    DataError,
    MaskError,
    RoutingError,
    StrategyError,
    TSNAffinityError,
)
from tsn_affinity.core.logging_config import setup_logging
from tsn_affinity.data.schemas.trajectory import Trajectory, discount_cumsum
from tsn_affinity.routing.metrics import (
    compute_action_affinity,
    compute_hybrid_affinity,
    compute_latent_affinity,
    compute_latent_affinity_batch,
    normalize_scores,
)
from tsn_affinity.routing.warmstarter import MaskWarmstarter
from tsn_affinity.sparse.conv2d import TSNSparseConv2d
from tsn_affinity.sparse.converter import (
    SparseConversionConfig,
    convert_to_sparse,
    iter_sparse_modules,
    kmeans_quantize,
    rebuild_optimizer,
)
from tsn_affinity.sparse.embedding import TSNSparseEmbedding
from tsn_affinity.sparse.linear import TSNSparseLinear
from tsn_affinity.sparse.topk import TopKMaskSTE
from tsn_affinity.strategies import (
    BaseStrategy,
    CopyManager,
    ModelCopy,
    TSNAffinityStrategy,
    TSNBaseStrategy,
    TSNCoreStrategy,
    TSNReplayKLStrategy,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "DecisionTransformer",
    "DTBackbone",
    "ObsEncoder",
    # Configs
    "ModelConfig",
    "SparseConfig",
    "RoutingConfig",
    "TSNAffinityConfig",
    "SparseConversionConfig",
    # Exceptions
    "TSNAffinityError",
    "RoutingError",
    "MaskError",
    "ConfigurationError",
    "DataError",
    "StrategyError",
    "BenchmarkError",
    # Logging
    "setup_logging",
    # Data
    "Trajectory",
    "discount_cumsum",
    # Routing
    "compute_action_affinity",
    "compute_hybrid_affinity",
    "compute_latent_affinity",
    "compute_latent_affinity_batch",
    "normalize_scores",
    "MaskWarmstarter",
    # Sparse
    "TSNSparseLinear",
    "TSNSparseConv2d",
    "TSNSparseEmbedding",
    "TopKMaskSTE",
    "convert_to_sparse",
    "iter_sparse_modules",
    "kmeans_quantize",
    "rebuild_optimizer",
    # Strategies
    "BaseStrategy",
    "TSNBaseStrategy",
    "TSNCoreStrategy",
    "TSNAffinityStrategy",
    "TSNReplayKLStrategy",
    "CopyManager",
    "ModelCopy",
]
