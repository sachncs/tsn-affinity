"""Configuration dataclasses for model architecture and training.

All strategy configuration is gathered here so callers do not need to
import individual strategy modules just to construct an instance. The
three primary dataclasses are:

    - :class:`ModelConfig`: Transformer backbone hyperparameters.
    - :class:`SparseConfig`: TSN sparse layer configuration.
    - :class:`RoutingConfig`: Routing configuration shared by
      ``TSNAffinityStrategy`` and ``TSNReplayKLStrategy``.

A convenience :class:`TSNAffinityConfig` composes the three into a
single object so legacy callers can keep using one constructor call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "ModelConfig",
    "SparseConfig",
    "RoutingConfig",
    "TSNAffinityConfig",
]


@dataclass
class ModelConfig:
    """Configuration for Decision Transformer model architecture.

    Attributes:
        obs_shape: Shape of observations (tuple).
        n_actions: Number of discrete actions (ignored when
            ``continuous_actions`` is ``True``).
        seq_len: Context length K for Decision Transformer.
        d_model: Transformer embedding dimension.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        p_drop: Dropout probability.
        max_ep_len: Maximum episode length for timestep encoding.
        rtg_scale: Scale factor for returns-to-go normalization.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        grad_clip: Gradient clipping norm.
        continuous_actions: Whether the action space is continuous.
        action_dim: Dimensionality of continuous actions
            (ignored for discrete actions).
    """

    obs_shape: tuple = (4,)
    n_actions: int = 2
    seq_len: int = 20
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    p_drop: float = 0.1
    max_ep_len: int = 10000
    rtg_scale: float = 1000.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    continuous_actions: bool = False
    action_dim: int = 1


@dataclass
class SparseConfig:
    """Configuration for TSN sparse layers.

    Attributes:
        keep_ratio: Fraction of weights to keep per task (0.5 = 50%).
        include_embeddings: Whether to apply sparsity to embedding layers.
        allow_weight_reuse: Whether new tasks can reuse frozen weights.
        skip_module_names: Module name prefixes to skip during conversion.
        quant_clusters: Number of k-means clusters for quantization.
        quantize_after_task: Whether to quantize weights after each task.
    """

    keep_ratio: float = 0.5
    include_embeddings: bool = True
    allow_weight_reuse: bool = False
    skip_module_names: tuple = ("dt.te",)
    quant_clusters: int = 16
    quantize_after_task: bool = True


@dataclass
class RoutingConfig:
    """Unified routing configuration shared by all routing strategies.

    The routing mode decides how new tasks are matched against existing
    model copies. Each mode has its own primary threshold; ignored
    thresholds are tolerated to keep callers from having to remember
    which one applies.

    Attributes:
        mode: Routing mode. One of ``"action"``, ``"latent"``,
            ``"hybrid"``, or ``"replay_kl"``.
        action_threshold: Cross-entropy threshold for action affinity routing.
        latent_threshold: KL divergence threshold for latent affinity routing.
        hybrid_threshold: Score threshold for hybrid routing.
        hybrid_alpha: Weight for action component in hybrid
            (``1 - alpha`` weight for the latent component).
        kl_threshold: KL divergence threshold for replay-memory routing.
        normalize_scores: Whether to min-max normalize similarity scores.
        routing_n_batches: Number of batches to estimate action affinity.
        routing_batch_size: Batch size for affinity estimation.
        relative_threshold: If ``True``, use a relative threshold that
            compares the best score against the rest (score range).
        copy_creation_margin: Multiplier for the relative threshold.
        memory_size: Number of observations stored per task for
            replay-memory routing.
        max_copies: Maximum number of model copies (replay-memory routing).
        max_model_copies: Maximum number of model copies (affinity routing).
            ``None`` means unlimited.
        warmstart: Whether to warm-start mask scores from the source task.
        warmstart_strength: Scaling factor for warm-started scores.
        warmstart_noise_std: Std dev of noise added during warm-start.
        warmstart_on_new_copy: Whether to warm-start even new copies.
        seed: Random seed passed to the strategy for reproducibility.
            ``None`` keeps the strategy's global RNG state.
        affinity_metric: Affinity metric used for action-based routing.
            ``"auto"`` selects the metric from the action shape.
    """

    mode: Literal["action", "latent", "hybrid", "replay_kl"] = "action"
    action_threshold: float = 12.0
    latent_threshold: float = 25.0
    hybrid_threshold: float = 0.50
    hybrid_alpha: float = 0.70
    kl_threshold: float = 0.25
    normalize_scores: bool = True
    routing_n_batches: int = 4
    routing_batch_size: int = 64
    relative_threshold: bool = True
    copy_creation_margin: float = 2.5
    memory_size: int = 256
    max_copies: int | None = None
    max_model_copies: int | None = None
    warmstart: bool = True
    warmstart_strength: float = 2.0
    warmstart_noise_std: float = 0.02
    warmstart_on_new_copy: bool = False
    seed: int | None = None
    affinity_metric: str = "auto"


@dataclass
class TSNAffinityConfig:
    """Combined configuration for TSN-Affinity strategy.

    Attributes:
        sparse: Sparse layer configuration.
        routing: Affinity routing configuration.
        warmstart: Whether to warm-start mask scores from source task.
        warmstart_strength: Scaling factor for warm-started scores.
        warmstart_noise_std: Std dev of noise added during warm-start.
        warmstart_on_new_copy: Whether to warm-start even new copies.
        max_model_copies: Maximum number of model copies (None = unlimited).
    """

    sparse: SparseConfig = field(default_factory=SparseConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    warmstart: bool = True
    warmstart_strength: float = 2.0
    warmstart_noise_std: float = 0.02
    warmstart_on_new_copy: bool = False
    max_model_copies: int | None = None
