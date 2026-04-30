"""Configuration dataclasses for model architecture and training."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for Decision Transformer model architecture.

    Attributes:
        obs_shape: Shape of observations (tuple).
        n_actions: Number of actions (int).
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
    """Configuration for affinity routing.

    Attributes:
        mode: Routing mode - "action", "latent", or "hybrid".
        action_threshold: Cross-entropy threshold for action affinity routing.
        latent_threshold: KL divergence threshold for latent affinity routing.
        hybrid_threshold: Score threshold for hybrid routing.
        hybrid_alpha: Weight for action component in hybrid (1-alpha for latent).
        normalize_scores: Whether to min-max normalize similarity scores.
        routing_n_batches: Number of batches to estimate action affinity.
        routing_batch_size: Batch size for affinity estimation.
    """

    mode: str = "action"
    action_threshold: float = 12.0
    latent_threshold: float = 25.0
    hybrid_threshold: float = 0.50
    hybrid_alpha: float = 0.70
    normalize_scores: bool = True
    routing_n_batches: int = 4
    routing_batch_size: int = 64


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