"""TSN-Core strategy (single copy, no routing - baseline)."""

from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.strategies.tsn_base import TSNBaseStrategy


class TSNCoreStrategy(TSNBaseStrategy):
    """TSN-Core: Single-copy TSN without any routing.

    This is the baseline TSN strategy with no model copy reuse.
    Each task gets its own sparse mask within a single model copy.
    No affinity routing, no replay-memory KL routing.

    This corresponds to the "TSN-Core" variant mentioned in the paper.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        seq_len: int = 20,
        device: str = "cuda",
        model_config: ModelConfig | None = None,
        sparse_config: SparseConfig | None = None,
    ) -> None:
        super().__init__(
            obs_shape, n_actions, seq_len, device, model_config, sparse_config
        )
