"""Configuration dataclass for replay-KL routing strategy."""

from dataclasses import dataclass


@dataclass
class ReplayKLConfig:
    """Configuration for replay-memory KL routing.

    Attributes:
        memory_size: Number of observations to store per task for routing.
        replay_ratio: Fraction of training batch from replay buffer.
        kl_threshold: KL divergence threshold for creating new model copy.
        max_copies: Maximum number of model copies (None = unlimited).
    """

    memory_size: int = 256
    replay_ratio: float = 0.3
    kl_threshold: float = 0.25
    max_copies: int | None = None