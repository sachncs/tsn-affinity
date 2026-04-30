"""Task specification for benchmark configuration."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TaskSpec:
    """Specification for a single benchmark task.

    Attributes:
        name: Unique task identifier string.
        env_id: Gymnasium environment ID (None for offline-only tasks).
        seed: Random seed for environment.
        params: Environment configuration parameters.
    """

    name: str
    env_id: str | None
    seed: int
    params: Dict[str, Any]