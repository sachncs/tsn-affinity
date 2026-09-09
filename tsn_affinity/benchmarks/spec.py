"""Task specification dataclass.

A ``TaskSpec`` describes a single continual-learning task in enough
detail that environment adapters can instantiate the underlying
environment, randomize it, and produce consistent observation/action
spaces.

Typical fields:
    - ``name``: Human-readable identifier (used in logs and filenames).
    - ``env_id``: Optional gymnasium environment identifier.
    - ``seed``: Random seed for the environment.
    - ``params``: Free-form parameters passed to the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["TaskSpec"]


@dataclass
class TaskSpec:
    """Description of a single continual-learning task.

    Attributes:
        name: Human-readable task identifier.
        env_id: Optional gymnasium environment identifier
            (for example, ``"ALE/Breakout-v5"``). May be ``None`` when
            the adapter does not require one (Panda, synthetic).
        seed: Random seed for environment initialization.
        params: Free-form keyword arguments passed to the adapter's
            ``create_env``. Defaults to an empty dictionary.
    """

    name: str
    env_id: str | None = None
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)
