"""Base protocol every environment adapter implements.

Adapters expose three operations:

    - ``create_env(spec)``: Instantiate the environment.
    - ``describe(env)``: Return observation/action metadata.
    - ``is_compatible(spec)``: Quickly check whether this adapter can
      handle a given ``TaskSpec``.

The protocol lives in its own module so callers can ``isinstance``
checks against ``BaseEnvAdapter`` without importing every concrete
adapter implementation.
"""

from __future__ import annotations

from typing import Any, Protocol

from tsn_affinity.benchmarks.spec import TaskSpec

__all__ = ["BaseEnvAdapter"]


class BaseEnvAdapter(Protocol):
    """Protocol every environment adapter implements."""

    def create_env(self, spec: TaskSpec) -> Any:
        """Create an environment instance for ``spec``.

        Args:
            spec: Task specification describing the environment to create.

        Returns:
            Instantiated environment object.
        """
        ...

    def describe(self, env: Any) -> dict[str, Any]:
        """Describe the observation and action spaces of ``env``.

        Args:
            env: Environment instance returned by ``create_env``.

        Returns:
            Dictionary with at least the keys ``obs_shape``, ``n_actions``,
            and ``env_type``. Continuous-action adapters should also set
            ``action_dim``.
        """
        ...

    def is_compatible(self, spec: TaskSpec) -> bool:
        """Return ``True`` if this adapter can handle ``spec``.

        Args:
            spec: Task specification to check.

        Returns:
            Whether the adapter supports the spec.
        """
        ...
