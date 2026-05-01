"""Benchmark and environment adapter interfaces."""

from typing import Any, Protocol

from tsn_affinity.benchmarks.spec import TaskSpec


class EnvAdapterInterface(Protocol):
    """Protocol for environment adapters."""

    def create_env(self, spec: TaskSpec) -> Any:
        """Create an environment from a task specification.

        Args:
            spec: Task specification.

        Returns:
            Instantiated environment.
        """
        ...

    def describe(self, env: Any) -> dict[str, Any]:
        """Describe the environment's observation and action spaces.

        Args:
            env: Environment instance.

        Returns:
            Dictionary with keys such as ``obs_shape`` and ``n_actions``.
        """
        ...

    def is_compatible(self, spec: TaskSpec) -> bool:
        """Check whether this adapter can handle the given task spec.

        Args:
            spec: Task specification.

        Returns:
            True if the adapter is compatible.
        """
        ...
