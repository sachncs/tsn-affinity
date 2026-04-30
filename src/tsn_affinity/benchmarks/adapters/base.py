"""Base environment adapter protocol."""

from typing import Any, Dict, Protocol

from tsn_affinity.benchmarks.task_spec import TaskSpec


class BaseEnvAdapter(Protocol):
    """Protocol for environment adapters.

    Implement this protocol to support different task families.
    """

    def create_env(self, spec: TaskSpec) -> Any:
        """Create a gymnasium environment from a task spec."""
        ...

    def describe(self, env: Any) -> Dict[str, Any]:
        """Return environment metadata."""
        ...

    def is_compatible(self, spec: TaskSpec) -> bool:
        """Check if this adapter can handle the given task spec."""
        ...