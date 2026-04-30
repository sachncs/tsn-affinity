"""Task registry for environment adapters."""

from typing import Any, Dict, Protocol, runtime_checkable

from tsn_affinity.benchmarks.task_spec import TaskSpec


@runtime_checkable
class EnvAdapter(Protocol):
    """Protocol for environment adapters.

    Adapters encapsulate environment creation and preprocessing
    for different task families (Atari, Panda, CartPole, etc.).
    """

    def create_env(self, spec: TaskSpec) -> Any:
        """Create a gymnasium environment from a task spec."""
        ...

    def describe(self, env: Any) -> Dict[str, Any]:
        """Return environment metadata (obs shape, n_actions, etc.)."""
        ...

    def normalize_obs(self, obs):
        """Normalize observation if needed."""
        ...

    def normalize_reward(self, reward):
        """Normalize reward if needed."""
        ...


class TaskRegistry:
    """Singleton registry for environment adapters.

    Usage:
        registry = TaskRegistry()
        registry.register("atari", AtariAdapter())
        registry.register("panda", PandaAdapter())
        adapter = registry.get("atari")
    """

    _instance = None
    _adapters: Dict[str, EnvAdapter] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._adapters = {}
        return cls._instance

    def register(self, name: str, adapter: EnvAdapter) -> None:
        self._adapters[name] = adapter

    def get(self, name: str) -> EnvAdapter:
        if name not in self._adapters:
            raise KeyError(f"No adapter registered for '{name}'")
        return self._adapters[name]

    def has(self, name: str) -> bool:
        return name in self._adapters

    def list_adapters(self) -> list:
        return list(self._adapters.keys())