"""Task registry: name -> adapter lookup.

The registry is a process-wide singleton that maps string identifiers
to environment adapters. Use it to register a custom adapter once at
startup and then resolve it by name anywhere in the program.

Example:
    >>> from tsn_affinity.benchmarks import TaskRegistry, AtariAdapter
    >>> TaskRegistry().register("atari", AtariAdapter())
    >>> TaskRegistry().get("atari")  # doctest: +ELLIPSIS
    <tsn_affinity.benchmarks.adapters.AtariAdapter object at 0x...>
"""

from __future__ import annotations

from collections.abc import Iterator
from threading import RLock

from tsn_affinity.benchmarks.adapters import BaseEnvAdapter

__all__ = ["TaskRegistry"]


class TaskRegistry:
    """Process-wide registry mapping adapter names to instances.

    Use :meth:`register` to associate a name with an adapter, and
    :meth:`get` to retrieve an adapter by name. The registry is a
    singleton: every call to the constructor returns the same object.
    """

    _instance: TaskRegistry | None = None
    _lock = RLock()
    _adapters: dict[str, BaseEnvAdapter]

    def __new__(cls) -> TaskRegistry:
        """Return the process-wide singleton, initializing it on first call."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._adapters = {}
            return cls._instance

    def register(self, name: str, adapter: BaseEnvAdapter) -> None:
        """Register an adapter under ``name``.

        Args:
            name: Identifier used to look up the adapter later.
            adapter: Adapter instance to register. Must implement the
                ``BaseEnvAdapter`` protocol.

        Raises:
            TypeError: When ``adapter`` is not a ``BaseEnvAdapter``
                instance (this is a soft check enforced for clarity).
        """
        if not hasattr(adapter, "create_env"):
            raise TypeError(f"Adapter for {name!r} must implement BaseEnvAdapter.")
        self._adapters[name] = adapter

    def unregister(self, name: str) -> None:
        """Remove the adapter registered under ``name`` (if any).

        Args:
            name: Identifier to remove.
        """
        self._adapters.pop(name, None)

    def get(self, name: str) -> BaseEnvAdapter:
        """Retrieve the adapter registered under ``name``.

        Args:
            name: Identifier previously passed to :meth:`register`.

        Returns:
            Registered adapter instance.

        Raises:
            KeyError: When no adapter has been registered under ``name``.
        """
        adapter: BaseEnvAdapter = self._adapters[name]
        return adapter

    def list_adapters(self) -> list[str]:
        """Return the list of currently registered adapter names.

        Returns:
            List of registered adapter names.
        """
        return sorted(self._adapters.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._adapters

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_adapters())

    def __len__(self) -> int:
        return len(self._adapters)
