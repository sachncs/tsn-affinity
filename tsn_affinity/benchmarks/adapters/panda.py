"""Panda robotics adapter (stub).

The full Panda backend depends on ``panda-gym``, which is not part of
TSN-Affinity's core dependencies. The adapter matches any spec whose
name contains the substring ``"panda"`` so that ``TaskRegistry`` and
documentation flows can advertise Panda support, but raises
``NotImplementedError`` when asked to create an environment.

Install ``panda-gym`` separately and extend this class to wire up
real environments.
"""

from __future__ import annotations

from typing import Any

from tsn_affinity.benchmarks.spec import TaskSpec

__all__ = ["PandaAdapter"]


class PandaAdapter:
    """Adapter for Panda robotics manipulation environments."""

    env_type = "panda"

    def is_compatible(self, spec: TaskSpec) -> bool:
        """Return ``True`` for Panda-shaped task specs.

        Args:
            spec: Task specification.

        Returns:
            ``True`` if the spec name or ``env_id`` contains
            ``"panda"``.
        """
        if spec.env_id and "panda" in spec.env_id.lower():
            return True
        return "panda" in spec.name.lower()

    def create_env(self, spec: TaskSpec) -> Any:
        """Instantiate a Panda environment.

        Args:
            spec: Task specification.

        Returns:
            A Panda environment.

        Raises:
            NotImplementedError: Always. The Panda backend is provided
                by the optional ``panda-gym`` package; install it
                separately to use this adapter.
        """
        raise NotImplementedError(
            "PandaAdapter requires the optional 'panda-gym' package. "
            "Install it with `pip install panda-gym` to enable Panda "
            "environments."
        )

    def describe(self, env: Any) -> dict[str, Any]:
        """Describe the observation and action spaces of a Panda env.

        Args:
            env: A Panda environment instance.

        Returns:
            Dictionary with ``obs_shape``, ``n_actions``, ``action_dim``,
            and ``env_type``.
        """
        obs_shape = tuple(getattr(env.observation_space, "shape", ()))
        action_space = env.action_space

        info: dict[str, Any] = {
            "obs_shape": obs_shape,
            "env_type": self.env_type,
        }
        if hasattr(action_space, "n"):
            info["n_actions"] = int(action_space.n)
        elif hasattr(action_space, "shape"):
            shape = action_space.shape
            info["action_dim"] = int(shape[0])
            info["continuous_actions"] = True
            info["n_actions"] = info["action_dim"]
        return info
