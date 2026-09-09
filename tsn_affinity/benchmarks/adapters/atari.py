"""Atari environment adapter backed by gymnasium's ALE namespace.

The adapter matches any ``TaskSpec`` whose ``env_id`` (or ``params``
``game`` field) starts with the gymnasium ``ALE/`` prefix. For
convenience, legacy bare names such as ``"Breakout"`` are also
accepted because the original ``tsn-atari`` CLI accepted them.
"""

from __future__ import annotations

from typing import Any

from tsn_affinity.benchmarks.spec import TaskSpec

__all__ = ["AtariAdapter"]


def _make_atari_env(target: str, seed: int) -> Any:
    """Lazily import gymnasium and instantiate the target environment.

    Args:
        target: Fully-qualified gymnasium environment identifier.
        seed: Random seed passed to ``env.reset``.

    Returns:
        A gymnasium ``Env`` instance.

    Raises:
        ImportError: When gymnasium is not installed.
    """
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "AtariAdapter requires gymnasium. Install with "
            "`pip install 'tsn-affinity[atari]'`."
        ) from exc

    env = gym.make(target)
    env.reset(seed=int(seed))
    return env


class AtariAdapter:
    """Adapter for Atari environments via gymnasium's ALE namespace."""

    env_type = "atari"

    @staticmethod
    def _resolve_env_id(spec: TaskSpec) -> str | None:
        if spec.env_id:
            return spec.env_id
        return spec.params.get("game") or spec.params.get("env_id")

    def is_compatible(self, spec: TaskSpec) -> bool:
        """Return ``True`` if the spec targets an Atari game.

        Args:
            spec: Task specification.

        Returns:
            ``True`` when the spec's ``env_id`` or ``params.game`` is an
            ALE environment identifier or a known bare game name.
        """
        env_id = self._resolve_env_id(spec)
        if env_id is None:
            return False
        if env_id.startswith("ALE/"):
            return True
        name = env_id.split("/")[-1].split("-")[0]
        return bool(name) and name[0].isupper()

    def create_env(self, spec: TaskSpec) -> Any:
        """Instantiate the Atari environment described by ``spec``.

        Args:
            spec: Task specification with an ALE ``env_id`` or
                ``params.game`` field.

        Returns:
            A gymnasium ``Env`` instance.

        Raises:
            ValueError: When no environment identifier is provided.
            ImportError: When gymnasium is not installed.
        """
        env_id = self._resolve_env_id(spec)
        if env_id is None:
            raise ValueError(
                "AtariAdapter requires spec.env_id or spec.params['game']."
            )
        if env_id.startswith("ALE/"):
            target = env_id
        else:
            bare = env_id.split("/")[-1].split("-")[0]
            target = f"ALE/{bare}-v5"

        return _make_atari_env(target, spec.seed)

    def describe(self, env: Any) -> dict[str, Any]:
        """Describe the observation and action spaces of an Atari env.

        Args:
            env: An Atari gymnasium environment.

        Returns:
            Dictionary with ``obs_shape``, ``n_actions``, ``env_type``.
        """
        obs_space = env.observation_space
        action_space = env.action_space

        info: dict[str, Any] = {
            "obs_shape": tuple(getattr(obs_space, "shape", ())),
            "env_type": self.env_type,
        }
        if hasattr(action_space, "n"):
            info["n_actions"] = int(action_space.n)
        elif hasattr(action_space, "shape"):
            shape = action_space.shape
            info["action_dim"] = int(shape[0])
            info["n_actions"] = int(shape[0])
            info["continuous_actions"] = True
        else:  # pragma: no cover - defensive fallback
            info["n_actions"] = 0
        return info
