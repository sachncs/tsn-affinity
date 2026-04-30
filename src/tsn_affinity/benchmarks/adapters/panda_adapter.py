"""Panda robotic manipulation adapter."""

from typing import Any, Dict, Tuple

from tsn_affinity.benchmarks.adapters.base import BaseEnvAdapter
from tsn_affinity.benchmarks.task_spec import TaskSpec


class PandaAdapter(BaseEnvAdapter):
    """Environment adapter for Panda robotic manipulation tasks.

    Handles continuous control tasks with observation/action spaces
    appropriate for robot control.

    Attributes:
        host: Host for Panda simulation.
        port: Port for Panda simulation.
        reward_threshold: Success reward threshold.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        reward_threshold: float = 3.0,
    ) -> None:
        self.host = host
        self.port = port
        self.reward_threshold = reward_threshold

    def create_env(self, spec: TaskSpec) -> Any:
        raise NotImplementedError("Panda environment requires panda-gym installation")

    def describe(self, env: Any) -> Dict[str, Any]:
        obs_space = env.observation_space
        act_space = env.action_space

        return {
            "obs_shape": obs_space.shape,
            "n_actions": act_space.shape[0],
            "action_dim": act_space.shape[0],
            "env_type": "panda",
        }

    def is_compatible(self, spec: TaskSpec) -> bool:
        return "panda" in spec.name.lower()