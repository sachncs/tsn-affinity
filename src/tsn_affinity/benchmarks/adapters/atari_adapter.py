"""Atari environment adapter for ALE tasks."""

from typing import Any, Dict, Tuple

import gymnasium as gym

from tsn_affinity.benchmarks.adapters.base import BaseEnvAdapter
from tsn_affinity.benchmarks.task_spec import TaskSpec


class AtariAdapter(BaseEnvAdapter):
    """Environment adapter for Atari games via ALE.

    Handles preprocessing: grayscale conversion, frame stacking,
    reward clipping, sticky actions, etc.

    Attributes:
        frameskip: Number of frames to skip.
        noop_max: Max no-ops at start.
        sticky_actions: Whether to use sticky action probability.
        repeat_action_prob: Probability of repeating last action.
        terminal_on_life_loss: End episode on life loss.
        clip_rewards: Whether to clip rewards to [-1, 1].
        grayscale_obs: Whether to convert to grayscale.
        scale_obs: Whether to scale observations to [0, 1].
        frame_stack: Number of frames to stack.
    """

    def __init__(
        self,
        frameskip: int = 4,
        noop_max: int = 30,
        sticky_actions: bool = True,
        repeat_action_prob: float = 0.25,
        terminal_on_life_loss: bool = True,
        clip_rewards: bool = True,
        grayscale_obs: bool = True,
        scale_obs: bool = False,
        frame_stack: int = 4,
    ) -> None:
        self.frameskip = frameskip
        self.noop_max = noop_max
        self.sticky_actions = sticky_actions
        self.repeat_action_prob = repeat_action_prob
        self.terminal_on_life_loss = terminal_on_life_loss
        self.clip_rewards = clip_rewards
        self.grayscale_obs = grayscale_obs
        self.scale_obs = scale_obs
        self.frame_stack = frame_stack

    def create_env(self, spec: TaskSpec) -> Any:
        params = spec.params

        env = gym.make(params.get("game", spec.name), frameskip=self.frameskip)

        return env

    def describe(self, env: Any) -> Dict[str, Any]:
        obs_space = env.observation_space
        act_space = env.action_space

        if hasattr(obs_space, "shape"):
            obs_shape = tuple(obs_space.shape)
        else:
            obs_shape = obs_space.shape

        return {
            "obs_shape": obs_shape,
            "n_actions": act_space.n if hasattr(act_space, "n") else act_space.shape[0],
            "env_type": "atari",
        }

    def is_compatible(self, spec: TaskSpec) -> bool:
        return "ALE" in str(spec.params.get("game", ""))