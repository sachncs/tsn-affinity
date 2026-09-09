# Add a new environment

TSN-Affinity ships with built-in adapters for Atari games. To plug
in a different family of environments — robotics simulators, custom
benchmarks, or your own task suite — implement the `BaseEnvAdapter`
protocol and register it with `TaskRegistry`.

## Define a `TaskSpec`

`TaskSpec` carries the information an adapter needs to instantiate an
environment:

```python
from tsn_affinity.benchmarks import TaskSpec

spec = TaskSpec(
    name="panda_reach_v0",
    env_id=None,
    seed=0,
    params={"robot": "franka", "task": "reach"},
)
```

`name` is human-readable and shown in logs. `env_id` is reserved for
gymnasium-style identifiers; custom adapters can ignore it. `seed`
propagates to the adapter's environment creation so runs are
reproducible. `params` is a free-form keyword bag.

## Implement an adapter

```python
from typing import Any
from tsn_affinity.benchmarks import TaskSpec

class PandaReachAdapter:
    env_type = "panda"

    def is_compatible(self, spec: TaskSpec) -> bool:
        return "reach" in spec.name.lower()

    def create_env(self, spec: TaskSpec) -> Any:
        import gymnasium as gym
        import panda_gym
        env = gym.make("PandaReach-v3")
        env.reset(seed=int(spec.seed))
        return env

    def describe(self, env: Any) -> dict[str, Any]:
        return {
            "obs_shape": tuple(env.observation_space.shape),
            "action_dim": int(env.action_space.shape[0]),
            "continuous_actions": True,
            "env_type": self.env_type,
        }
```

## Register and use

```python
from tsn_affinity.benchmarks import TaskRegistry

TaskRegistry().register("panda_reach", PandaReachAdapter())
adapter = TaskRegistry().get("panda_reach")
print(adapter.describe(adapter.create_env(spec)))
```

The registry is a process-wide singleton. Register adapters at import
time (typically in your project's `__init__.py`) so they are
discoverable everywhere.

## Plug into the affinity router

The router does not need to know about specific environments; it
works on whatever trajectory objects you pass it. To train a
`TSNAffinityStrategy` on your environment, build a list of
`Trajectory` objects and call `strategy.train_task(...)` as usual.

```python
from tsn_affinity import (
    RoutingConfig,
    SparseConfig,
    ModelConfig,
    TSNAffinityStrategy,
)
from tsn_affinity.data.schemas.trajectory import Trajectory

strategy = TSNAffinityStrategy(
    obs_shape=spec.params["obs_shape"],
    n_actions=spec.params["action_dim"],
    seq_len=20,
    device="cuda",
    model_config=ModelConfig(d_model=128, n_layers=3, n_heads=4),
    sparse_config=SparseConfig(keep_ratio=0.3),
    affinity_config=RoutingConfig(mode="hybrid"),
    seed=0,
)

trajectories = collect_trajectories(adapter, spec, n_trajectories=10)
strategy.train_task(trajectories, steps=2000, batch_size=64)
strategy.after_task(trajectories)
```

## Inspect registered adapters

```python
TaskRegistry().list_adapters()  # -> ["atari", "panda", "panda_reach", ...]
```

Use this to verify your registration took effect and to detect
accidental shadowing.
