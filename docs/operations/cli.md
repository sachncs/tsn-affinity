# CLI reference

TSN-Affinity ships three first-party CLIs plus an umbrella
`tsn-affinity` entry point that dispatches to them.

## `tsn-benchmark`

Run the synthetic benchmark suite across multiple strategies and runs.

```text
usage: tsn-benchmark [-h] [--strategies STRATEGIES [STRATEGIES ...]]
                     [--n-tasks N_TASKS] [--trajs-per-task TRAJS_PER_TASK]
                     [--train-steps TRAIN_STEPS] [--n-runs N_RUNS]
                     [--output OUTPUT] [--device DEVICE] [--seed SEED]
```

| Flag | Default | Description |
| --- | --- | --- |
| `--strategies` | `tsn_core tsn_affinity` | Strategies to benchmark. |
| `--n-tasks` | `5` | Number of tasks. |
| `--trajs-per-task` | `10` | Trajectories per task. |
| `--train-steps` | `200` | Optimizer steps per task. |
| `--n-runs` | `3` | Independent runs per strategy. |
| `--output` | `runs/benchmark` | Output directory. |
| `--device` | `cpu` | Torch device. |
| `--seed` | `42` | Random seed. |

## `tsn-atari`

Train and evaluate a TSN strategy on Atari games.

```text
usage: tsn-atari [-h] [--strategy {tsn_affinity,tsn_core}]
                 [--output OUTPUT] [--device DEVICE]
                 [--n-trajectories N_TRAJECTORIES] [--max-steps MAX_STEPS]
                 [--train-steps TRAIN_STEPS] [--eval-rollouts EVAL_ROLLOUTS]
                 [--seed SEED]
```

| Flag | Default | Description |
| --- | --- | --- |
| `--strategy` | `tsn_affinity` | Strategy name. |
| `--output` | `runs/atari_tsn_affinity` | Output directory. |
| `--device` | `cpu` | Torch device. |
| `--n-trajectories` | `10` | Trajectories per game. |
| `--max-steps` | `2000` | Max steps per trajectory. |
| `--train-steps` | `2000` | Optimizer steps per task. |
| `--eval-rollouts` | `3` | Greedy rollouts per evaluation. |
| `--seed` | `0` | Random seed. |

## `tsn-atari-collect`

Collect random-policy Atari trajectories and save a manifest.

```text
usage: tsn-atari-collect [-h] --game GAME [--n-trajectories N_TRAJECTORIES]
                         [--max-steps MAX_STEPS] [--seed SEED] --output OUTPUT
```

| Flag | Default | Description |
| --- | --- | --- |
| `--game` | *(required)* | Atari game name (bare, for example `Breakout`). |
| `--n-trajectories` | `10` | Trajectories to collect. |
| `--max-steps` | `2000` | Max steps per trajectory. |
| `--seed` | `0` | Random seed. |
| `--output` | *(required)* | Output directory. |

## `tsn-panda`

Train a TSN strategy on Panda continuous-control data.

```text
usage: tsn-panda [-h] --data DATA [--strategy STRATEGY] [--output OUTPUT]
                 [--device DEVICE] [--train-steps TRAIN_STEPS] [--seed SEED]
```

| Flag | Default | Description |
| --- | --- | --- |
| `--data` | *(required)* | Path to the Panda offline pickle file. |
| `--strategy` | `tsn_affinity` | Strategy name. |
| `--output` | `runs/panda_tsn_affinity` | Output directory. |
| `--device` | `cuda` | Torch device. |
| `--train-steps` | `2000` | Optimizer steps per task. |
| `--seed` | `0` | Random seed. |

## Common patterns

### Custom output directory

All CLIs accept `--output` to redirect results:

```bash
tsn-benchmark --output results/2026-09-09/quick-test
```

### Reproducibility

All CLIs accept `--seed`. Combine it with `TSN_SEED` (an environment
variable read by `TrainingService`) to make Python-side runs
reproducible too.

### CI integration

The CLI exits with status `0` on success and non-zero on failure, so
they slot into CI pipelines without additional glue:

```yaml
- name: Smoke benchmark
  run: tsn-benchmark --strategies tsn_core --n-tasks 3 --train-steps 50 --n-runs 1 --output runs/ci
```
