# Run Atari experiments

The Atari CLI is the most reproducible way to evaluate a TSN-Affinity
strategy. It collects random-policy trajectories, trains on them, and
evaluates the trained policy with greedy rollouts. Human-normalized
scores use the standard ALE baselines.

## Install the Atari extra

```bash
pip install "tsn-affinity[atari]"
```

This pulls in the `gymnasium[accept-license-requests]` distribution
needed for the ALE environments.

## Collect trajectories

```bash
tsn-atari-collect \
    --game Breakout \
    --n-trajectories 20 \
    --max-steps 2000 \
    --output runs/breakout_random
```

The CLI writes a `manifest.json` file with the action, reward, and
return-to-go arrays for every trajectory.

## Train and evaluate

```bash
tsn-atari \
    --strategy tsn_affinity \
    --n-trajectories 20 \
    --max-steps 2000 \
    --train-steps 2000 \
    --eval-rollouts 3 \
    --seed 0 \
    --output runs/atari_tsn_affinity
```

The CLI writes three files under `runs/atari_tsn_affinity/`:

- `performance_matrix.npy` — human-normalized scores on the
  `[n_tasks, n_tasks]` performance matrix.
- `returns_matrix.npy` — raw return values per (task, eval_task)
  combination.
- `results.json` — metrics summary, baseline citations, and the
  recorded task similarity per task.

## Read the results

```python
import json
import numpy as np
from tsn_affinity.benchmarks.metrics import StandardCLMetrics

with open("runs/atari_tsn_affinity/results.json") as f:
    results = json.load(f)

print(results["strategy"], results["metrics"])
print("Random baselines:", results["random_baselines"])
print("Human baselines:", results["human_baselines"])

matrix = np.load("runs/atari_tsn_affinity/performance_matrix.npy")
cl = StandardCLMetrics(matrix)
print(cl.summary())
```

The summary block prints `ACC`, `BWT`, `Forgetting`, and `FWT` in
uppercase. Negative `BWT` means the model is forgetting previous
tasks; positive `FWT` means training on a task improves performance
on tasks seen later.

## Citation policy

The Atari baseline numbers bundled with TSN-Affinity come from the
Arcade Learning Environment (ALE) and the DQN paper:

- Mnih et al. (2015). *Human-level control through deep reinforcement
  learning.* Nature 518, 529–533.
- The Arcade Learning Environment (ALE) README, baseline tables.

The CLI records the citation in `results.json` under
`human_baselines_source` so downstream consumers do not lose the
provenance.

## Debugging

| Symptom                                          | Likely cause                                | Fix                                                      |
|--------------------------------------------------|---------------------------------------------|----------------------------------------------------------|
| `ModuleNotFoundError: gymnasium`                 | Atari extra not installed                   | `pip install 'tsn-affinity[atari]'`                      |
| `RuntimeError: ALE/Breakout-v5 not found`         | ALE bindings missing                        | `pip install ale-py`                                     |
| NaN losses                                       | LR too high or dataset too small            | Lower `--train-steps`, raise `--n-trajectories`          |
| Performance scores close to random               | Insufficient training                       | Increase `--train-steps` or `--n-trajectories`           |
