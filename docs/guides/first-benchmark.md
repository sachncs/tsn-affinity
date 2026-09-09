# Your first benchmark

This walkthrough runs the canonical synthetic benchmark end to end
and explains every output file. It takes about a minute on a laptop.

## Step 1: Run the benchmark

```bash
tsn-benchmark \
    --strategies tsn_core tsn_affinity \
    --n-tasks 5 \
    --trajs-per-task 10 \
    --train-steps 200 \
    --n-runs 3 \
    --output runs/benchmark
```

What each flag does:

| Flag | Effect |
| --- | --- |
| `--strategies` | Which strategies to compare. |
| `--n-tasks` | Number of synthetic tasks in the sequence. |
| `--trajs-per-task` | Trajectories per task used for training. |
| `--train-steps` | Optimizer steps per task. |
| `--n-runs` | Independent runs per strategy for variance estimates. |
| `--output` | Directory to save results. |

The CLI also accepts `--seed` (default 42) and `--device` (default
`cpu`).

## Step 2: Read the output

The CLI prints a final summary that looks like:

```text
================================================================
FINAL SUMMARY
================================================================

tsn_core:
  ACC:    0.5388 +/- 0.0028
  BWT:    -0.0012 +/- 0.0015
  Forgetting: 0.0028 +/- 0.0007
  FWT:    -0.0012 +/- 0.0015
  Time/task: 70.58s +/- 3.74s

tsn_affinity:
  ACC:    0.4017 +/- 0.0041
  BWT:    -0.0008 +/- 0.0016
  Forgetting: 0.0026 +/- 0.0016
  FWT:    -0.0008 +/- 0.0016
  Time/task: 106.35s +/- 4.18s
```

## Step 3: Inspect the saved files

```text
runs/benchmark/
├── summary.json
├── tsn_core/
│   ├── run_0/
│   │   ├── metrics.json
│   │   ├── performance_matrix.npy
│   │   └── task_similarity.json
│   └── run_1/
│       └── …
└── tsn_affinity/
    └── …
```

- `summary.json` aggregates every run. Useful for plotting.
- `performance_matrix.npy` is a `[n_tasks, n_tasks]` numpy array.
- `task_similarity.json` records which copy each task was routed to
  and the affinity scores that drove the decision.

## Step 4: Interpret the numbers

ACC is the headline number: the average score on each task immediately
after training on that task. Higher is better.

BWT (backward transfer) measures how much learning new tasks hurts
previous tasks. Negative numbers mean the model is forgetting. A
negative BWT close to zero (less than 1% drop) means forgetting is
negligible.

Forgetting measures the same phenomenon as BWT but using the maximum
drop per task rather than the average.

FWT (forward transfer) measures how much training on task `i` helps
performance on tasks seen later. Positive numbers are a sign that
shared representations help.

!!! note "Why is `tsn_affinity` ACC lower than `tsn_core` in this run?"
    On the synthetic benchmark, `tsn_core` happens to score higher ACC
    because the routing overhead drags training time down without a
    compensating benefit on a small task suite. The affinity router
    shines on longer task sequences and visually distinct environments
    where routing meaningfully reuses knowledge. See
    [Routing modes](routing-modes.md) for guidance on when each
    strategy wins.

## Step 5: Plot the result

```python
import json
import matplotlib.pyplot as plt

with open("runs/benchmark/summary.json") as f:
    summary = json.load(f)

for strategy, metrics in summary.items():
    plt.bar(strategy, float(metrics["acc"].split()[0]), yerr=float(metrics["acc"].split()[-1]))
plt.ylabel("ACC")
plt.title("Synthetic benchmark")
plt.show()
```

For a deeper analysis, compute additional metrics with
[`StandardCLMetrics`](../api/benchmarks.md#tsn_affinity.benchmarks.metrics.StandardCLMetrics)
on the saved performance matrices.
