# Reproducibility

Reproducibility is a first-class concern. This page explains what
TSN-Affinity does to make runs deterministic, and what you have to do
on your side.

## What the library controls

Every strategy constructor accepts a `seed` argument. When provided:

- `np.random.seed(seed)` is called before any random sampling.
- `torch.manual_seed(seed)` is called before model construction.
- `RoutingConfig.seed` (when set) seeds the affinity router and
  warm-starter as well.

To reproduce a run end to end, pass the same seed to every
constructor and CLI invocation:

```python
strategy = TSNAffinityStrategy(
    obs_shape=(4,),
    n_actions=2,
    seq_len=10,
    device="cpu",
    affinity_config=RoutingConfig(seed=0),
    seed=0,
)
```

```bash
tsn-benchmark --seed 42 --output runs/repro
```

## What you control

Reproducibility is the caller's responsibility for:

- **Hardware.** A run on CPU may not bit-identically match a run on
  CUDA, and even CUDA versions can drift between minor releases.
- **Library versions.** Pin `torch`, `gymnasium`, and `numpy` in your
  requirements file.
- **Parallelism.** Multiple workers in the data loader introduce
  nondeterminism. Set `num_workers=0` if you need bit-identical runs.
- **CUBLAS workspace size.** On CUDA, set
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` to disable heuristic selection.

## Random sources TSN-Affinity uses

The library relies on these random sources:

| Source | Used by | Seedable via |
| --- | --- | --- |
| `np.random` | trajectory sampling, mask initialization | `numpy.random.seed` |
| `torch` | model parameter initialization | `torch.manual_seed` |
| Local `numpy.random.Generator` in `TSNBaseStrategy` | per-task memory sampling | `seed` constructor argument |

Determinism in PyTorch is enabled by default in CPU mode. On CUDA,
set `torch.use_deterministic_algorithms(True)` to enforce fully
deterministic operations.

## Inspecting a saved run

Every CLI writes a `results.json` (or equivalent) next to the saved
matrices. The file records:

- The strategy name and full configuration.
- The seed used for the run (if any).
- All computed metrics, plus baseline citations where applicable.

Save the file alongside the run and reference it in any paper or
report that cites the numbers.

## Deterministic benchmarking

`tsn-benchmark` calls `np.random.seed` and `torch.manual_seed` at the
start of every run with `config.seed + seed_offset`. The `--n-runs`
flag triggers that many independent runs to surface variance. The
final summary reports the mean and standard deviation of every
metric.
