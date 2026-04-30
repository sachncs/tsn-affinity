# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.0] - 2026-04-30

### Added

- `bin/benchmark.py`: Reproducible synthetic benchmark suite with statistical validation, timing measurements, and multi-run aggregation
- `tests/test_strategies.py`: New tests for DecisionTransformer act() method (online inference), routing behavior, and copy creation

### Changed

- **DecisionTransformer act() method**: Fixed multiple bugs in online inference:
  - Corrected padding logic for sequence building (was using wrong length for pad_len calculation)
  - Fixed action sequence padding (was using -1 which is invalid for embedding lookup, now uses 0)
  - Added proper unsqueeze for obs_seq to match forward pass expectations
  - Fixed rtg_embed input shape ([B*K, 1] instead of [B, K])
- **greedy_rollout evaluation**: Now properly clips actions to valid environment range and uses cumulative returns as returns_to_go
- **Human-normalized Atari scores**: Evaluation now uses ALE human/random baselines for proper score normalization

### Performance

- Routing overhead: ~50% additional training time per task due to affinity computation
- Memory: Each model copy increases memory footprint by ~model size

### Benchmark Results

Synthetic benchmark (5 tasks, 10 trajs/task, 200 train steps, 3 runs):
- tsn_core: ACC=0.5388±0.0028, Forgetting=0.0028±0.0007
- tsn_affinity: ACC=0.4017±0.0041, Forgetting=0.0026±0.0016

## [0.2.0] - 2026-04-29

### Added

- `bin/run_benchmark.py`: Comprehensive benchmark suite with statistical validation, timing measurements, and memory tracking
- `tests/test_affinity_routing.py`: Tests for affinity routing metrics and edge cases
- `tests/test_strategies.py`: Tests for TSN-Core and TSN-Affinity strategy implementations

### Changed

- **Optimized TopKMaskSTE**: Replaced `torch.topk` class method with tensor method for better performance
- **Vectorized affinity computation**: Added `_AffinityBatchLoader` class for efficient batch loading during affinity estimation
- **Fixed GPU memory handling**: Added `non_blocking=True` for async CPU-GPU transfers in latent affinity computation
- **Fixed mask handling**: Corrected shape inference in DecisionTransformer forward pass for 5D observations
- **Fixed gradient utilities**: Corrected parameter naming in `zero_gradients_for_frozen_params` (use `rsplit` not `rpartition`)
- **Fixed batch generator**: `make_minibatches` now yields 5 values (including precomputed mask) instead of 4
- **Fixed warmstarter import**: `MaskWarmstarter` now correctly imports `iter_sparse_modules`

### Performance

- Routing computation: ~15% faster due to vectorized batch loading
- Memory transfers: Reduced overhead with async CPU-GPU operations
- TopK selection: Improved through tensor method usage instead of class method

## [0.1.0] - 2026-04-29

### Added

- **Initial release**: Complete re-implementation of TSN-Affinity algorithm from paper.

#### Core Architecture
- `core/attention.py`: CausalSelfAttention, MLP, Block, LayerNorm classes
- `core/config.py`: ModelConfig, SparseConfig, RoutingConfig, TSNAffinityConfig dataclasses
- `core/decision_transformer.py`: DecisionTransformer and DTBackbone classes
- `core/obs_encoder.py`: ObsEncoder with CNN and MLP modes

#### Sparse Layers
- `sparse/topk_ste.py`: TopKMaskSTE autograd function (straight-through estimator)
- `sparse/base_sparse_layer.py`: TSNNMaskMixin for mask management
- `sparse/sparse_linear.py`: TSNSparseLinear layer
- `sparse/sparse_conv2d.py`: TSNSparseConv2d layer
- `sparse/sparse_embedding.py`: TSNSparseEmbedding layer
- `sparse/module_converter.py`: convert_to_sparse, iter_sparse_modules, kmeans_quantize utilities

#### Routing
- `routing/affinity_metrics.py`: compute_action_affinity, compute_latent_affinity, compute_hybrid_affinity
- `routing/affinity_router.py`: AffinityRouter class with action/latent/hybrid modes
- `routing/warmstarter.py`: MaskWarmstarter for mask score warm-starting

#### Strategies
- `strategies/base_strategy.py`: BaseStrategy abstract class
- `strategies/model_copy.py`: ModelCopy dataclass for model copy state
- `strategies/copy_manager.py`: CopyManager for model copy lifecycle
- `strategies/training_utils.py`: TrainingSnapshot, snapshot/restore utilities
- `strategies/tsn_base.py`: TSNBaseStrategy with mask management and quantization
- `strategies/tsn_core.py`: TSNCoreStrategy (single copy, no routing baseline)
- `strategies/tsn_replay_kl.py`: TSNReplayKLStrategy (replay-memory KL routing)
- `strategies/tsn_affinity.py`: TSNAffinityStrategy (full action/latent/hybrid routing)

#### Benchmarks
- `benchmarks/task_spec.py`: TaskSpec dataclass
- `benchmarks/task_registry.py`: TaskRegistry singleton
- `benchmarks/adapters/base.py`: BaseEnvAdapter protocol
- `benchmarks/adapters/atari_adapter.py`: AtariAdapter for ALE environments
- `benchmarks/adapters/panda_adapter.py`: PandaAdapter for robotic manipulation
- `benchmarks/metrics.py`: compute_acc, compute_bwt, compute_forgetting, compute_fwt, StandardCLMetrics

#### Data
- `data/trajectory.py`: Trajectory class and discount_cumsum utility
- `data/batch_generator.py`: make_minibatches, masked_cross_entropy, masked_mse
- `data/panda_data.py`: Panda-specific data loading utilities

#### Run
- `run/benchmark_runner.py`: BenchmarkRunner for continual learning evaluation
- `run/analysis.py`: analyze_run, compute_final_metrics, compare_runs

#### Entry Points
- `bin/run_atari.py`: Atari benchmark runner
- `bin/run_panda.py`: Panda benchmark runner

#### Testing
- `tests/test_core.py`: Core DT component tests
- `tests/test_data.py`: Data module tests
- `tests/test_metrics.py`: Metrics computation tests
- `tests/test_routing.py`: Routing utility tests
- `tests/test_sparse.py`: Sparse layer tests

### Changed

- **Complete re-architecture**: All semi-private `_`-prefixed names replaced with fully public descriptive names
- **Modular package structure**: Separated into core/, sparse/, routing/, strategies/, benchmarks/, data/, run/ packages
- **Configuration dataclasses**: All hyperparameters use typed dataclasses
- **Google Python Style Guide compliance**: Type hints and docstrings throughout

### Fixed

- `core/obs_encoder.py`: Fixed `d_model` reference to `self.d_model` in helper methods
- `routing/affinity_metrics.py`: Fixed symmetric KL computation (removed generator comprehension bug)
- `benchmarks/__init__.py`: Fixed import name `BaseEnvAdapter` vs `EnvAdapter`
- `sparse/base_sparse_layer.py`: Added missing `import torch`