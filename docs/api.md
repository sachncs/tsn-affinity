# TSN-Affinity

Training-Aware Sparse Networks with Affinity Routing for Continual Offline RL.

## Installation

```bash
pip install tsn-affinity
```

## Quick Start

```bash
tsn-benchmark --strategies tsn_core tsn_affinity --n-tasks 5
```

## API Reference

### Core

::: tsn_affinity.core.config
::: tsn_affinity.core.decision_transformer
::: tsn_affinity.core.attention
::: tsn_affinity.core.encoder

### Sparse Layers

::: tsn_affinity.sparse.base
::: tsn_affinity.sparse.linear
::: tsn_affinity.sparse.conv2d
::: tsn_affinity.sparse.embedding
::: tsn_affinity.sparse.topk

### Routing

::: tsn_affinity.routing.metrics
::: tsn_affinity.routing.warmstarter

### Strategies

::: tsn_affinity.strategies.base
::: tsn_affinity.strategies.tsn_core
::: tsn_affinity.strategies.tsn_affinity
::: tsn_affinity.strategies.tsn_replay_kl
::: tsn_affinity.strategies.copy_manager

### Data

::: tsn_affinity.data.trajectory
