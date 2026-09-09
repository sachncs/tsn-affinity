# Routing Module

Affinity metrics and routing decisions for model copy selection.

## Components

- `metrics.py`: Action affinity, latent affinity, hybrid scores, normalization
- `router.py`: AffinityRouter for copy selection
- `warmstarter.py`: MaskWarmstarter for accelerating new tasks

## Extension Points

Add new affinity metrics in `metrics.py` and wire them into `AffinityRouter`.
