# API reference

The API reference is split into per-package pages so each section
stays focused. Use the navigation bar to jump to a specific package,
or browse from here.

## Public surface

TSN-Affinity exposes the following public packages. Every symbol
listed below is also re-exported from the top-level `tsn_affinity`
namespace, so `from tsn_affinity import DecisionTransformer` works
without navigating the package layout.

<div class="tsn-package-grid" markdown>

-   **`tsn_affinity.core`** — Model definitions, configs, exceptions,
    logging. See [Core](api/core.md).

-   **`tsn_affinity.sparse`** — Sparse layers and conversion
    utilities. See [Sparse layers](api/sparse.md).

-   **`tsn_affinity.routing`** — Affinity metrics and warm-starting.
    See [Routing](api/routing.md).

-   **`tsn_affinity.strategies`** — Continual learning strategies.
    See [Strategies](api/strategies.md).

-   **`tsn_affinity.data`** — Trajectory schemas, loaders, batch
    generation. See [Data](api/data.md).

-   **`tsn_affinity.benchmarks`** — Adapters, registries, metrics,
    Atari baselines. See [Benchmarks](api/benchmarks.md).

-   **`tsn_affinity.interfaces`** — Abstract protocols.
    See [Interfaces](api/interfaces.md).

-   **`tsn_affinity.run`** — Result analysis. Covered in
    [Benchmarks](api/benchmarks.md).

-   **`tsn_affinity.services`** — High-level orchestration. Covered
    in [Strategies](api/strategies.md).

-   **`tsn_affinity.cli`** — Command-line entry points. See
    [Operations → CLI](operations/cli.md).

</div>

## Top-level re-exports

The following symbols are re-exported from the top-level
`tsn_affinity` namespace. Importing from the top level avoids the
cost of importing submodules on hot paths.

=== "Version and core"

    ```python
    from tsn_affinity import __version__
    from tsn_affinity import (
        DecisionTransformer,
        DTBackbone,
        ObsEncoder,
    )
    ```

=== "Configs and exceptions"

    ```python
    from tsn_affinity import (
        ModelConfig,
        SparseConfig,
        RoutingConfig,
        TSNAffinityConfig,
        SparseConversionConfig,
        TSNAffinityError,
        RoutingError,
        MaskError,
        ConfigurationError,
        DataError,
        StrategyError,
        BenchmarkError,
        setup_logging,
    )
    ```

=== "Data and routing"

    ```python
    from tsn_affinity import (
        Trajectory,
        discount_cumsum,
        compute_action_affinity,
        compute_hybrid_affinity,
        compute_latent_affinity,
        compute_latent_affinity_batch,
        normalize_scores,
        MaskWarmstarter,
    )
    ```

=== "Sparse layers and strategies"

    ```python
    from tsn_affinity import (
        TSNSparseLinear,
        TSNSparseConv2d,
        TSNSparseEmbedding,
        TopKMaskSTE,
        convert_to_sparse,
        iter_sparse_modules,
        kmeans_quantize,
        rebuild_optimizer,
        BaseStrategy,
        TSNBaseStrategy,
        TSNCoreStrategy,
        TSNAffinityStrategy,
        TSNReplayKLStrategy,
        CopyManager,
        ModelCopy,
    )
    ```

## Per-package references

- [Core](api/core.md)
- [Sparse layers](api/sparse.md)
- [Routing](api/routing.md)
- [Strategies](api/strategies.md)
- [Data](api/data.md)
- [Benchmarks](api/benchmarks.md)
- [Interfaces](api/interfaces.md)
