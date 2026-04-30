"""Module conversion utilities for TSN sparse layers."""

from dataclasses import dataclass
from typing import Dict, Generator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from tsn_affinity.sparse.sparse_conv2d import TSNSparseConv2d
from tsn_affinity.sparse.sparse_embedding import TSNSparseEmbedding
from tsn_affinity.sparse.sparse_linear import TSNSparseLinear


@dataclass
class SparseConversionConfig:
    """Configuration for sparse module conversion.

    Attributes:
        keep_ratio: Fraction of weights to keep per task.
        include_embeddings: Whether to convert embedding layers.
        allow_weight_reuse: Whether to allow weight reuse across tasks.
        skip_module_names: Module name prefixes to skip during conversion.
        quant_clusters: Number of k-means clusters for post-task quantization.
        quantize_after_task: Whether to quantize new weights after each task.
    """

    keep_ratio: float = 0.5
    include_embeddings: bool = True
    allow_weight_reuse: bool = False
    skip_module_names: tuple = ("dt.te",)
    quant_clusters: int = 16
    quantize_after_task: bool = True


def iter_sparse_modules(
    model: nn.Module,
) -> Generator[Tuple[str, nn.Module], None, None]:
    """Iterate over all TSN sparse modules in a model.

    Args:
        model: PyTorch model to search.

    Yields:
        Tuples of (module_name, module) for each TSN sparse layer.
    """
    for name, module in model.named_modules():
        if isinstance(module, (TSNSparseLinear, TSNSparseConv2d, TSNSparseEmbedding)):
            yield name, module


def convert_to_sparse(model: nn.Module, config: SparseConversionConfig) -> None:
    """Convert standard PyTorch layers to TSN sparse versions in-place.

    Converts Linear -> TSNSparseLinear, Conv2d -> TSNSparseConv2d,
    and optionally Embedding -> TSNSparseEmbedding.

    Args:
        model: PyTorch model to convert.
        config: SparseConversionConfig with conversion parameters.
    """
    for name, child in list(model.named_children()):
        if name in config.skip_module_names:
            continue

        new_child: Optional[nn.Module] = None

        if isinstance(child, nn.Linear) and not isinstance(child, TSNSparseLinear):
            new_child = TSNSparseLinear(
                child.in_features,
                child.out_features,
                bias=(child.bias is not None),
                keep_ratio=config.keep_ratio,
                allow_weight_reuse=config.allow_weight_reuse,
            )
            with torch.no_grad():
                new_child.weight.copy_(child.weight)
                if child.bias is not None and new_child.bias is not None:
                    new_child.bias.copy_(child.bias)

        elif isinstance(child, nn.Conv2d) and not isinstance(child, TSNSparseConv2d):
            new_child = TSNSparseConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None),
                padding_mode=child.padding_mode,
                keep_ratio=config.keep_ratio,
                allow_weight_reuse=config.allow_weight_reuse,
            )
            with torch.no_grad():
                new_child.weight.copy_(child.weight)
                if child.bias is not None and new_child.bias is not None:
                    new_child.bias.copy_(child.bias)

        elif (
            config.include_embeddings
            and isinstance(child, nn.Embedding)
            and not isinstance(child, TSNSparseEmbedding)
        ):
            new_child = TSNSparseEmbedding(
                child.num_embeddings,
                child.embedding_dim,
                padding_idx=child.padding_idx,
                max_norm=child.max_norm,
                norm_type=child.norm_type,
                scale_grad_by_freq=child.scale_grad_by_freq,
                sparse=child.sparse,
                keep_ratio=config.keep_ratio,
                allow_weight_reuse=config.allow_weight_reuse,
            )
            with torch.no_grad():
                new_child.weight.copy_(child.weight)

        if new_child is not None:
            setattr(model, name, new_child)
            child = new_child

        convert_to_sparse(child, config)


def kmeans_quantize(
    tensor: torch.Tensor,
    select_mask: torch.Tensor,
    n_clusters: int,
) -> Optional[np.ndarray]:
    """Apply k-means quantization to selected weight elements.

    Args:
        tensor: Weight tensor to quantize in-place.
        select_mask: Boolean mask of elements to quantize.
        n_clusters: Number of k-means clusters.

    Returns:
        Cluster centroids as numpy array, or None if quantization not possible.
    """
    select_mask = select_mask.to(device=tensor.device, dtype=torch.bool)
    flat_t = tensor.detach().view(-1)
    flat_m = select_mask.view(-1)

    n_selected = int(flat_m.sum().item())
    if n_selected < 2:
        return None

    selected = flat_t[flat_m].detach().cpu().numpy().reshape(-1, 1)
    unique_vals = np.unique(selected)
    actual_clusters = int(
        min(max(1, n_clusters), n_selected, len(unique_vals))
    )
    if actual_clusters < 2:
        return None

    kmeans = KMeans(n_clusters=actual_clusters, n_init=4, random_state=0)
    labels = kmeans.fit_predict(selected)
    centers = kmeans.cluster_centers_.reshape(-1)

    quantized = flat_t.detach().cpu().numpy().copy()
    selected_indices = np.where(flat_m.detach().cpu().numpy())[0]
    quantized[selected_indices] = centers[labels]

    with torch.no_grad():
        tensor.copy_(torch.from_numpy(quantized).view_as(tensor).to(
            device=tensor.device, dtype=tensor.dtype
        ))

    return centers


def rebuild_optimizer(
    old_opt: torch.optim.Optimizer,
    params,
) -> torch.optim.Optimizer:
    """Re-create an optimizer of the same family with same hyperparameters.

    Supports AdamW, Adam, and SGD.

    Args:
        old_opt: Original optimizer to copy configuration from.
        params: New parameter group to optimize.

    Returns:
        New optimizer with same configuration as old_opt.
    """
    if isinstance(old_opt, torch.optim.AdamW):
        pg = old_opt.param_groups[0]
        return torch.optim.AdamW(
            params,
            lr=float(pg.get("lr", 3e-4)),
            weight_decay=float(pg.get("weight_decay", 0.0)),
            betas=tuple(pg.get("betas", (0.9, 0.999))),
            eps=float(pg.get("eps", 1e-8)),
        )
    if isinstance(old_opt, torch.optim.Adam):
        pg = old_opt.param_groups[0]
        return torch.optim.Adam(
            params,
            lr=float(pg.get("lr", 3e-4)),
            weight_decay=float(pg.get("weight_decay", 0.0)),
            betas=tuple(pg.get("betas", (0.9, 0.999))),
            eps=float(pg.get("eps", 1e-8)),
        )
    if isinstance(old_opt, torch.optim.SGD):
        pg = old_opt.param_groups[0]
        return torch.optim.SGD(
            params,
            lr=float(pg.get("lr", 1e-3)),
            weight_decay=float(pg.get("weight_decay", 0.0)),
            momentum=float(pg.get("momentum", 0.0)),
            nesterov=bool(pg.get("nesterov", False)),
        )
    return torch.optim.AdamW(params, lr=3e-4)