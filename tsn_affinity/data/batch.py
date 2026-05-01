"""Minibatch generation (re-exported from loaders).

This module re-exports batch utilities from
:mod:`tsn_affinity.data.loaders.batch_loader` for backward
compatibility. New code should import directly from the loaders
subpackage.
"""

from tsn_affinity.data.loaders.batch_loader import (
    make_minibatches,
    masked_cross_entropy,
    masked_mse,
    unpack_batch_discrete,
)

__all__ = [
    "make_minibatches",
    "unpack_batch_discrete",
    "masked_cross_entropy",
    "masked_mse",
]
