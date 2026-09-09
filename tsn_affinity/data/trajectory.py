"""Trajectory data structure (re-exported from schemas).

This module re-exports :class:`Trajectory` and :func:`discount_cumsum`
from :mod:`tsn_affinity.data.schemas.trajectory` for backward
compatibility. New code should import directly from the schemas
subpackage.
"""

from tsn_affinity.data.schemas.trajectory import Trajectory, discount_cumsum

__all__ = ["Trajectory", "discount_cumsum"]
