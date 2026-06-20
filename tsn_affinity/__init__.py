"""TSN-Affinity: Training-Aware Sparse Networks with Affinity Routing."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tsn-affinity")
except PackageNotFoundError:
    __version__ = "0.0.0"
