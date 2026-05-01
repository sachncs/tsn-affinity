"""Centralized exception hierarchy for TSN-Affinity.

All domain-specific exceptions inherit from TSNAffinityError
to enable targeted error handling across the codebase.
"""


class TSNAffinityError(Exception):
    """Base exception for all TSN-Affinity errors."""


class RoutingError(TSNAffinityError):
    """Raised when affinity routing computation fails."""


class MaskError(TSNAffinityError):
    """Raised when sparse mask operations are invalid."""


class ConfigurationError(TSNAffinityError):
    """Raised when configuration values are invalid or incompatible."""


class DataError(TSNAffinityError):
    """Raised when data loading or batch generation fails."""


class StrategyError(TSNAffinityError):
    """Raised when a strategy operation fails."""


class BenchmarkError(TSNAffinityError):
    """Raised when benchmark execution fails."""
