"""Centralized structured logging configuration.

Replaces all print statements with a configured logger.
Provides setup_logging() for CLI entry points.
"""

import logging
import sys


def setup_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    handler: logging.Handler | None = None,
) -> logging.Logger:
    """Configure structured logging for TSN-Affinity.

    Args:
        level: Logging level (default: INFO).
        format_string: Optional custom format string.
        handler: Optional handler to use instead of StreamHandler.

    Returns:
        Configured root logger for the package.
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logger = logging.getLogger("tsn_affinity")
    logger.setLevel(level)

    if not logger.handlers:
        if handler is None:
            handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
