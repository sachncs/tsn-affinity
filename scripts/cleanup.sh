#!/bin/bash
# Cleanup script for tsn-affinity project

set -e

echo "Cleaning up artifacts..."

# Cache and artifact directories to remove
find . -type d \( \
    -name "__pycache__" \
    -o -name ".pytest_cache" \
    -o -name ".ruff_cache" \
    -o -name ".mypy_cache" \
    -o -name ".coverage" \
    -o -name ".benchmarks" \
    -o -name "tsn_affinity.egg-info" \
\) -exec rm -rf {} + 2>/dev/null || true

# Remove top-level egg-info that find might miss
if [ -e "tsn_affinity.egg-info" ]; then
    echo "  Removing tsn_affinity.egg-info"
    rm -rf "tsn_affinity.egg-info"
fi

echo "Cleanup complete."