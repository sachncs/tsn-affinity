# syntax=docker/dockerfile:1

# ── Base ─────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── Dependencies ─────────────────────────────────────────────────────────────
FROM base AS deps

COPY pyproject.toml README.md LICENSE ./
COPY tsn_affinity/__init__.py tsn_affinity/__init__.py

RUN pip install --no-cache-dir .

# ── Full install (with dev and atari extras) ────────────────────────────────
FROM deps AS full

COPY . .

RUN pip install --no-cache-dir -e ".[dev,atari]"

# ── Production (no dev tools) ───────────────────────────────────────────────
FROM base AS production

COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY tsn_affinity/ tsn_affinity/
COPY configs/ configs/

ENTRYPOINT ["python", "-m", "tsn_affinity.cli"]
CMD ["--help"]
