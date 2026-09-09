.PHONY: help install dev test lint format typecheck clean build publish docs

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────────

install: ## Install package in current environment
	pip install -e .

dev: ## Install package with dev and atari extras
	pip install -e ".[dev,atari]"
	pre-commit install

# ── Quality ──────────────────────────────────────────────────────────────────

test: ## Run test suite
	pytest

test-cov: ## Run tests with HTML coverage report
	pytest --cov=tsn_affinity --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint: ## Run ruff linter
	ruff check tsn_affinity/ tests/

lint-fix: ## Run ruff linter with auto-fix
	ruff check --fix tsn_affinity/ tests/

format: ## Format code with ruff
	ruff format tsn_affinity/ tests/

format-check: ## Check formatting without modifying
	ruff format --check tsn_affinity/ tests/

typecheck: ## Run mypy type checking
	mypy tsn_affinity/

check: lint format-check typecheck ## Run all checks (lint + format + typecheck)

# ── Build & Publish ──────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .mypy_cache .ruff_cache .pytest_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

build: clean ## Build source and wheel distributions
	python -m build

publish: build ## Publish to PyPI (use TWINE_PASSWORD for auth)
	twine upload dist/*

publish-test: build ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build: ## Build Docker image
	docker build -t tsn-affinity:latest .

docker-run: ## Run Docker container
	docker run --rm -it tsn-affinity:latest

# ── Docs ─────────────────────────────────────────────────────────────────────

docs: ## Serve docs locally with mkdocs
	mkdocs serve

docs-build: ## Build static docs site
	mkdocs build

docs-deploy: ## Deploy docs to GitHub Pages
	mkdocs gh-deploy --force

# ── Benchmarks ───────────────────────────────────────────────────────────────

benchmark: ## Run synthetic benchmark
	tsn-benchmark --strategies tsn_core tsn_affinity --n-tasks 5 --trajs-per-task 10 --train-steps 200 --n-runs 3 --output runs/benchmark

benchmark-atari: ## Run Atari benchmark
	tsn-atari --strategy tsn_affinity --output runs/atari --n-trajectories 10 --max-steps 2000 --train-steps 2000
