# Contributing to TSN-Affinity

Thank you for considering contributing to TSN-Affinity! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branch Naming](#branch-naming)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Running Tests](#running-tests)
- [Documentation](#documentation)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/tsn-affinity.git
   cd tsn-affinity
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/sachncs/tsn-affinity.git
   ```
4. Create a branch from `master`:
   ```bash
   git checkout -b feat/my-feature master
   ```

## Development Setup

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev,atari]"

# Install pre-commit hooks
pre-commit install
```

## Branch Naming

Use descriptive branch names with a prefix:

| Prefix | Purpose |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `refactor/` | Code refactoring |
| `test/` | Adding or updating tests |
| `chore/` | Maintenance tasks |

Example: `feat/add-dmcontrol-adapter`

## Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Code style changes (formatting, no logic change) |
| `refactor` | Code refactoring (no feature or fix) |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks (deps, CI, config) |
| `perf` | Performance improvements |

### Examples

```
feat(routing): add adaptive threshold learning
fix(sparse): correct mask shape inference for 5D obs
docs: update installation instructions
test(strategies): add TSN-ReplayKL integration tests
chore: bump torch to >=2.1
```

## Pull Request Process

1. Ensure your branch is up to date with `master`:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```
2. Run the full test suite and linters:
   ```bash
   pytest
   ruff check tsn_affinity/ tests/
   ruff format --check tsn_affinity/ tests/
   mypy tsn_affinity/
   ```
3. Push your branch and open a Pull Request against `master`
4. Fill out the PR template completely
5. Request a review from a maintainer
6. Address review feedback with additional commits
7. Once approved, a maintainer will merge your PR

## Coding Standards

This project follows the **Google Python Style Guide**:

- Use type hints for all public function signatures
- Write docstrings for all public classes and functions (Google style)
- Keep functions focused and under 50 lines where possible
- Use descriptive variable names
- No `TODO` comments in production code without a linked issue

### Linting

We use `ruff` for linting and formatting:

```bash
# Check for lint errors
ruff check tsn_affinity/ tests/

# Auto-fix lint errors
ruff check --fix tsn_affinity/ tests/

# Check formatting
ruff format --check tsn_affinity/ tests/

# Auto-format
ruff format tsn_affinity/ tests/
```

### Type Checking

We use `mypy` for static type checking:

```bash
mypy tsn_affinity/
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/core/test_attention.py -v

# Run with coverage report
pytest --cov=tsn_affinity --cov-report=html

# Run slow tests
pytest -m slow

# Skip slow tests (default)
pytest -m "not slow"
```

## Documentation

- Update README.md if adding new features or changing usage
- Update or add relevant docs in `docs/`
- Add docstrings to all new public APIs
- Update CHANGELOG.md under the `[Unreleased]` section

## Questions?

Open a [Discussion](https://github.com/sachncs/tsn-affinity/discussions) on GitHub.
