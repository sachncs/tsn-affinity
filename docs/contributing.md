# Contributing

Thanks for your interest in contributing to TSN-Affinity! This page
covers the workflow, conventions, and review process. The full
checklist lives in [CONTRIBUTING.md](https://github.com/sachncs/tsn-affinity/blob/master/CONTRIBUTING.md).

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Getting started](#getting-started)
- [Development setup](#development-setup)
- [Branch naming](#branch-naming)
- [Commit conventions](#commit-conventions)
- [Pull request process](#pull-request-process)
- [Coding standards](#coding-standards)
- [Running tests](#running-tests)
- [Documentation](#documentation)

## Code of conduct

This project follows the [Contributor Covenant](code-of-conduct.md).
By participating you agree to uphold its terms.

## Getting started

1. Fork the repository on GitHub.
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

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
pip install -e ".[dev,atari]"
pre-commit install
```

The `[dev]` extra pulls in `pytest`, `ruff`, `mypy`, `mkdocs`, and
`mkdocstrings`. The `[atari]` extra pulls in `gymnasium` and the ALE
bindings for the Atari CLI.

## Branch naming

Use a descriptive prefix:

| Prefix | Purpose |
| --- | --- |
| `feat/` | New features. |
| `fix/` | Bug fixes. |
| `docs/` | Documentation changes. |
| `refactor/` | Code refactoring. |
| `test/` | Adding or updating tests. |
| `chore/` | Maintenance tasks. |

Example: `feat/add-dmcontrol-adapter`.

## Commit conventions

TSN-Affinity follows [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

| Type | Description |
| --- | --- |
| `feat` | A new feature. |
| `fix` | A bug fix. |
| `docs` | Documentation only. |
| `style` | Formatting, no logic change. |
| `refactor` | Code refactoring. |
| `test` | Adding or updating tests. |
| `chore` | Maintenance tasks. |
| `perf` | Performance improvements. |

## Pull request process

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
3. Push your branch and open a Pull Request against `master`.
4. Fill out the PR template completely.
5. Request a review from a maintainer.
6. Address review feedback with additional commits.
7. Once approved, a maintainer will merge your PR.

## Coding standards

TSN-Affinity follows the **Google Python Style Guide**:

- Use type hints for all public function signatures.
- Write docstrings for all public classes and functions (Google
  style).
- Keep functions focused and short.
- Use descriptive variable names.
- No `TODO` comments in production code without a linked issue.

### Linting

```bash
ruff check tsn_affinity/ tests/
ruff check --fix tsn_affinity/ tests/
ruff format tsn_affinity/ tests/
```

### Type checking

```bash
mypy tsn_affinity/
```

## Running tests

```bash
pytest                         # All tests
pytest -m "not slow"            # Skip slow tests
pytest -m gpu                  # CUDA-only tests
pytest --cov=tsn_affinity      # With coverage
```

## Documentation

- Update `README.md` if you change the public API or installation
  steps.
- Update or add the relevant page under `docs/`.
- Add docstrings to every new public function or class.
- Add an entry to `CHANGELOG.md` under the `[Unreleased]` section.

To preview the docs site locally:

```bash
mkdocs serve
```

Then open `http://localhost:8000` in a browser.

## Questions?

Open a
[Discussion](https://github.com/sachncs/tsn-affinity/discussions) on
GitHub.
