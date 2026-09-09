# Installation

This page covers every supported install path and what to do when
something goes wrong. If you only want the short version, use the
[getting started guide](../getting-started.md).

## Supported platforms

| Platform       | Python    | Torch      | Notes                          |
|----------------|-----------|------------|--------------------------------|
| Linux x86_64   | 3.10–3.12 | CPU or CUDA| Reference CI environment       |
| macOS arm64    | 3.10–3.12 | CPU        | Apple Silicon supported         |
| Windows x86_64 | 3.10–3.12 | CPU or CUDA| Tested in GitHub Actions runners|

## From PyPI

The simplest way to install the package:

```bash
pip install tsn-affinity
```

This pulls in PyTorch, NumPy, scikit-learn, and gymnasium. To run the
Atari benchmark you also need the ALE bindings, which live in the
optional `atari` extra:

```bash
pip install "tsn-affinity[atari]"
```

## From source

Clone the repository and install in editable mode if you intend to
modify the code:

```bash
git clone https://github.com/sachncs/tsn-affinity.git
cd tsn-affinity
pip install -e ".[dev,atari]"
```

The `[dev]` extra adds the linting, testing, and documentation tools
used by the CI pipeline. See [Contributing](../contributing.md) for the
full list.

## Verify the install

```bash
python -c "import tsn_affinity; print(tsn_affinity.__version__)"
tsn-benchmark --help
```

A successful run prints the version string and the CLI help text.

## CUDA setup

TSN-Affinity follows PyTorch's CUDA conventions. Pick a wheel that
matches your CUDA version, for example CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "tsn-affinity[atari]"
```

Then point the runtime at the GPU you want:

```bash
export TORCH_DEVICE=cuda:0
tsn-benchmark --strategies tsn_affinity --device cuda ...
```

## Apple Silicon

The `pip install` workflow is sufficient on macOS arm64; the package
falls back to PyTorch's MPS backend automatically when
`TORCH_DEVICE=mps` is set.

## Offline / air-gapped installs

Download wheels on a connected machine and install them with the
`--no-index` flag:

```bash
pip download tsn-affinity --dest wheels/
pip install --no-index --find-links wheels/ tsn-affinity
```

## Troubleshooting

| Symptom                                  | Likely cause                  | Fix                                   |
|------------------------------------------|-------------------------------|---------------------------------------|
| `ModuleNotFoundError: tsn_affinity`      | Install in different venv     | `which python && python -m pip list` |
| `ImportError: gymnasium`                 | Atari extra not installed     | `pip install 'tsn-affinity[atari]'`   |
| `torch.cuda.OutOfMemoryError`            | Sequence too long / batch too big | Lower `--batch-size` or `seq_len`     |
| `mkdocstrings` warnings                  | Docs toolchain not installed  | `pip install '.[dev]'`                |

Still stuck? Open an issue with the output of `python -c "import tsn_affinity, torch; print(torch.__version__)"`.
