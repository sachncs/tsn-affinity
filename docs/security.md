# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.4.x | Yes |
| 0.3.x | Yes |
| < 0.3 | No |

## Reporting a vulnerability

If you discover a security vulnerability in TSN-Affinity, please
report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Email **sachncs@gmail.com** with:

- A description of the vulnerability.
- Steps to reproduce the issue.
- Potential impact.
- A suggested fix (if any).

### What to expect

- **Acknowledgment** — within 48 hours.
- **Assessment** — within 5 business days.
- **Fix** — coordinated disclosure.
- **Disclosure** — a public security advisory once a fix is
  available.

## Security best practices

- **Dependencies** — keep dependencies up to date (`pip install --upgrade`).
- **Virtual environments** — always use a virtual environment.
- **Data handling** — be cautious when loading untrusted trajectory
  data; the `load_panda_offline_pkl` helper uses `pickle` and can
  execute arbitrary code.
- **Model files** — only load model checkpoints from trusted sources.
- **GPU access** — restrict GPU access in shared environments.

## Scope

This security policy applies to:

- The `tsn-affinity` Python package.
- Code in this repository.
- Official CLI tools (`tsn-benchmark`, `tsn-atari`, `tsn-atari-collect`,
  `tsn-panda`).

It does **not** cover third-party libraries (report to their
maintainers) or issues in PyTorch / Gymnasium.
