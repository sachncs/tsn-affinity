# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in TSN-Affinity, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email **sachncs@gmail.com** with:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Assessment**: We will assess the severity and validity within 5 business days
- **Fix**: We will work on a fix and coordinate disclosure with you
- **Disclosure**: We will publish a security advisory once a fix is available

## Security Best Practices

When using TSN-Affinity:

- **Dependencies**: Keep dependencies up to date (`pip install --upgrade`)
- **Virtual environments**: Always use a virtual environment
- **Data handling**: Be cautious when loading untrusted trajectory data
- **Model files**: Only load model checkpoints from trusted sources
- **GPU access**: Restrict GPU access in shared environments

## Scope

This security policy applies to:

- The `tsn-affinity` Python package
- Code in this repository
- Official CLI tools (`tsn-benchmark`, `tsn-atari`, `tsn-panda`)

This does **not** cover:

- Third-party libraries (report to their respective maintainers)
- Issues in the underlying PyTorch or Gymnasium frameworks
