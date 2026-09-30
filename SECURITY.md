# Security Policy

## Supported Versions

Only the latest release receives security fixes.

| Version | Supported |
|---|---|
| 0.7.x (latest) | Yes |
| Older releases | No |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Use GitHub's private advisory system to report vulnerabilities confidentially:

1. Go to the [Security Advisories](https://github.com/modelship-ai/modelship/security/advisories) page
2. Click **Report a vulnerability**
3. Fill in the details — affected component, reproduction steps, potential impact

You can expect an initial response within **72 hours**. If a fix is warranted, a patched release will be published and the advisory made public once the fix is available.

## Scope

This project is a self-hosted server intended to run on private infrastructure. Relevant security concerns include:

- Unauthenticated API access — the gateway doesn't authenticate callers; users are expected to put an auth layer in front of it or restrict it at the network level
- Dependency vulnerabilities in vLLM, Ray, or other upstream packages
- Docker image vulnerabilities
- Prompt injection or model abuse vectors if the API is publicly exposed
