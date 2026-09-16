# Contributing to Modelship

Thanks for your interest in contributing to Modelship! This document covers the basics for getting started.

## License

Modelship is licensed under [Apache 2.0](LICENSE), and **will always remain available under Apache 2.0** — that promise isn't contingent on the CLA below.

Per Apache 2.0 §5, any contribution you intentionally submit for inclusion is licensed to the project under Apache 2.0 by default. Before your first PR can be merged, you'll also need to sign the [Contributor License Agreement](CLA.md) (individual or corporate) via the CLA Assistant bot, which will comment on your PR with a link. You keep copyright in your own contributions — the CLA grants the project the right to distribute and, if ever needed, relicense them, it does not assign them away.

## Development Setup

The recommended way to develop is via the VS Code Dev Container. See [docs/development.md](docs/development.md) for full instructions.

**Quick version:**

1. Install Docker + NVIDIA Container Toolkit
2. Set the `HF_TOKEN` environment variable
3. Open the project in VS Code and select **Dev Containers: Reopen in Container**
4. Inside the container:

```bash
uv sync --extra dev
uv run mship_deploy.py   # starts its own Ray head, auto-detecting CPUs/GPUs
```

## Code Style

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [Pyright](https://github.com/microsoft/pyright) for type checking. Both run in CI on every pull request.

```bash
make lint        # ruff check + ruff format --check + pyright — all three must pass
make lint-fix    # auto-fix ruff issues
```

`make lint` requires the `cuda` extra installed (`uv sync --extra dev --extra cuda`) — pyright resolves imports against the active venv and some packages ship cuda-only. Line length is 120, not 88; config lives in `pyproject.toml`.

## Submitting Changes

1. Fork the repository and create a branch from `main`
2. Make your changes
3. Ensure `make lint` and `make test` pass
4. Open a pull request against `main` with a clear description of what changed and why

## Getting Help

For usage questions, join the [Discord](https://discord.gg/apbtTdk2Am) — `#help` is faster than an issue for "how do I configure X".

## Reporting Issues

Use [GitHub Issues](https://github.com/modelship-ai/modelship/issues) for confirmed bugs and feature requests. For bugs, include:

- GPU model and VRAM
- Your `models.yaml` configuration
- Docker and NVIDIA driver versions
- Relevant logs from the container

## Security

For security vulnerabilities, see [SECURITY.md](SECURITY.md) for responsible disclosure instructions.
