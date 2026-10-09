import tomllib
from pathlib import Path


def test_the_dockerfile_copies_the_uv_version_pyproject_requires():
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    version = pyproject["tool"]["uv"]["required-version"].removeprefix("==")

    assert f"COPY --from=ghcr.io/astral-sh/uv:{version} " in (root / "Dockerfile").read_text()
