"""Downloads the pinned llama.cpp binary and checks its own --help output for every
flag the llama_server loader and its preflight depend on — an upstream rename or
removal shows up here before it breaks a real deploy. Skips (not fails) when the
release can't be reached, since this needs real network access."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP_DIR = _REPO_ROOT / "bootstrap"
if str(_BOOTSTRAP_DIR) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_DIR))

from mship_bootstrap.fetch import FetchError, fetch_and_extract_archive  # noqa: E402
from mship_bootstrap.llama_cpp import _ASSETS  # noqa: E402

# Every flag llama_server_infer.py's _launch sends to `llama serve`.
_SERVE_FLAGS = [
    "--host",
    "--port",
    "-m",
    "-c",
    "-b",
    "-ub",
    "-fa",
    "-ctk",
    "-ctv",
    "--parallel",
    "--jinja",
    "--reasoning-format",
    "--no-webui",
    "--api-key",
    "-ngl",
    "-ts",
    "--threads",
    "--chat-template",
    "--chat-template-file",
    "--mmproj",
    "--embedding",
    "--cache-reuse",
    "--context-shift",
    "--cache-ram",
]

# Every flag modelship/preflight/llama_cpp.py sends to `llama fit-params`.
_FIT_PARAMS_FLAGS = [
    "-m",
    "--parallel",
    "-b",
    "-ub",
    "-fa",
    "-ctk",
    "-ctv",
    "-fitc",
    "-fitt",
    "-fitp",
    "-dev",
    "-c",
    "-ngl",
    "-ts",
]


def _flag_present(help_text: str, flag: str) -> bool:
    """Bounded match so e.g. "-c" doesn't match inside "--cache-reuse"."""
    return re.search(r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])", help_text) is not None


@pytest.fixture(scope="module")
def llama_binary(tmp_path_factory) -> str:
    asset = _ASSETS[("Linux", "x86_64")]
    root = tmp_path_factory.mktemp("llama-cpp")
    extract_dir = root / "extracted"
    try:
        fetch_and_extract_archive(asset.url, asset.sha256, str(root / "archive.tar.gz"), str(extract_dir), flatten=True)
    except FetchError as e:
        pytest.skip(f"could not fetch pinned llama.cpp build: {e}")
    binary = extract_dir / "llama"
    binary.chmod(binary.stat().st_mode | 0o111)
    return str(binary)


def _help_text(binary: str, subcommand: str) -> str:
    result = subprocess.run([binary, subcommand, "--help"], capture_output=True, text=True, timeout=30, check=False)
    return result.stdout + result.stderr


def test_serve_flags_still_recognized(llama_binary):
    help_text = _help_text(llama_binary, "serve")
    missing = [f for f in _SERVE_FLAGS if not _flag_present(help_text, f)]
    assert not missing, f"llama serve no longer documents: {missing}"


def test_fit_params_flags_still_recognized(llama_binary):
    help_text = _help_text(llama_binary, "fit-params")
    missing = [f for f in _FIT_PARAMS_FLAGS if not _flag_present(help_text, f)]
    assert not missing, f"llama fit-params no longer documents: {missing}"
