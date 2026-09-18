import contextlib
import hashlib
import logging
import os
import random
import re
import shutil
import string
import tarfile
from collections.abc import Callable, Iterable
from typing import Any

import requests

from modelship.utils.cache import resolve_cache_root
from modelship.utils.request_id import base_request_id as base_request_id
from modelship.utils.request_id import random_uuid

_RAND_CHARS = string.ascii_lowercase + string.digits

# per connect and per read, not for the whole transfer
_DOWNLOAD_TIMEOUT_SECONDS = 30

_MEMORY_SIZE_RE = re.compile(r"(\d+)\s*(ki|mi|gi|ti)?", re.IGNORECASE)
_MEMORY_UNIT_MULTIPLIERS = {"": 1, "ki": 1024, "mi": 1024**2, "gi": 1024**3, "ti": 1024**4}


def parse_memory_bytes(value: str) -> int:
    """Parse a memory size into bytes: a bare integer (bytes) or a binary-unit
    suffix (Ki/Mi/Gi/Ti, case-insensitive) — matching the Helm chart's own
    `memory: 8Gi` convention (helm/modelship/values.yaml) rather than introducing
    a second SI/decimal convention alongside it."""
    match = _MEMORY_SIZE_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Invalid memory size {value!r}; expected e.g. '8Gi', '512Mi', or a plain byte count")
    num, unit = match.groups()
    return int(num) * _MEMORY_UNIT_MULTIPLIERS[(unit or "").lower()]


def drop_reserved_kwargs(
    kwargs: dict[str, Any], reserved: Iterable[str], *, logger: logging.Logger, context: str
) -> dict[str, Any]:
    """Strip keys the caller passes to ``apply_chat_template`` itself.

    User-supplied ``chat_template_kwargs`` are splatted alongside explicit
    arguments (``tokenize``, ``tools``, ``add_generation_prompt``, …); a collision
    is a duplicate-keyword ``TypeError`` (or silently flips an explicit value).
    Drop the offenders with a warning so misconfiguration surfaces instead.
    """
    reserved = set(reserved)
    dropped = sorted(k for k in kwargs if k in reserved)
    if dropped:
        logger.warning("%s: ignoring reserved chat_template_kwargs %s", context, dropped)
    return {k: v for k, v in kwargs.items() if k not in reserved}


def rand_suffix(length: int = 5) -> str:
    return "".join(random.choices(_RAND_CHARS, k=length))


def is_pathy(s: str) -> bool:
    """A local-path-shaped string (`/...`, `./...`, `~...`), as opposed to an
    HF repo id, registry name, or other bare identifier."""
    return s.startswith(("/", "./", "~"))


def download(url: str, file_path: str, overwrite: bool = False, on_chunk: Callable[[int], None] | None = None):
    """Download ``url`` to ``file_path`` via a unique temp file + atomic rename, skipping if it
    already exists. ``on_chunk`` receives each written chunk's byte count."""
    if not overwrite and os.path.isfile(file_path):
        return

    tmp_path = f"{file_path}.{random_uuid()}.tmp"
    try:
        with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        if on_chunk is not None:
                            on_chunk(len(chunk))
        os.replace(tmp_path, file_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def verify_sha256(path: str, expected: str) -> None:
    """Raise ValueError if the file at ``path`` doesn't hash to ``expected``."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"{path} failed sha256 verification: expected {expected}, got {actual}")


def fetch_and_extract_archive(
    url: str,
    sha256: str,
    archive_path: str,
    extract_dir: str,
    *,
    flatten: bool = False,
    keep_archive: bool = False,
) -> None:
    """Download `url` to `archive_path`, verify its sha256, and extract into
    `extract_dir` via a private tmp dir + atomic `os.replace` (safe against
    concurrent extractors and partial reads). A sha256 mismatch deletes the
    archive so a retry doesn't re-verify the same bytes.

    `flatten=True` extracts every member to its basename; otherwise a single
    shared top-level directory is stripped so `extract_dir` holds its
    contents directly."""
    os.makedirs(os.path.dirname(archive_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(extract_dir) or ".", exist_ok=True)

    download(url, archive_path)
    try:
        verify_sha256(archive_path, sha256)
    except ValueError:
        os.remove(archive_path)  # don't let a retry re-verify the same corrupt bytes
        raise

    tmp_dir = f"{extract_dir}.{random_uuid()}.tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        with tarfile.open(archive_path) as tar:
            if flatten:
                for member in tar.getmembers():
                    name = os.path.basename(member.name)
                    if not name:
                        continue
                    member.name = name
                    tar.extract(member, path=tmp_dir, filter="data")
            else:
                tar.extractall(tmp_dir, filter="data")

        root = tmp_dir
        if not flatten:
            entries = os.listdir(tmp_dir)
            if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0])):
                root = os.path.join(tmp_dir, entries[0])

        with contextlib.suppress(OSError):  # a concurrent extractor already won
            os.replace(root, extract_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not keep_archive:
        os.remove(archive_path)


def cache_dir() -> str:
    path = resolve_cache_root()
    os.makedirs(path, exist_ok=True)
    return path
