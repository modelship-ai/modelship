import contextlib
import fcntl
import os
from typing import NamedTuple

import requests

from modelship.infer.sources.errors import ModelSourceError
from modelship.infer.sources.local import check_required
from modelship.infer.sources.progress import DownloadProgress
from modelship.logging import get_logger
from modelship.utils import cache_dir, download, fetch_and_extract_archive, random_uuid

logger = get_logger("startup")

_HEAD_TIMEOUT_SECONDS = 30


class ArchiveSource(NamedTuple):
    """A tarball pinned by sha256, extracted to `dest` under the shared cache
    root. `dest` should embed the hash, so a new tarball gets a new directory."""

    url: str
    sha256: str
    dest: str
    required: tuple[str, ...]
    total_bytes: int | None


def check_archive_source(url: str, sha256: str, dest: str, required: tuple[str, ...]) -> ArchiveSource:
    try:
        response = requests.head(url, allow_redirects=True, timeout=_HEAD_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to reach {url}: {e}") from e
    length = response.headers.get("content-length", "")
    total_bytes = int(length) if length.isdigit() and int(length) > 0 else None
    return ArchiveSource(url, sha256, dest, required, total_bytes)


def download_archive_source(source: ArchiveSource) -> str:
    """Publication is extract-to-temp + `os.replace`, so an existing `dest` is
    complete; one failing `required` is reported, never deleted."""
    dest = os.path.join(cache_dir(), source.dest)
    if os.path.isdir(dest):
        return _published(dest, source.required)

    root, name = os.path.split(dest)
    os.makedirs(root, exist_ok=True)
    # flock dedups fetches only as far as the mount shares locks; unique archive paths keep the rest safe.
    with open(os.path.join(root, f".{name}.lock"), "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if os.path.isdir(dest):
                return _published(dest, source.required)
            archive_path = os.path.join(root, f".{name}.{random_uuid()}.archive")
            try:
                progress = DownloadProgress(name, source.total_bytes)
                success = False
                try:
                    download(source.url, archive_path, on_chunk=progress.add)
                    success = True
                finally:
                    progress.finish(success)
                logger.info("%s: verifying and extracting", name)
                # the archive is already in place, so this skips its own download
                fetch_and_extract_archive(source.url, source.sha256, archive_path, dest)
            finally:
                # the helper only removes it on success or a sha256 mismatch
                with contextlib.suppress(FileNotFoundError):
                    os.remove(archive_path)
            # the helper suppresses os.replace errors, assuming a concurrent extractor won
            if not os.path.isdir(dest):
                raise OSError(f"extraction did not publish {dest!r}")
            return _published(dest, source.required)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _published(dest: str, required: tuple[str, ...]) -> str:
    try:
        check_required(dest, required)
    except ValueError as e:
        raise ModelSourceError(f"{e}; delete it to fetch again") from None
    return dest
