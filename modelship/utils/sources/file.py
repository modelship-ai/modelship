import contextlib
import glob
import os
from typing import NamedTuple

from modelship.logging import get_logger
from modelship.utils import cache_dir, download, random_uuid, verify_sha256
from modelship.utils.sources.errors import ModelSourceError
from modelship.utils.sources.http import content_length
from modelship.utils.sources.progress import DownloadProgress

logger = get_logger("startup")


class FileSource(NamedTuple):
    """One file pinned by sha256, published at `dest` under the shared cache root.
    `dest` should embed the hash, so a new file gets a new name."""

    url: str
    sha256: str
    dest: str
    total_bytes: int | None


def check_file_source(url: str, sha256: str, dest: str) -> FileSource:
    return FileSource(url, sha256, dest, content_length(url))


def _dest(source: FileSource) -> str:
    return os.path.join(cache_dir(), source.dest)


def is_file_cached(source: FileSource) -> bool:
    return os.path.isfile(_dest(source))


def download_file_source(source: FileSource) -> str:
    """Publication is download-to-temp + verify + `os.replace`. An existing `dest`
    is verified again, and one that fails is reported, never deleted."""
    dest = _dest(source)
    if os.path.isfile(dest):
        return _published(dest, source.sha256)

    root, name = os.path.split(dest)
    os.makedirs(root, exist_ok=True)
    # unique, so concurrent fetchers never share a download
    tmp_path = os.path.join(root, f".{name}.{random_uuid()}.download")
    try:
        progress = DownloadProgress(name, source.total_bytes)
        success = False
        try:
            download(source.url, tmp_path, on_chunk=progress.add)
            success = True
        finally:
            progress.finish(success)
        verify_sha256(tmp_path, source.sha256)
        os.replace(tmp_path, dest)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp_path)
    return dest


def _published(dest: str, sha256: str) -> str:
    try:
        verify_sha256(dest, sha256)
    except ValueError as e:
        raise ModelSourceError(f"{e}; delete it to fetch again") from None
    return dest


def remove_file_leftovers(source: FileSource) -> int:
    """Deletes unfinished downloads; a published `dest` is kept."""
    root, name = os.path.split(_dest(source))
    removed = 0
    # the download and download()'s temp beside it
    for path in glob.glob(os.path.join(glob.escape(root), f".{glob.escape(name)}.*.download*")):
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)
            removed += 1
    return removed
