"""Model sources: pinned on the driver without fetching bytes, downloaded per
replica by `download_model_source`, which returns the path a loader reads."""

from pathlib import Path

from modelship.infer.sources import archive, hf, local
from modelship.infer.sources.archive import ArchiveSource, check_archive_source
from modelship.infer.sources.hf import HfSource
from modelship.infer.sources.local import LocalSource, check_required
from modelship.utils.model_ref import parse_model_ref

__all__ = [
    "ArchiveSource",
    "HfSource",
    "LocalSource",
    "ModelDownloadError",
    "PinnedSource",
    "check_archive_source",
    "check_model_source",
    "check_required",
    "download_model_source",
    "resolve_model_source",
]

PinnedSource = LocalSource | HfSource | ArchiveSource


class ModelDownloadError(Exception):
    """A validated source failed to download; `ModelDeployment` treats it as
    transient, unlike the check errors."""


def check_model_source(model_ref: str, trust_remote_code: bool = False) -> LocalSource | HfSource:
    """Driver-side: validate a local path or HF repo ref without fetching any weight bytes."""
    source, selector, is_local = parse_model_ref(model_ref)
    # A relative path that exists isn't pathy, but is still local.
    if is_local or Path(source).exists():
        return local.check_local_source(source, selector)
    return hf.check_hf_source(source, selector, trust_remote_code=trust_remote_code)


def download_model_source(pinned: PinnedSource) -> str:
    """Download (or confirm already-cached) *pinned* and return its final absolute local path."""
    match pinned:
        case LocalSource():
            return local.download_local_source(pinned)
        case HfSource():
            return hf.download_hf_source(pinned)
        case ArchiveSource():
            return archive.download_archive_source(pinned)


def resolve_model_source(model_ref: str, trust_remote_code: bool = False) -> str:
    """One-shot check + download, for callers without the driver/replica split."""
    return download_model_source(check_model_source(model_ref, trust_remote_code=trust_remote_code))
