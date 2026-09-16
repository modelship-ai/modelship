"""Fetch, verify, and extract sherpa_onnx's curated model bundles into the
shared cache dir, via modelship.utils.fetch_and_extract_archive."""

import fcntl
import os
import shutil

from modelship.infer.sherpa_onnx.registry import REGISTRY, SherpaOnnxRegistryEntry
from modelship.logging import get_logger
from modelship.utils import cache_dir, fetch_and_extract_archive, is_pathy, random_uuid

logger = get_logger("infer.sherpa_onnx.bundle")


def resolve_bundle_dir(model: str) -> tuple[str, SherpaOnnxRegistryEntry]:
    """`model` is a registry name (fetched into the shared cache) or a local
    directory path (used in place, basename must match a registry name —
    config validation already guarantees this)."""
    if is_pathy(model):
        path = os.path.expanduser(model)
        name = os.path.basename(path.rstrip("/"))
        entry = REGISTRY[name]
        validate_bundle(path, entry)
        return path, entry

    entry = REGISTRY[model]
    bundle_dir = os.path.join(cache_dir(), "sherpa_onnx", model)
    if _is_valid(bundle_dir, entry):
        return bundle_dir, entry

    # flock dedups fetches only as far as the mount shares locks; unique archive paths keep the rest safe.
    root = os.path.join(cache_dir(), "sherpa_onnx")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f".{model}.lock"), "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if _is_valid(bundle_dir, entry):  # a lock-holder ahead of us may have fixed it
                return bundle_dir, entry
            if os.path.isdir(bundle_dir):
                logger.warning("cached sherpa_onnx bundle %r failed validation, re-fetching", model)
                shutil.rmtree(bundle_dir, ignore_errors=True)

            archive_path = os.path.join(root, f".{model}-{random_uuid()}.tar.bz2")
            fetch_and_extract_archive(entry.tarball_url, entry.sha256, archive_path, bundle_dir)
            validate_bundle(bundle_dir, entry)
            return bundle_dir, entry
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _is_valid(bundle_dir: str, entry: SherpaOnnxRegistryEntry) -> bool:
    try:
        validate_bundle(bundle_dir, entry)
        return True
    except ValueError:
        return False


def validate_bundle(bundle_dir: str, entry: SherpaOnnxRegistryEntry) -> None:
    """Raises ValueError if a declared file/dir is missing. Existence only —
    the tarball's own sha256 already covers content integrity."""
    if not os.path.isdir(bundle_dir):
        raise ValueError(f"sherpa_onnx bundle directory not found: {bundle_dir!r}")

    for slot, path in entry.files.items():
        _check_exists(bundle_dir, path, os.path.isfile, f"files[{slot!r}]")
    for i, path in enumerate(entry.lexicon):
        _check_exists(bundle_dir, path, os.path.isfile, f"lexicon[{i}]")
    for slot, path in entry.dirs.items():
        _check_exists(bundle_dir, path, os.path.isdir, f"dirs[{slot!r}]")


def _check_exists(bundle_dir: str, rel_path: str, check, context: str) -> None:
    if not check(os.path.join(bundle_dir, rel_path)):
        raise ValueError(f"sherpa_onnx bundle {bundle_dir!r}: missing {context} {rel_path!r}")
