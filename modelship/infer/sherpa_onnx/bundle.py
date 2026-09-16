"""Pins sherpa_onnx registry bundles as model sources."""

import os

from modelship.infer.sherpa_onnx.registry import REGISTRY, SherpaOnnxRegistryEntry, registry_name
from modelship.infer.sources import ArchiveSource, LocalSource, check_archive_source, check_required
from modelship.utils import is_pathy


def bundle_source(model: str) -> LocalSource | ArchiveSource:
    """`model` is a registry name (fetched into the shared cache) or a local
    bundle directory whose basename is one, validated here."""
    entry = REGISTRY[registry_name(model)]
    required = bundle_paths(entry)
    if is_pathy(model):
        path = os.path.abspath(os.path.expanduser(model))
        check_required(path, required)
        return LocalSource(path, required)
    dest = os.path.join("sherpa_onnx", f"{model}-{entry.sha256[:8]}")
    return check_archive_source(entry.tarball_url, entry.sha256, dest, required)


def bundle_paths(entry: SherpaOnnxRegistryEntry) -> tuple[str, ...]:
    """Every path the entry points sherpa at, relative to the bundle root."""
    return (*entry.files.values(), *entry.lexicon, *(f"{d}/" for d in entry.dirs.values()))
