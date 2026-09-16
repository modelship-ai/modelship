import os
from pathlib import Path
from typing import NamedTuple

from modelship.logging import get_logger

logger = get_logger("startup")


class LocalSource(NamedTuple):
    """A file or directory already on disk. `required` lists paths the tree
    must contain (see `check_required`), re-checked on the replica's node."""

    path: str
    required: tuple[str, ...] = ()

    @property
    def resolves_to_gguf(self) -> bool:
        return self.path.lower().endswith(".gguf")


def check_required(root: str, required: tuple[str, ...]) -> None:
    """Raises ValueError if `root` lacks a required path; a trailing `/` marks a directory."""
    if not os.path.isdir(root) and required:
        raise ValueError(f"directory not found: {root!r}")
    for rel_path in required:
        path = os.path.join(root, rel_path)
        if not (os.path.isdir(path) if rel_path.endswith("/") else os.path.isfile(path)):
            raise ValueError(f"{root!r} is missing {rel_path!r}")


def check_local_source(source: str, selector: str | None) -> LocalSource:
    path = Path(source).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Local path not found: {path}")

    if selector and path.is_dir():
        matches = sorted(path.glob(selector)) or sorted(path.rglob(selector))
        if not matches:
            raise FileNotFoundError(f"Selector {selector!r} matched no files in {path}")
        if len(matches) > 1:
            # Sharded weights (e.g. model-00001-of-00003.gguf): llama.cpp auto-loads
            # the rest given the first shard's path.
            logger.info(
                "Selector %r matched %d files in %s; returning first shard %s",
                selector,
                len(matches),
                path,
                matches[0].name,
            )
        return LocalSource(str(matches[0].absolute()))

    return LocalSource(str(path.absolute()))


def download_local_source(source: LocalSource) -> str:
    # the driver only checked its own node
    if not os.path.exists(source.path):
        raise FileNotFoundError(f"Local path not found: {source.path}")
    check_required(source.path, source.required)
    return source.path
