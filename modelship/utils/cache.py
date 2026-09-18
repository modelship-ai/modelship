"""Single source of truth for the cache root directories. Ray/torch-free."""

from __future__ import annotations

import os
import uuid

_CONTAINER_CACHE_DIR = "/.cache"
_CACHE_ID_FILE = ".mship-cache-id"


def resolve_cache_root() -> str:
    """MSHIP_CACHE_DIR if set -> writable `/.cache` -> else `~/.modelship/cache`."""
    if env_dir := os.environ.get("MSHIP_CACHE_DIR"):
        return env_dir
    if os.path.isdir(_CONTAINER_CACHE_DIR) and os.access(_CONTAINER_CACHE_DIR, os.W_OK):
        return _CONTAINER_CACHE_DIR
    home_cache = os.path.expanduser("~/.modelship/cache")
    os.makedirs(home_cache, exist_ok=True)
    return home_cache


def resolve_node_cache_root() -> str:
    """MSHIP_NODE_CACHE_DIR if set -> else `${MSHIP_HOME:-~/.modelship}/node-cache`. Must stay node-local."""
    if env_dir := os.environ.get("MSHIP_NODE_CACHE_DIR"):
        return env_dir
    home = os.environ.get("MSHIP_HOME") or "~/.modelship"
    return os.path.join(os.path.abspath(os.path.expanduser(home)), "node-cache")


def reject_unset_cache_roots() -> None:
    """runtime_env cache paths expand from these; Ray turns an unset one into "", putting them under /."""
    missing = [var for var in ("MSHIP_CACHE_DIR", "MSHIP_NODE_CACHE_DIR") if not os.environ.get(var)]
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not set on this node; export it in the environment that starts the node's Ray."
        )


def shared_cache_id() -> str:
    """Names the MSHIP_CACHE_DIR mount: every node sharing it reads the same id."""
    root = resolve_cache_root()
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, _CACHE_ID_FILE)
    if not os.path.exists(path):
        tmp = f"{path}.{uuid.uuid4().hex}.tmp"
        with open(tmp, "w") as f:
            f.write(uuid.uuid4().hex)
        try:
            # unlike a rename, a link never replaces: of racing creators, the first wins
            os.link(tmp, path)
        except FileExistsError:
            pass
        finally:
            os.remove(tmp)
    with open(path) as f:
        return f.read()
