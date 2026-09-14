"""Single source of truth for the cache root directories. Ray/torch-free."""

from __future__ import annotations

import os

_CONTAINER_CACHE_DIR = "/.cache"


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
