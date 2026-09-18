"""Replica-side download under the cluster-wide lease (see `download_leases`)."""

import asyncio
import contextlib
import os
import socket
import time

import ray
from ray.exceptions import ActorDiedError, ActorUnavailableError

from modelship.infer.download_leases import RENEW_SECONDS, get_or_create_leases, remove_leftovers_on
from modelship.infer.sources import (
    ArchiveSource,
    HfSource,
    LocalSource,
    ModelDownloadError,
    PinnedSource,
    RemoteSource,
    download_model_source,
    is_cached,
)
from modelship.logging import get_logger
from modelship.utils import random_uuid
from modelship.utils.cache import shared_cache_id

logger = get_logger("startup")

POLL_SECONDS = 2.0
_RPC_TIMEOUT_SECONDS = 5.0
_WAIT_LOG_SECONDS = 60.0
# an idle-exiting actor's name resolves for ~2s after it stops answering
_MAX_LOOKUPS = 3


async def locked_download(source: PinnedSource, model_name: str) -> str:
    """`download_model_source` holding `source`'s lease, after removing its leftovers.
    Local and already-cached sources transfer nothing, so they skip the lease."""
    loop = asyncio.get_running_loop()
    if isinstance(source, LocalSource) or await loop.run_in_executor(None, is_cached, source):
        return await loop.run_in_executor(None, download_model_source, source)

    key = await loop.run_in_executor(None, lease_key, source)
    holder = f"{model_name}@{socket.gethostname()}/{os.getpid()}/{random_uuid()[:8]}"
    node_id = ray.get_runtime_context().get_node_id()
    leases = await _acquire(key, source, holder, node_id, model_name)
    if leases is None:
        return await loop.run_in_executor(None, download_model_source, source)
    # a cancel stops neither the cleanup task nor the download thread
    held = asyncio.create_task(_download_held(leases, key, holder, node_id, source, model_name))
    _held.add(held)
    held.add_done_callback(_held.discard)
    return await asyncio.shield(held)


# asyncio references tasks only weakly
_held: set[asyncio.Task] = set()


async def _download_held(leases, key: str, holder: str, node_id: str, source: RemoteSource, model_name: str) -> str:
    renewer = asyncio.create_task(_renew_forever(leases, key, holder, model_name))
    try:
        removed = await remove_leftovers_on(node_id, source)
        if removed:
            logger.info("%s: removed %d leftover download file(s)", model_name, removed)
        return await asyncio.get_running_loop().run_in_executor(None, download_model_source, source)
    finally:
        renewer.cancel()
        # an unreleased lease expires on its own
        with contextlib.suppress(Exception):
            await asyncio.wait_for(leases.release.remote(key, holder), _RPC_TIMEOUT_SECONDS)


def lease_key(source: RemoteSource) -> str:
    """Scoped to the cache mount, so nodes with separate caches download in parallel."""
    cache_id = shared_cache_id()
    match source:
        case HfSource():
            # per repo, not revision: revisions share the repo's blobs
            return f"{cache_id}/hf/{source.repo}"
        case ArchiveSource():
            return f"{cache_id}/archive/{source.dest}"


async def _acquire(key: str, source: RemoteSource, holder: str, node_id: str, model_name: str):
    """Polls until granted, without limit, or None once `source` is cached meanwhile;
    the lease actor being unreachable is a retryable failure."""
    loop = asyncio.get_running_loop()
    leases = get_or_create_leases()
    lookups = 1
    logged: tuple[str, float] | None = None
    while True:
        try:
            blocker = await leases.acquire.remote(key, source, holder, node_id)
        except (ActorDiedError, ActorUnavailableError) as e:
            if lookups >= _MAX_LOOKUPS:
                raise ModelDownloadError(f"download lease service unreachable: {e}") from e
            await asyncio.sleep(POLL_SECONDS)
            leases = get_or_create_leases()
            lookups += 1
            continue
        if blocker is None:
            return leases
        now = time.monotonic()
        if logged is None or blocker != logged[0] or now - logged[1] >= _WAIT_LOG_SECONDS:
            logger.info("%s: waiting for download lease %s (%s)", model_name, key, blocker)
            logged = (blocker, now)
        await asyncio.sleep(POLL_SECONDS)
        if await loop.run_in_executor(None, is_cached, source):
            return None


async def _renew_forever(leases, key: str, holder: str, model_name: str) -> None:
    """Exits the process once the lease is lost: HF's download threads can't be cancelled."""
    while True:
        await asyncio.sleep(RENEW_SECONDS)
        try:
            renewed = await asyncio.wait_for(leases.renew.remote(key, holder), _RPC_TIMEOUT_SECONDS)
            reason = "renewal refused"
        except Exception as e:
            renewed, reason = False, repr(e)
        if not renewed:
            logger.error("%s: lost download lease %s (%s); exiting", model_name, key, reason)
            os._exit(1)
