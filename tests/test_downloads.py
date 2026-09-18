"""locked_download against a fake lease actor: when it takes the lease, and what
it does while holding it."""

import asyncio
import inspect
import threading
import time
from unittest.mock import MagicMock

import pytest
from ray.exceptions import ActorDiedError, ActorUnavailableError

from modelship.infer import downloads
from modelship.infer.sources import ArchiveSource, HfSource, LocalSource, ModelDownloadError

_SOURCE = HfSource("org/repo", "a" * 40, "model.gguf", None, None, None)


def _raise(error: Exception):
    raise error


class _Method:
    def __init__(self, fn):
        self.fn = fn
        self.calls: list[tuple] = []

    def remote(self, *args):
        self.calls.append(args)

        async def call():
            result = self.fn(*args)
            return await result if inspect.isawaitable(result) else result

        return call()


class _FakeLeases:
    def __init__(self, blockers=(), renew=lambda *_: True):
        pending = list(blockers)
        self.acquire = _Method(lambda *_: pending.pop(0) if pending else None)
        self.renew = _Method(renew)
        self.release = _Method(lambda *_: None)


@pytest.fixture
def env(monkeypatch):
    """Patches every collaborator; `env.events` records the order they ran in."""
    env = MagicMock()
    env.events = []
    env.leases = _FakeLeases()
    env.exits = []

    def remove_leftovers_on(node_id, source):
        env.events.append(("cleanup", node_id, source))

        async def done():
            return 0

        return done()

    def download(source):
        env.events.append(("download", source))
        return "/cache/model.gguf"

    context = MagicMock()
    context.get_node_id.return_value = "node-1"
    monkeypatch.setattr(downloads, "get_or_create_leases", lambda: env.leases)
    monkeypatch.setattr(downloads, "remove_leftovers_on", remove_leftovers_on)
    monkeypatch.setattr(downloads, "download_model_source", download)
    monkeypatch.setattr(downloads, "is_cached", lambda source: False)
    monkeypatch.setattr(downloads, "shared_cache_id", lambda: "cid")
    monkeypatch.setattr(downloads.ray, "get_runtime_context", lambda: context)
    monkeypatch.setattr(downloads.os, "_exit", env.exits.append)
    monkeypatch.setattr(downloads, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(downloads, "RENEW_SECONDS", 0.01)
    monkeypatch.setattr(downloads, "_RPC_TIMEOUT_SECONDS", 0.05)
    return env


@pytest.mark.asyncio
class TestSkipsTheLease:
    async def test_cached_source(self, env, monkeypatch):
        monkeypatch.setattr(downloads, "is_cached", lambda source: True)
        assert await downloads.locked_download(_SOURCE, "m") == "/cache/model.gguf"
        assert env.leases.acquire.calls == []
        assert env.events == [("download", _SOURCE)]

    async def test_local_source(self, env, tmp_path, monkeypatch):
        monkeypatch.setattr(downloads, "download_model_source", lambda source: source.path)
        assert await downloads.locked_download(LocalSource(str(tmp_path)), "m") == str(tmp_path)
        assert env.leases.acquire.calls == []


@pytest.mark.asyncio
class TestHoldsTheLease:
    async def test_cleans_then_downloads_then_releases(self, env):
        assert await downloads.locked_download(_SOURCE, "m") == "/cache/model.gguf"
        [(key, source, holder, node_id)] = env.leases.acquire.calls
        assert (key, source, node_id) == ("cid/hf/org/repo", _SOURCE, "node-1")
        assert holder.startswith("m@")
        assert env.events == [("cleanup", "node-1", _SOURCE), ("download", _SOURCE)]
        assert env.leases.release.calls == [(key, holder)]

    async def test_archive_key_is_its_dest(self, env):
        source = ArchiveSource("http://x/a.tar.bz2", "0" * 64, "sherpa_onnx/a-00000000", (), None)
        await downloads.locked_download(source, "m")
        assert env.leases.acquire.calls[0][0] == "cid/archive/sherpa_onnx/a-00000000"

    async def test_waits_until_granted_logging_each_new_reason(self, env, caplog):
        env.leases = _FakeLeases(blockers=["lease service starting", "held by x", "held by x"])
        with caplog.at_level("INFO"):
            await downloads.locked_download(_SOURCE, "m")
        assert len(env.leases.acquire.calls) == 4
        assert caplog.text.count("waiting for download lease cid/hf/org/repo (lease service starting)") == 1
        assert caplog.text.count("waiting for download lease cid/hf/org/repo (held by x)") == 1

    async def test_releases_when_the_download_fails(self, env, monkeypatch):
        monkeypatch.setattr(downloads, "download_model_source", MagicMock(side_effect=OSError("network blip")))
        with pytest.raises(OSError, match="network blip"):
            await downloads.locked_download(_SOURCE, "m")
        assert len(env.leases.release.calls) == 1

    async def test_a_hung_release_does_not_hold_up_the_path(self, env):
        env.leases.release = _Method(lambda *_: asyncio.sleep(10))
        assert await asyncio.wait_for(downloads.locked_download(_SOURCE, "m"), 1) == "/cache/model.gguf"

    async def test_a_cancel_keeps_the_lease_until_the_download_ends(self, env, monkeypatch):
        started, finish = threading.Event(), threading.Event()

        def download(source):
            started.set()
            finish.wait(5)
            return "/cache/model.gguf"

        monkeypatch.setattr(downloads, "download_model_source", download)
        caller = asyncio.create_task(downloads.locked_download(_SOURCE, "m"))
        await asyncio.to_thread(started.wait, 5)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        await asyncio.sleep(0.05)
        assert env.leases.release.calls == []
        assert env.leases.renew.calls

        finish.set()
        [held] = downloads._held
        assert await held == "/cache/model.gguf"
        assert len(env.leases.release.calls) == 1


@pytest.mark.asyncio
class TestLosingTheLease:
    def _slow_download(self, monkeypatch):
        def download(source):
            time.sleep(0.2)
            return "/cache/model.gguf"

        monkeypatch.setattr(downloads, "download_model_source", download)

    async def test_refused_renewal_exits(self, env, monkeypatch):
        env.leases = _FakeLeases(renew=lambda *_: False)
        self._slow_download(monkeypatch)
        await downloads.locked_download(_SOURCE, "m")
        assert env.exits and env.exits[0] == 1

    async def test_renewal_timeout_exits(self, env, monkeypatch):
        env.leases = _FakeLeases(renew=lambda *_: asyncio.sleep(1))
        self._slow_download(monkeypatch)
        await downloads.locked_download(_SOURCE, "m")
        assert env.exits and env.exits[0] == 1

    async def test_renewals_keep_the_lease(self, env, monkeypatch):
        self._slow_download(monkeypatch)
        await downloads.locked_download(_SOURCE, "m")
        assert len(env.leases.renew.calls) > 1
        assert env.exits == []


@pytest.mark.asyncio
class TestUnreachableService:
    async def test_is_a_retryable_error_after_three_lookups(self, env, monkeypatch):
        lookups = []

        def lookup():
            lookups.append(1)
            leases = _FakeLeases()
            leases.acquire = _Method(lambda *_: _raise(ActorDiedError()))
            return leases

        monkeypatch.setattr(downloads, "get_or_create_leases", lookup)
        with pytest.raises(ModelDownloadError, match="download lease service unreachable"):
            await downloads.locked_download(_SOURCE, "m")
        assert len(lookups) == 3
        assert env.events == []

    async def test_recovers_on_a_fresh_lookup(self, env, monkeypatch):
        exiting = _FakeLeases()
        exiting.acquire = _Method(lambda *_: _raise(ActorUnavailableError("exiting", None)))
        handles = [exiting, env.leases]
        monkeypatch.setattr(downloads, "get_or_create_leases", lambda: handles.pop(0))
        assert await downloads.locked_download(_SOURCE, "m") == "/cache/model.gguf"
        assert len(env.leases.acquire.calls) == 1
