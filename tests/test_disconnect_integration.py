"""Integration test for client-disconnect propagation against a real Ray
cluster: RequestWatcher -> DisconnectRegistry (real detached actor) -> the
BaseInfer per-replica batched pump (real `is_set_many` RPC) ->
run_cancellable/run_cancellable_stream.

Other tests stub `get_disconnect_registry()` out (see conftest.py); this one
exercises the actual cross-process wiring against a small local Ray instance
rather than a full `mship_deploy` cluster.
"""

import asyncio
import contextlib
from unittest.mock import MagicMock

import pytest
import ray

from modelship.infer.base_infer import BaseInfer, ClientDisconnectedError
from modelship.infer.infer_config import (
    RawRequestProxy,
    RequestWatcher,
    get_disconnect_registry,
    reset_disconnect_registry,
)


class _FakeClientRequest:
    """Stands in for the real FastAPI Request RequestWatcher watches;
    `disconnect()` flips `is_disconnected()` to True, as a dropped client
    socket would."""

    def __init__(self) -> None:
        self._disconnected = False

    def disconnect(self) -> None:
        self._disconnected = True

    async def is_disconnected(self) -> bool:
        return self._disconnected


class _Infer(BaseInfer):
    """Minimal concrete BaseInfer with the real (unmocked) `_poll_disconnected_ids`
    — every request routed through it hits the real DisconnectRegistry actor."""

    def __init__(self) -> None:
        super().__init__(MagicMock())
        self.aborted = 0

    def shutdown(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def warmup(self) -> None:
        pass

    async def on_generation_aborted(self) -> None:
        self.aborted += 1


@pytest.fixture(autouse=True)
def neutralize_request_watcher():
    """Override the conftest stub (see conftest.py): this module's whole point is
    to exercise the real RequestWatcher/DisconnectRegistry wiring, not mocks."""
    yield


@pytest.fixture(scope="module", autouse=True)
def real_ray_cluster():
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    reset_disconnect_registry()
    try:
        yield
    finally:
        reset_disconnect_registry()
        ray.shutdown()


def _raw_request(request_id: str) -> RawRequestProxy:
    """Mirrors production: resolves its own handle to the shared named actor
    rather than reusing a Python reference from the gateway side."""
    return RawRequestProxy(get_disconnect_registry(), {}, request_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_disconnect_propagates_end_to_end_through_real_registry():
    infer = _Infer()
    request_id = "integration-req-solo"
    client_request = _FakeClientRequest()
    watcher = RequestWatcher(client_request, request_id)

    async def slow_work():
        await asyncio.sleep(30)
        return "should never get here"  # pragma: no cover

    task = asyncio.ensure_future(infer.run_cancellable(slow_work(), _raw_request(request_id)))
    await asyncio.sleep(0.05)  # let the pump register the request and take its first poll

    client_request.disconnect()  # the real signal a dropped socket produces

    with pytest.raises(ClientDisconnectedError):
        await asyncio.wait_for(task, timeout=3.0)

    assert infer.aborted == 1
    watcher.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_disconnect_propagates_end_to_end_through_real_registry():
    infer = _Infer()
    request_id = "integration-req-stream"
    client_request = _FakeClientRequest()
    watcher = RequestWatcher(client_request, request_id)

    async def slow_stream():
        yield "first"
        await asyncio.sleep(30)
        yield "unreachable"  # pragma: no cover

    received = []
    with pytest.raises(ClientDisconnectedError):
        async with asyncio.timeout(3.0):
            async for item in infer.run_cancellable_stream(slow_stream(), _raw_request(request_id)):
                received.append(item)
                client_request.disconnect()

    assert received == ["first"]
    assert infer.aborted == 1
    watcher.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_requests_share_one_pump_and_stay_isolated():
    """N concurrent requests on one replica share a single background poller
    (one RPC per interval); a disconnect on one must not affect the others."""
    infer = _Infer()
    request_ids = [f"integration-req-concurrent-{i}" for i in range(3)]

    async def slow_work(tag: int):
        await asyncio.sleep(30)
        return tag  # pragma: no cover

    tasks = [
        asyncio.ensure_future(infer.run_cancellable(slow_work(i), _raw_request(rid)))
        for i, rid in enumerate(request_ids)
    ]
    await asyncio.sleep(0.05)

    assert len(infer._watched) == 3
    pump_task = infer._pump_task
    assert pump_task is not None and not pump_task.done()

    registry = get_disconnect_registry()
    await registry.set.remote(request_ids[1])  # disconnect only the middle request

    with pytest.raises(ClientDisconnectedError):
        await asyncio.wait_for(tasks[1], timeout=3.0)

    assert infer.aborted == 1
    assert infer._pump_task is pump_task  # same shared pump, unaffected by the other's abort
    for i in (0, 2):
        assert not tasks[i].done()

    for i in (0, 2):
        tasks[i].cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tasks[i]
