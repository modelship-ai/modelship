import asyncio
import gc
import threading
import time
from unittest.mock import MagicMock

import pytest

from modelship.infer.base_infer import BaseInfer


class ExclusiveInfer(BaseInfer):
    def __init__(self):
        super().__init__(MagicMock())

    def shutdown(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def warmup(self) -> None:
        pass


async def wait_for_turn(infer: BaseInfer) -> None:
    await infer.run_blocking(lambda _stop: None)


@pytest.mark.asyncio
async def test_a_run_holds_the_turn_until_its_thread_exits_after_its_caller_left():
    infer = ExclusiveInfer()
    started = threading.Event()
    seen = []

    def slow(stop: threading.Event) -> None:
        started.set()
        seen.append(f"stopped={stop.wait(5)}")
        time.sleep(0.1)
        seen.append("slow ended")

    first = asyncio.ensure_future(infer.run_blocking(slow))
    await asyncio.to_thread(started.wait, 5)
    first.cancel()
    await infer.run_blocking(lambda _stop: seen.append("next"))

    assert seen == ["stopped=True", "slow ended", "next"]
    assert first.cancelled()


@pytest.mark.asyncio
async def test_a_queued_run_whose_caller_left_never_runs():
    infer = ExclusiveInfer()
    gate = threading.Event()
    seen = []

    first = asyncio.ensure_future(infer.run_blocking(lambda _stop: gate.wait(5)))
    await asyncio.sleep(0)
    queued = asyncio.ensure_future(infer.run_blocking(lambda _stop: seen.append("queued")))
    await asyncio.sleep(0)
    queued.cancel()
    gate.set()
    await first
    await wait_for_turn(infer)

    assert seen == []


@pytest.mark.asyncio
async def test_runs_that_are_not_exclusive_overlap():
    infer = ExclusiveInfer()
    both = threading.Barrier(2, timeout=5)

    await asyncio.gather(
        infer.run_blocking(lambda _stop: both.wait(), exclusive=False),
        infer.run_blocking(lambda _stop: both.wait(), exclusive=False),
    )


@pytest.mark.asyncio
async def test_a_result_and_an_error_reach_the_caller():
    infer = ExclusiveInfer()

    def fails(_stop: threading.Event) -> None:
        raise RuntimeError("boom")

    assert await infer.run_blocking(lambda _stop: 42) == 42
    with pytest.raises(RuntimeError, match="boom"):
        await infer.run_blocking(fails)


@pytest.mark.asyncio
async def test_an_error_after_the_caller_left_is_not_reported_as_unretrieved():
    infer = ExclusiveInfer()
    loop = asyncio.get_running_loop()
    reported = []
    loop.set_exception_handler(lambda _loop, context: reported.append(context))
    started = threading.Event()

    def fails_when_stopped(stop: threading.Event) -> None:
        started.set()
        stop.wait(5)
        raise RuntimeError("stopped")

    try:
        task = asyncio.ensure_future(infer.run_blocking(fails_when_stopped))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await wait_for_turn(infer)
        del task
        gc.collect()
    finally:
        loop.set_exception_handler(None)

    assert reported == []
