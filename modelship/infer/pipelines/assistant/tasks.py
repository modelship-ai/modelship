"""`BackgroundTasks`: the tasks a session starts beside its event loop, held until they finish."""

import asyncio
from collections.abc import Coroutine
from typing import Any


class BackgroundTasks:
    """A set of tasks, each dropped from it once it finishes, so the session can cancel or wait for the ones still
    running when it closes. asyncio keeps only a weak reference to a task, so one nothing holds can be collected
    mid-run."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Future] = set()

    def __len__(self) -> int:
        return len(self._tasks)

    def spawn(self, work: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """Runs *work* as a task held here."""
        task = asyncio.ensure_future(work)
        self.add(task)
        return task

    def add(self, task: asyncio.Future) -> None:
        """Holds *task*, started elsewhere, until it finishes."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def cancel(self) -> None:
        """Cancels every task still running."""
        for task in self._tasks:
            task.cancel()

    async def wait(self, timeout: float) -> None:
        """Waits for the tasks still running, at most *timeout* seconds."""
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=timeout)
