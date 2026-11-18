"""A backend HTTP server run as a child process (llama-server, sd-server): launch, log draining, health wait,
exit watch and stop."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import httpx

from modelship.logging import TRACE, get_logger

logger = get_logger("infer.server_process")

_HEALTH_POLL_INTERVAL_S = 0.5
_RECENT_LOG_LINES = 50
_STOP_TIMEOUT_S = 10

# asyncio holds only a weak reference to a task, so a fire-and-forget close could be GC'd before it runs.
_pending_client_closes: set[asyncio.Task] = set()


class ServerProcess:
    """`label` names the server in logs and errors; `on_exit` gets a reason when the process exits outside `stop()`."""

    def __init__(self, label: str, model_name: str, on_exit: Callable[[str], Any]):
        self.label = label
        self.model_name = model_name
        self.proc: subprocess.Popen | None = None
        self._on_exit = on_exit
        self._log_lock = threading.Lock()
        self._recent_log_lines: deque[str] = deque(maxlen=_RECENT_LOG_LINES)
        self._stopping = False

    async def launch(self, args: list[str], *, cwd: str | None = None) -> None:
        logger.info("%s launch args for '%s': %s", self.label, self.model_name, args)
        proc = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: subprocess.Popen(
                args,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                errors="replace",
            ),
        )
        self.proc = proc
        assert proc.stdout is not None
        assert proc.stderr is not None
        for stream, tag in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
            threading.Thread(target=self._drain_stream, args=(stream, tag), daemon=True).start()

    def _drain_stream(self, stream: Any, tag: str) -> None:
        """Keeps the child from blocking on a full pipe; `readline()` returns '' at EOF or once `stop()` closes it."""
        try:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.rstrip("\n")
                logger.log(TRACE, "[%s:%s] %s", self.model_name, tag, line)
                with self._log_lock:
                    self._recent_log_lines.append(line)
        except (ValueError, OSError):
            pass
        finally:
            with contextlib.suppress(Exception):
                stream.close()

    def recent_logs(self) -> str:
        with self._log_lock:
            return "\n".join(self._recent_log_lines)

    async def wait_healthy(self, probe: httpx.AsyncClient, path: str, timeout_s: float) -> httpx.Response:
        """Polls `path` until it answers 200. Any exit first is a failure, exit code 0 included."""
        assert self.proc is not None
        deadline = time.monotonic() + timeout_s
        while True:
            rc = self.proc.poll()
            if rc is not None:
                raise RuntimeError(
                    f"{self.label} for '{self.model_name}' exited (rc={rc}) during startup: {self.recent_logs()}"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(f"{self.label} for '{self.model_name}' did not become healthy within {timeout_s}s")
            try:
                resp = await probe.get(path, timeout=2.0)
                if resp.status_code == 200:
                    return resp
            except httpx.HTTPError:
                pass
            await asyncio.sleep(_HEALTH_POLL_INTERVAL_S)

    def watch(self) -> None:
        assert self.proc is not None
        threading.Thread(target=self._watch, args=(self.proc,), daemon=True).start()

    def _watch(self, proc: subprocess.Popen) -> None:
        rc = proc.wait()
        if self._stopping:
            return
        logger.error("%s for '%s' exited unexpectedly (rc=%s)", self.label, self.model_name, rc)
        self._on_exit(f"{self.label} exited (rc={rc})")

    def stop(self) -> None:
        """SIGTERM, then SIGKILL after 10 s from a thread; never blocks."""
        self._stopping = True
        proc, self.proc = self.proc, None
        if proc is None:
            return
        if proc.poll() is None:
            logger.info("Shutting down %s for %s", self.label, self.model_name)
            proc.terminate()

            def _wait_and_kill():
                try:
                    proc.wait(timeout=_STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(Exception):
                        proc.kill()
                except Exception:
                    pass

            threading.Thread(target=_wait_and_kill, daemon=True).start()

        # Unblocks the drain threads.
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()


def close_client(client: httpx.AsyncClient) -> None:
    """Closes `client` on the running loop; without one (interpreter teardown), GC reclaims it."""
    try:
        task = asyncio.get_running_loop().create_task(client.aclose())
        _pending_client_closes.add(task)
        task.add_done_callback(_pending_client_closes.discard)
    except (RuntimeError, AttributeError, TypeError):
        pass
