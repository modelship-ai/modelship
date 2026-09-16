import threading
import time

from modelship.logging import get_logger

logger = get_logger("startup")

_MIB = 1024 * 1024
_HEARTBEAT_INTERVAL_SECONDS = 15


class DownloadProgress:
    """Logs a model's aggregate byte progress: a start line on the first byte, a timed
    heartbeat (HF's 10 MiB reads can stall tqdm for minutes), and a completion line."""

    def __init__(self, label: str, total_bytes: int | None) -> None:
        self.label = label
        self.total_bytes = total_bytes
        self.downloaded_bytes = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_t = 0.0

    def add(self, n: int) -> None:
        started_thread: threading.Thread | None = None
        with self._lock:
            self.downloaded_bytes += n
            if self._thread is None:
                self._start_t = time.monotonic()
                self._thread = threading.Thread(target=self._heartbeat, daemon=True)
                started_thread = self._thread
        if started_thread is not None:
            if self.total_bytes is not None:
                logger.info("%s: downloading (%.0f MiB total)", self.label, self.total_bytes / _MIB)
            else:
                logger.info("%s: downloading", self.label)
            started_thread.start()

    def _heartbeat(self) -> None:
        while not self._stop.wait(_HEARTBEAT_INTERVAL_SECONDS):
            logger.info("%s", self._describe())

    def _describe(self) -> str:
        mb, rate = self._stats()
        if self.total_bytes is not None and self.total_bytes > 0:
            pct = 100 * self.downloaded_bytes / self.total_bytes
            return (
                f"{self.label}: downloading {pct:3.0f}% ({mb:.0f}/{self.total_bytes / _MIB:.0f} MiB, {rate:.1f} MiB/s)"
            )
        return f"{self.label}: downloading {mb:.0f} MiB ({rate:.1f} MiB/s)"

    def _stats(self) -> tuple[float, float]:
        elapsed = max(time.monotonic() - self._start_t, 1e-9)
        mb = self.downloaded_bytes / _MIB
        return mb, mb / elapsed

    def finish(self, success: bool) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            if success:
                logger.info("%s: download complete (%.0f MiB, %.1f MiB/s)", self.label, *self._stats())
