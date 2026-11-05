"""espeak-ng phonemization in a restartable helper process.

The native phonemizer segfaults on some short inputs, so it runs in a spawned
subprocess: a crash raises EspeakError for the one request and the helper is
restarted, leaving the replica alive. One lock serializes the pipe."""

import multiprocessing as mp
import threading
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess


class EspeakError(RuntimeError):
    pass


def _worker(conn) -> None:
    import piper_phonemize

    while True:
        try:
            text, voice = conn.recv()
        except EOFError:
            return
        conn.send(piper_phonemize.phonemize_espeak(text, voice))


class Espeak:
    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._proc: BaseProcess
        self._parent: Connection
        self._start()

    def _start(self) -> None:
        self._parent, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=_worker, args=(child,), daemon=True)
        self._proc.start()
        child.close()

    def phonemize(self, text: str, voice: str) -> list[list[str]]:
        """A list of sentences, each a list of phoneme-symbol strings."""
        with self._lock:
            try:
                self._parent.send((text, voice))
                return self._parent.recv()
            except (EOFError, BrokenPipeError) as err:
                self._proc.join()
                code = self._proc.exitcode
                self._start()
                raise EspeakError(f"espeak phonemizer crashed (exit {code})") from err

    def close(self) -> None:
        with self._lock:
            self._proc.terminate()
            self._proc.join()
            self._parent.close()
