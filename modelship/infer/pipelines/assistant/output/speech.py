"""Speaking an answer while the LLM writes it: `Chunker` cuts the text into pieces as it arrives, and `Speech` has the
pipeline's tts model speak them one at a time, in order, joining the pieces that wait when the model is free.
`Spoken` records what each call said, which `conversation.item.truncate` cuts the transcript by."""

import asyncio
import contextlib
import math
import re
from collections.abc import Awaitable, Callable

from modelship.infer.pipelines.assistant.output.chunker import MAX_CHARS, Chunker, separator, speakable
from modelship.logging import get_logger
from modelship.openai.protocol.realtime import RealtimeError

logger = get_logger("realtime")

_WORD = re.compile(r"\S+")


class Spoken:
    """What a spoken answer has said: for each tts call, where its text ends in the transcript and how much of its
    audio went out, in ms."""

    def __init__(self) -> None:
        self._calls: list[list[float]] = []
        # the audio after this was truncated away
        self._limit_ms = math.inf

    @property
    def ms(self) -> float:
        """The audio sent, as long as the client was told."""
        return min(sum(ms for _, ms in self._calls), self._limit_ms)

    def said(self, end: int) -> None:
        """A tts call for the transcript up to *end* begins."""
        self._calls.append([end, 0.0])

    def sent(self, ms: float) -> None:
        """*ms* more of the last call's audio went out."""
        self._calls[-1][1] += ms

    def truncate(self, audio_end_ms: float, transcript: str) -> str:
        """*transcript* as far as the first *audio_end_ms* of the audio says it: each call heard in full, then the words
        of the call it cuts in the share of its audio heard, rounded down."""
        self._limit_ms = min(self._limit_ms, audio_end_ms)
        start, start_ms = 0, 0.0
        for end, ms in self._calls:
            # a call that said nothing counts as heard only when audio after it was
            heard = start_ms + ms <= audio_end_ms if ms else start_ms < audio_end_ms
            if heard:
                start, start_ms = int(end), start_ms + ms
                continue
            if not ms:
                break
            words = list(_WORD.finditer(transcript, start, int(end)))
            kept = int(len(words) * (audio_end_ms - start_ms) / ms)
            return transcript[: words[kept - 1].end()] if kept else transcript[:start]
        return transcript[:start]


# speaks a piece of text: the WAV, or why it couldn't
Speak = Callable[[str], Awaitable[bytes | RealtimeError]]
# turns the tts model's WAV into the client's format
Encode = Callable[[bytes], bytes]
# sends a piece of the audio in the client's format; whether it went out
SendAudio = Callable[[bytes], Awaitable[bool]]


class Speech:
    """One spoken answer: *speak* says each piece of the text written to it, and the audio goes out through *send* in
    pieces of *step* bytes, *bytes_per_second* of the encoded audio a second."""

    def __init__(self, speak: Speak, encode: Encode, send: SendAudio, step: int, bytes_per_second: int):
        self._speak = speak
        self._encode = encode
        self._send = send
        self._step = step
        self._bytes_per_second = bytes_per_second
        self._chunker = Chunker()
        # the text written so far, and where the pieces queued so far end in it
        self._text = ""
        self._queued_to = 0
        # each piece to speak, with where it ends in the text
        self._pieces: list[tuple[str, int]] = []
        self._written = False
        self._wake = asyncio.Event()
        self.spoken = Spoken()
        # why the answer stopped being spoken, or None
        self.failure: RealtimeError | None = None
        self._task = asyncio.ensure_future(self._run())

    def write(self, text: str) -> None:
        self._text += text
        self._queue(self._chunker.push(text))

    async def finish(self) -> RealtimeError | None:
        """Speaks what is left; why it couldn't, or None."""
        self._queue(self._chunker.flush())
        self._written = True
        self._wake.set()
        await self._task
        return self.failure

    async def stop(self) -> None:
        """Stops speaking, now."""
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    def _queue(self, pieces: list[str]) -> None:
        for piece in pieces:
            # a piece is the text it was cut from, stripped
            self._queued_to = self._text.index(piece, self._queued_to) + len(piece)
            self._pieces.append((piece, self._queued_to))
        if pieces:
            self._wake.set()

    def _take(self, first: bool) -> tuple[str, int]:
        text, end = self._pieces.pop(0)
        if not first:
            while self._pieces:
                joined = text + separator(text) + self._pieces[0][0]
                if len(joined) > MAX_CHARS:
                    break
                text, end = joined, self._pieces.pop(0)[1]
        return text, end

    async def _run(self) -> None:
        first = True
        try:
            while True:
                while not self._pieces:
                    if self._written:
                        return
                    self._wake.clear()
                    await self._wake.wait()
                piece, end = self._take(first)
                text = speakable(piece)
                first = False
                self.spoken.said(end)
                if text and not await self._say(text):
                    return
        except Exception:
            logger.exception("Speaking an answer failed")
            self.failure = RealtimeError(
                message="Internal error while speaking the answer.", type="server_error", code="server_error"
            )

    async def _say(self, text: str) -> bool:
        spoken = await self._speak(text)
        if isinstance(spoken, RealtimeError):
            self.failure = spoken
            return False
        audio = self._encode(spoken)
        for start in range(0, len(audio), self._step):
            piece = audio[start : start + self._step]
            if await self._send(piece):
                self.spoken.sent(len(piece) * 1000 / self._bytes_per_second)
        return True
