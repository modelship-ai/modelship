"""Speaking an answer while the LLM writes it: the text is cut into pieces as it arrives, and the pipeline's tts model
speaks them one at a time, in order. The first piece is short, so audio starts soon; the pieces after it are whole
sentences, and those waiting when the model is free go in one call."""

import asyncio
import contextlib
import math
import re
from collections.abc import Awaitable, Callable

from modelship.logging import get_logger
from modelship.openai.protocol.realtime import RealtimeError

logger = get_logger("realtime")

# the fewest characters a piece holds, unless it ends the answer
MIN_CHARS = 20
# the most the first piece holds, and the most any other does
FIRST_CHARS = 80
MAX_CHARS = 200

_SENTENCE_ENDS = frozenset(".!?…")
_CJK_SENTENCE_ENDS = frozenset("。！？")
_CLAUSE_MARKS = frozenset(",;:")
_EM_DASH = "—"
_DASHES = frozenset(_EM_DASH + "–")  # noqa: RUF001
_CJK_CLAUSE_MARKS = frozenset("，、；：")
# quotes, brackets and guillemets
_CLOSERS = frozenset("\"')]”’»」』")  # noqa: RUF001
_OPENERS = "\"'([“‘«「『"  # noqa: RUF001
# words a period follows without ending the sentence
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "fig", "approx", "inc", "ltd", "co"}
)

_MARKDOWN = (
    # code fences, with their language
    (re.compile(r"```[^\n]*"), " "),
    # links and images, as their text
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # rules
    (re.compile(r"^\s*([-*_]\s*){3,}$", re.MULTILINE), " "),
    # headings, quotes and bullets
    (re.compile(r"^\s{0,3}(#{1,6}|>+|[-*+])\s+", re.MULTILINE), ""),
    (re.compile(r"\*+|`+|~~"), ""),
    # underscores around a word, not inside one
    (re.compile(r"(?<!\w)_+(?=\S)|(?<=\S)_+(?!\w)"), ""),
    (re.compile(r"\|"), " "),
)


def speakable(text: str) -> str:
    """*text* without its Markdown; empty when nothing in it can be spoken."""
    for pattern, replacement in _MARKDOWN:
        text = pattern.sub(replacement, text)
    text = " ".join(text.split())
    return text if any(char.isalnum() for char in text) else ""


def _ends_sentence(text: str, period: int) -> bool:
    """Whether the period at *period* ends a sentence: not after an abbreviation, an initial or a number."""
    start = max(text.rfind(" ", 0, period), text.rfind("\n", 0, period)) + 1
    word = text[start:period].lstrip(_OPENERS).lower()
    if not word or word in _ABBREVIATIONS or word.isdigit():
        return False
    # "J." and "U.S."
    return not (len(word.replace(".", "")) <= 3 and (len(word) == 1 or "." in word))


def _separator(piece: str) -> str:
    """What goes between *piece* and the piece after it: nothing after a mark the text runs on from unspaced."""
    end = piece.rstrip("".join(_CLOSERS))[-1:]
    unspaced = end in _CJK_SENTENCE_ENDS or end in _CJK_CLAUSE_MARKS or end == _EM_DASH
    return "" if unspaced else " "


class Chunker:
    """Cuts streamed text into pieces to speak: the first short, at a sentence or clause, the rest at sentences."""

    def __init__(self) -> None:
        self._text = ""
        self._first = True

    def push(self, text: str) -> list[str]:
        """The pieces *text* completes."""
        self._text += text
        return self._drain(final=False)

    def flush(self) -> list[str]:
        """The pieces left once the text is all written."""
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[str]:
        pieces = []
        while self._text:
            cut = self._cut(final)
            if cut is None:
                break
            piece, self._text = self._text[:cut].strip(), self._text[cut:]
            if piece:
                pieces.append(piece)
                self._first = False
        return pieces

    def _boundaries(self, final: bool) -> list[tuple[int, bool]]:
        """Where the text can be cut, and whether each ends a sentence. A boundary at the end of the text counts
        only once the text is final, since more might follow it."""
        text = self._text
        found = []
        for i, char in enumerate(text):
            if char == "\n":
                found.append((i + 1, True))
                continue
            sentence = char in _SENTENCE_ENDS or char in _CJK_SENTENCE_ENDS
            clause = char in _CLAUSE_MARKS or char in _DASHES or char in _CJK_CLAUSE_MARKS
            if not sentence and not clause:
                continue
            end = i + 1
            while end < len(text) and text[end] in _CLOSERS:
                end += 1
            if end == len(text):
                if final:
                    found.append((end, sentence))
                continue
            # Latin punctuation and the en dash end at a space: "3.14", "1,000" and "10-20" don't
            unspaced = char in _CJK_SENTENCE_ENDS or char in _CJK_CLAUSE_MARKS or char == _EM_DASH
            if not unspaced and not text[end].isspace():
                continue
            if char == "." and not _ends_sentence(text, i):
                continue
            found.append((end, sentence))
        return found

    def _cut(self, final: bool) -> int | None:
        text = self._text
        limit = FIRST_CHARS if self._first else MAX_CHARS
        boundaries = [
            (end, sentence) for end, sentence in self._boundaries(final) if len(text[:end].strip()) >= MIN_CHARS
        ]
        for end, sentence in boundaries:
            if end > limit:
                break
            if sentence or self._first:
                return end
        if len(text) > limit:
            # a long run without a sentence end: its last clause, else its last word
            clauses = [end for end, _ in boundaries if end <= limit]
            if clauses:
                return clauses[-1]
            space = max(text.rfind(" ", 0, limit + 1), text.rfind("\n", 0, limit + 1))
            return space if space > 0 and len(text[:space].strip()) >= MIN_CHARS else limit
        if final:
            return len(text)
        return None


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
            if start_ms + ms <= audio_end_ms:
                start, start_ms = int(end), start_ms + ms
                continue
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
                joined = text + _separator(text) + self._pieces[0][0]
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
