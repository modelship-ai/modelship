"""How a session calls its pipeline's models. The pipeline replica implements it (`model_calls.py`), with the
connection's headers, identity and disconnect registry bound, so a session never touches Ray."""

from collections.abc import AsyncGenerator
from typing import Any, Protocol

from modelship.infer.pipelines.routes import ModelState
from modelship.openai.protocol import (
    ErrorResponse,
    RawSpeechResponse,
    SpeechRequest,
    SttCapabilities,
    SttStreamOptions,
    SttStreamText,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
)
from modelship.openai.protocol.responses.schemas import ResponsesRequest

# a model's streamed answer: Responses event dicts, or the ErrorResponse its loader answered with instead; closing
# it stops the model
Answer = AsyncGenerator[Any, None]


class PipelineCalls(Protocol):
    def model_state(self, model: str) -> ModelState:
        """Whether *model* can be called now."""
        ...

    def respond(self, model: str, request: ResponsesRequest, session_id: str) -> Answer | None:
        """Starts *model* answering *request* for session *session_id*; None while nothing serves *model*."""
        ...

    async def stt_capabilities(self, model: str) -> SttCapabilities | None:
        """What *model* can do as speech to text; None when it doesn't transcribe or nothing serves it."""
        ...

    async def transcribe(
        self, model: str, audio: bytes, request: TranscriptionRequest, session_id: str
    ) -> TranscriptionResponse | TranscriptionResponseVerbose | ErrorResponse | None:
        """*model*'s transcript of *audio* (a WAV), as *request* asks; None while nothing serves *model*."""
        ...

    async def transcribe_push(
        self, model: str, key: str, pcm: bytes, rate: int, options: SttStreamOptions
    ) -> SttStreamText | ErrorResponse | None:
        """Appends *pcm* (16-bit mono at *rate*) to *model*'s stream *key*, opening it with *options*; the stream's
        text so far. None while nothing serves *model*."""
        ...

    async def transcribe_finish(
        self, model: str, key: str, pcm: bytes, rate: int, options: SttStreamOptions
    ) -> SttStreamText | ErrorResponse | None:
        """Appends *pcm* to *model*'s stream *key* and ends it; its final text. None while nothing serves *model*."""
        ...

    async def transcribe_close(self, model: str, key: str) -> None:
        """Drops *model*'s stream *key*."""
        ...

    async def speak(
        self, model: str, request: SpeechRequest, session_id: str
    ) -> RawSpeechResponse | ErrorResponse | None:
        """*model* speaking *request*'s text, as a WAV; None while nothing serves *model*."""
        ...
