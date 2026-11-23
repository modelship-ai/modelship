"""How a session calls its pipeline's models. The gateway implements it (`api.py`), with the connection's headers,
identity and disconnect registry bound, so a session never touches Ray."""

from collections.abc import AsyncGenerator
from typing import Any, Protocol

from modelship.openai.protocol import (
    ErrorResponse,
    SttCapabilities,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
)
from modelship.openai.protocol.responses.schemas import ResponsesRequest

# a model's streamed answer: Responses event dicts, or the ErrorResponse its loader answered with instead; closing
# it stops the model
Answer = AsyncGenerator[Any, None]


class PipelineCalls(Protocol):
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
