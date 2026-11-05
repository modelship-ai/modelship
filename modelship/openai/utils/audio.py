import base64
from collections.abc import AsyncGenerator, Callable

from fastapi.responses import Response

from modelship.openai.protocol import ErrorResponse, SpeechResponse, TranscriptionSegment, create_error_response

_TRANSCRIPT_MEDIA_TYPE = "text/plain; charset=utf-8"


async def sse_stream_speech(chunks: AsyncGenerator[tuple[bytes, int], None]) -> AsyncGenerator[str, None]:
    """Frame a stream of `(pcm16_bytes, sample_rate)` chunks as
    `speech.audio.delta`/`speech.audio.done` SSE events. Shared by every
    loader that streams TTS output."""
    async for pcm, _sample_rate in chunks:
        audio_b64 = base64.b64encode(pcm).decode("ascii")
        event = SpeechResponse(type="speech.audio.delta", audio=audio_b64)
        yield f"data: {event.model_dump_json()}\n\n"
    done = SpeechResponse(type="speech.audio.done")
    yield f"data: {done.model_dump_json()}\n\n"


def _timestamp(seconds: float, separator: str) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def _cue_body(text: str) -> str:
    return "\n".join(stripped for line in text.splitlines() if (stripped := line.strip()))


def render_transcript(segments: list[TranscriptionSegment], response_format: str) -> str:
    vtt = response_format == "vtt"
    separator = "." if vtt else ","
    cues = []
    for index, segment in enumerate(segments, start=1):
        times = f"{_timestamp(segment.start, separator)} --> {_timestamp(segment.end, separator)}"
        header = times if vtt else f"{index}\n{times}"
        cues.append(f"{header}\n{_cue_body(segment.text)}")
    body = "\n\n".join(cues)
    return f"WEBVTT\n\n{body}\n" if vtt else f"{body}\n"


def transcript_render_plan(
    response_format: str, model: str
) -> tuple[str, Callable[..., Response | ErrorResponse] | None]:
    if response_format in ("json", "verbose_json"):
        return response_format, None
    if response_format == "text":
        return "json", lambda result: Response(content=result.text, media_type=_TRANSCRIPT_MEDIA_TYPE)

    def render(result) -> Response | ErrorResponse:
        if not result.segments:
            return create_error_response(
                f"{model} returned no segment timestamps, which response_format `{response_format}` requires",
                err_type="invalid_request_error",
            )
        return Response(content=render_transcript(result.segments, response_format), media_type=_TRANSCRIPT_MEDIA_TYPE)

    return "verbose_json", render
