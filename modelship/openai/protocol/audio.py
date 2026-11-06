"""Audio endpoint schemas: transcription, translation, and text-to-speech."""

from typing import Literal

from fastapi import UploadFile
from pydantic import BaseModel, Field

from modelship.openai.protocol.base import OpenAIBaseModel

# ---------------------------------------------------------------------------
# Speech-to-text (transcription)
# ---------------------------------------------------------------------------

AudioResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]


class TranscriptionUsageAudio(OpenAIBaseModel):
    type: Literal["duration"] = "duration"
    seconds: float


class TranscriptionRequest(OpenAIBaseModel):
    file: UploadFile
    model: str = Field(..., description="ID of the model to use.")
    language: str | None = None
    prompt: str = Field(default="")
    response_format: AudioResponseFormat = Field(default="json")
    timestamp_granularities: list[Literal["word", "segment"]] = Field(
        alias="timestamp_granularities[]", default_factory=list
    )
    stream: bool | None = False
    temperature: float = Field(default=0.0)
    # modelship extension (not in OpenAI spec) — used for deterministic sampling
    # in evals/regression tests. Documented in docs/extensions.md.
    seed: int | None = None


class TranscriptionResponse(OpenAIBaseModel):
    text: str
    usage: TranscriptionUsageAudio


class TranscriptionWord(OpenAIBaseModel):
    word: str
    start: float = Field(..., description="start time in seconds")
    end: float = Field(..., description="end time in seconds")


class TranscriptionSegment(OpenAIBaseModel):
    id: int
    seek: int = 0
    start: float = Field(..., description="start time in seconds")
    end: float = Field(..., description="end time in seconds")
    text: str
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class TranscriptionResponseVerbose(OpenAIBaseModel):
    # `task` is missing from the openai-openapi schema definition but is
    # explicitly emitted by the OpenAI API (see the spec's own example payload)
    # and expected by strict client deserializers that don't tolerate unknown
    # field absence. Pinned to "transcribe" — Whisper's only valid value for
    # this route.
    task: Literal["transcribe"] = "transcribe"
    language: str
    duration: float = Field(..., description="The duration of the input audio in seconds.")
    text: str
    words: list[TranscriptionWord] = Field(default_factory=list)
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    usage: TranscriptionUsageAudio | None = None


# ---------------------------------------------------------------------------
# Speech-to-text (translation)
# ---------------------------------------------------------------------------


class TranslationRequest(OpenAIBaseModel):
    file: UploadFile
    model: str = Field(..., description="ID of the model to use.")
    prompt: str = Field(default="")
    response_format: AudioResponseFormat = Field(default="json")
    temperature: float = Field(default=0.0)
    # modelship extensions (not in OpenAI spec). Documented in
    # docs/extensions.md.
    stream: bool | None = False
    seed: int | None = None
    language: str | None = None
    to_language: str | None = None


class TranslationResponse(OpenAIBaseModel):
    text: str


class TranslationResponseVerbose(OpenAIBaseModel):
    # Same `task` story as TranscriptionResponseVerbose — emitted by the OpenAI
    # API but missing from the openapi.yaml schema definition. Pinned to
    # "translate".
    task: Literal["translate"] = "translate"
    language: str = Field(..., description="The language of the output translation (always `english`).")
    duration: float = Field(..., description="The duration of the input audio in seconds.")
    text: str
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    usage: TranscriptionUsageAudio | None = None


# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------


class SpeechRequest(OpenAIBaseModel):
    input: str = Field(..., max_length=4096, description="The text to generate audio for")
    model: str = Field(
        ...,
        description="The model to use for generation.",
    )
    voice: str = Field(
        ...,
        description="The voice to use for generation.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
    stream_format: Literal["sse", "audio"] = Field(
        default="audio",
        description="The stream format to return the audio in.",
    )


class SpeechResponse(OpenAIBaseModel):
    audio: str | None = Field(default=None, description="The generated audio data encoded in base 64")
    type: Literal["speech.audio.delta", "speech.audio.done"] = Field(
        ...,
        description="Type of audio chunk",
    )


class RawSpeechResponse(BaseModel):
    audio: bytes = Field(..., description="full audio file bytes")
    media_type: Literal["audio/wav"] = Field(default="audio/wav", description="audio bytes media type")


__all__ = [
    "AudioResponseFormat",
    "RawSpeechResponse",
    "SpeechRequest",
    "SpeechResponse",
    "TranscriptionRequest",
    "TranscriptionResponse",
    "TranscriptionResponseVerbose",
    "TranscriptionSegment",
    "TranscriptionUsageAudio",
    "TranscriptionWord",
    "TranslationRequest",
    "TranslationResponse",
    "TranslationResponseVerbose",
]
