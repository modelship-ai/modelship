"""OpenAI Realtime (GA) events, as modelship sends and reads them. Our own models, so the engine doesn't depend
on `openai`; field names and shapes follow openai-python's `types.realtime`. Experimental."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from modelship.utils import random_uuid

SessionType = Literal["realtime", "transcription"]
Modality = Literal["text", "audio"]
MaxOutputTokens = Annotated[int, Field(ge=1)] | Literal["inf"]
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
ResponseStatus = Literal["in_progress", "completed", "cancelled", "incomplete", "failed"]

SESSION_OBJECTS: dict[str, str] = {
    "realtime": "realtime.session",
    "transcription": "realtime.transcription_session",
}


def event_id() -> str:
    return f"event_{random_uuid()[:20]}"


class RealtimeReasoning(BaseModel):
    effort: ReasoningEffort | None = None


class RealtimeSession(BaseModel):
    """The session fields a pipeline answers with so far; more arrive as the pipeline uses them."""

    type: SessionType
    object: str
    id: str
    model: str
    instructions: str | None = None
    output_modalities: list[Modality] | None = None
    max_output_tokens: MaxOutputTokens | None = None
    reasoning: RealtimeReasoning | None = None


class SessionFields(BaseModel):
    """What a client's `session.update` may set. Fields the pipeline doesn't use yet are accepted and ignored."""

    model_config = ConfigDict(extra="allow")

    type: SessionType | None = None
    instructions: str | None = None
    output_modalities: list[Modality] | None = None
    max_output_tokens: MaxOutputTokens | None = None
    reasoning: RealtimeReasoning | None = None


class ResponseParams(BaseModel):
    """What a client's `response.create` may set for that response. Fields the pipeline doesn't use yet are
    accepted and ignored."""

    model_config = ConfigDict(extra="allow")

    conversation: Literal["auto", "none"] | None = None
    input: list[dict[str, Any]] | None = None
    instructions: str | None = None
    max_output_tokens: MaxOutputTokens | None = None
    metadata: dict[str, str] | None = None
    output_modalities: list[Modality] | None = None
    reasoning: RealtimeReasoning | None = None


class StatusDetails(BaseModel):
    type: Literal["completed", "cancelled", "incomplete", "failed"]
    reason: Literal["client_cancelled", "max_output_tokens", "content_filter"] | None = None
    error: dict[str, str | None] | None = None


class RealtimeResponse(BaseModel):
    id: str
    object: Literal["realtime.response"] = "realtime.response"
    status: ResponseStatus = "in_progress"
    status_details: StatusDetails | None = None
    output: list[dict[str, Any]] = Field(default_factory=list)
    # None for a response kept out of the conversation
    conversation_id: str | None = None
    output_modalities: list[Modality]
    max_output_tokens: MaxOutputTokens
    metadata: dict[str, str] | None = None
    usage: dict[str, Any] | None = None


class RealtimeError(BaseModel):
    message: str
    type: str = "invalid_request_error"
    code: str | None = None
    param: str | None = None
    # the client event this error answers
    event_id: str | None = None


def server_event(kind: str, **fields: Any) -> dict:
    return {"type": kind, "event_id": event_id(), **fields}


def session_event(kind: Literal["session.created", "session.updated"], session: RealtimeSession) -> dict:
    return server_event(kind, session=session.model_dump(mode="json", exclude_none=True))


def response_event(kind: Literal["response.created", "response.done"], response: RealtimeResponse) -> dict:
    return server_event(kind, response=response.model_dump(mode="json"))


def error_event(error: RealtimeError) -> dict:
    return server_event("error", error=error.model_dump(mode="json"))
