"""OpenAI Realtime (GA) events, as modelship sends and reads them. Our own models, so the engine doesn't depend
on `openai`; field names and shapes follow openai-python's `types.realtime`. Experimental."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from modelship.utils import random_uuid

SessionType = Literal["realtime", "transcription"]
Modality = Literal["text", "audio"]
Modalities = Annotated[list[Modality], Field(min_length=1)]
MaxOutputTokens = Annotated[int, Field(ge=1)] | Literal["inf"]
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
ResponseStatus = Literal["in_progress", "completed", "cancelled", "incomplete", "failed"]
ItemStatus = Literal["completed", "incomplete", "in_progress"]

SESSION_OBJECTS: dict[str, str] = {
    "realtime": "realtime.session",
    "transcription": "realtime.transcription_session",
}


def event_id() -> str:
    return f"event_{random_uuid()[:20]}"


def item_id() -> str:
    return f"item_{random_uuid()[:20]}"


# Conversation items. A message has the shape of a Responses input item, so the LLM call takes it as it is.


class InputText(BaseModel):
    type: Literal["input_text"]
    text: str


class OutputText(BaseModel):
    type: Literal["output_text"]
    text: str


class _Message(BaseModel):
    id: str = Field(default_factory=item_id)
    type: Literal["message"]
    object: Literal["realtime.item"] = "realtime.item"
    status: ItemStatus = "completed"


class UserMessage(_Message):
    role: Literal["user"]
    content: list[InputText]


class SystemMessage(_Message):
    role: Literal["system"]
    content: list[InputText]


class AssistantMessage(_Message):
    role: Literal["assistant"]
    content: list[OutputText]


Item = UserMessage | SystemMessage | AssistantMessage
Message = Annotated[Item, Field(discriminator="role")]


class ItemReference(BaseModel):
    """An item of the conversation, by id."""

    type: Literal["item_reference"]
    id: str


InputItem = Annotated[Message | ItemReference, Field(discriminator="type")]


# Session and response settings


class RealtimeReasoning(BaseModel):
    effort: ReasoningEffort | None = None


class RealtimeSession(BaseModel):
    """The session fields a pipeline answers with so far; more arrive as the pipeline uses them."""

    type: SessionType
    object: str
    id: str
    model: str
    instructions: str | None = None
    output_modalities: Modalities | None = None
    max_output_tokens: MaxOutputTokens | None = None
    reasoning: RealtimeReasoning | None = None


class SessionFields(BaseModel):
    """What a client's `session.update` may set. Fields the pipeline doesn't use yet are accepted and ignored."""

    model_config = ConfigDict(extra="allow")

    type: SessionType | None = None
    instructions: str | None = None
    output_modalities: Modalities | None = None
    max_output_tokens: MaxOutputTokens | None = None
    reasoning: RealtimeReasoning | None = None


class ResponseParams(BaseModel):
    """What a client's `response.create` may set for that response. Fields the pipeline doesn't use yet are
    accepted and ignored."""

    model_config = ConfigDict(extra="allow")

    conversation: Literal["auto", "none"] | None = None
    input: list[InputItem] | None = None
    instructions: str | None = None
    max_output_tokens: MaxOutputTokens | None = None
    metadata: dict[str, str] | None = None
    output_modalities: Modalities | None = None
    reasoning: RealtimeReasoning | None = None


# Client events


class _ClientEvent(BaseModel):
    event_id: str | None = None


class SessionUpdate(_ClientEvent):
    type: Literal["session.update"]
    session: SessionFields


class ItemCreate(_ClientEvent):
    type: Literal["conversation.item.create"]
    item: Message
    previous_item_id: str | None = None


class ItemRetrieve(_ClientEvent):
    type: Literal["conversation.item.retrieve"]
    item_id: str


class ItemDelete(_ClientEvent):
    type: Literal["conversation.item.delete"]
    item_id: str


class ResponseCreate(_ClientEvent):
    type: Literal["response.create"]
    response: ResponseParams = Field(default_factory=ResponseParams)


class ResponseCancel(_ClientEvent):
    type: Literal["response.cancel"]
    response_id: str | None = None


ClientEvent = SessionUpdate | ItemCreate | ItemRetrieve | ItemDelete | ResponseCreate | ResponseCancel
CLIENT_EVENTS: TypeAdapter[ClientEvent] = TypeAdapter(Annotated[ClientEvent, Field(discriminator="type")])


# Server events


class StatusDetails(BaseModel):
    type: Literal["completed", "cancelled", "incomplete", "failed"]
    reason: Literal["client_cancelled", "max_output_tokens", "content_filter"] | None = None
    error: dict[str, str | None] | None = None


class RealtimeResponse(BaseModel):
    id: str
    object: Literal["realtime.response"] = "realtime.response"
    status: ResponseStatus = "in_progress"
    status_details: StatusDetails | None = None
    output: list[AssistantMessage] = Field(default_factory=list)
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
    """A server event; model fields are sent in their JSON shape."""
    dumped = {
        name: value.model_dump(mode="json") if isinstance(value, BaseModel) else value for name, value in fields.items()
    }
    return {"type": kind, "event_id": event_id(), **dumped}


def session_event(kind: Literal["session.created", "session.updated"], session: RealtimeSession) -> dict:
    return server_event(kind, session=session.model_dump(mode="json", exclude_none=True))


def response_event(kind: Literal["response.created", "response.done"], response: RealtimeResponse) -> dict:
    return server_event(kind, response=response)


def error_event(error: RealtimeError) -> dict:
    return server_event("error", error=error)
