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


class InputAudio(BaseModel):
    """A committed turn of the client's audio; the session keeps only its transcript."""

    type: Literal["input_audio"]
    # None until the turn is transcribed
    transcript: str | None = None


class OutputText(BaseModel):
    type: Literal["output_text"]
    text: str


class OutputAudio(BaseModel):
    """The assistant's spoken answer; the session keeps only its transcript."""

    type: Literal["output_audio"]
    transcript: str = ""


class _Message(BaseModel):
    id: str = Field(default_factory=item_id)
    type: Literal["message"]
    object: Literal["realtime.item"] = "realtime.item"
    status: ItemStatus = "completed"


class UserMessage(_Message):
    role: Literal["user"]
    content: list[Annotated[InputText | InputAudio, Field(discriminator="type")]]


class SystemMessage(_Message):
    role: Literal["system"]
    content: list[InputText]


class AssistantMessage(_Message):
    role: Literal["assistant"]
    content: list[Annotated[OutputText | OutputAudio, Field(discriminator="type")]]


Message = Annotated[UserMessage | SystemMessage | AssistantMessage, Field(discriminator="role")]


def call_id() -> str:
    return f"call_{random_uuid()[:20]}"


class FunctionCall(BaseModel):
    """The model calling one of the session's functions; the client runs it."""

    id: str = Field(default_factory=item_id)
    type: Literal["function_call"]
    object: Literal["realtime.item"] = "realtime.item"
    status: ItemStatus = "completed"
    call_id: str = Field(default_factory=call_id)
    name: str
    arguments: str


class FunctionCallOutput(BaseModel):
    """What a function the model called returned, from the client."""

    id: str = Field(default_factory=item_id)
    type: Literal["function_call_output"]
    object: Literal["realtime.item"] = "realtime.item"
    status: ItemStatus = "completed"
    call_id: str
    output: str


Item = UserMessage | SystemMessage | AssistantMessage | FunctionCall | FunctionCallOutput
ConversationItem = Annotated[Message | FunctionCall | FunctionCallOutput, Field(discriminator="type")]


class ItemReference(BaseModel):
    """An item of the conversation, by id."""

    type: Literal["item_reference"]
    id: str


InputItem = Annotated[Message | FunctionCall | FunctionCallOutput | ItemReference, Field(discriminator="type")]


# Tools


class FunctionTool(BaseModel):
    type: Literal["function"]
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolChoiceFunction(BaseModel):
    type: Literal["function"]
    name: str


ToolChoice = Literal["auto", "none", "required"] | ToolChoiceFunction


# Session and response settings


class RealtimeReasoning(BaseModel):
    effort: ReasoningEffort | None = None


class AudioPcm(BaseModel):
    type: Literal["audio/pcm"]
    # OpenAI's spec allows 24000 only
    rate: int = Field(default=24000, ge=8000, le=48000)


class AudioG711(BaseModel):
    """G.711 at 8 kHz: mu-law (`audio/pcmu`) or A-law (`audio/pcma`)."""

    type: Literal["audio/pcmu", "audio/pcma"]


AudioFormat = Annotated[AudioPcm | AudioG711, Field(discriminator="type")]


def _pcm24k() -> AudioPcm:
    return AudioPcm(type="audio/pcm")


class AudioTranscription(BaseModel):
    """How the session transcribes the client's audio. `model` is the pipeline's; other fields are ignored."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    language: str | None = None
    prompt: str | None = None


class AudioInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    format: AudioFormat = Field(default_factory=_pcm24k)
    transcription: AudioTranscription | None = None
    # null: the client commits each turn
    turn_detection: dict[str, Any] | None = None


class VoiceId(BaseModel):
    id: str


Voice = str | VoiceId
# OpenAI's range
Speed = Annotated[float, Field(ge=0.25, le=1.5)]


class AudioOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    format: AudioFormat = Field(default_factory=_pcm24k)
    # None: the tts model's default voice
    voice: Voice | None = None
    speed: Speed = 1.0


class SessionAudio(BaseModel):
    model_config = ConfigDict(extra="allow")

    input: AudioInput = Field(default_factory=AudioInput)
    # None in a transcription session
    output: AudioOutput | None = None


class ResponseAudioOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    format: AudioFormat | None = None
    voice: Voice | None = None


class ResponseAudio(BaseModel):
    model_config = ConfigDict(extra="allow")

    output: ResponseAudioOutput | None = None


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
    tools: list[FunctionTool] | None = None
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    audio: SessionAudio = Field(default_factory=SessionAudio)


class SessionFields(BaseModel):
    """What a client's `session.update` may set. Fields the pipeline doesn't use yet are accepted and ignored."""

    model_config = ConfigDict(extra="allow")

    type: SessionType | None = None
    instructions: str | None = None
    output_modalities: Modalities | None = None
    max_output_tokens: MaxOutputTokens | None = None
    reasoning: RealtimeReasoning | None = None
    tools: list[FunctionTool] | None = None
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    audio: SessionAudio | None = None


class ResponseParams(BaseModel):
    """What a client's `response.create` may set for that response. Fields the pipeline doesn't use yet are
    accepted and ignored."""

    model_config = ConfigDict(extra="allow")

    conversation: Literal["auto", "none"] | None = None
    audio: ResponseAudio | None = None
    input: list[InputItem] | None = None
    instructions: str | None = None
    max_output_tokens: MaxOutputTokens | None = None
    metadata: dict[str, str] | None = None
    output_modalities: Modalities | None = None
    reasoning: RealtimeReasoning | None = None
    tools: list[FunctionTool] | None = None
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None


# Client events


class _ClientEvent(BaseModel):
    event_id: str | None = None


class SessionUpdate(_ClientEvent):
    type: Literal["session.update"]
    session: SessionFields


class ItemCreate(_ClientEvent):
    type: Literal["conversation.item.create"]
    item: ConversationItem
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


class AudioAppend(_ClientEvent):
    type: Literal["input_audio_buffer.append"]
    # base64, in the session's input format
    audio: str


class AudioCommit(_ClientEvent):
    type: Literal["input_audio_buffer.commit"]


class AudioClear(_ClientEvent):
    type: Literal["input_audio_buffer.clear"]


ClientEvent = (
    SessionUpdate
    | ItemCreate
    | ItemRetrieve
    | ItemDelete
    | ResponseCreate
    | ResponseCancel
    | AudioAppend
    | AudioCommit
    | AudioClear
)
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
    output: list[AssistantMessage | FunctionCall] = Field(default_factory=list)
    # None for a response kept out of the conversation
    conversation_id: str | None = None
    output_modalities: list[Modality]
    # the format and voice of a spoken response
    audio: ResponseAudio | None = None
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
    return server_event(kind, session=session.model_dump(mode="json"))


def response_event(kind: Literal["response.created", "response.done"], response: RealtimeResponse) -> dict:
    return server_event(kind, response=response)


def error_event(error: RealtimeError) -> dict:
    return server_event("error", error=error)
