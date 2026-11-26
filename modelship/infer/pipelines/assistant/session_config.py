"""A session's settings: what a new session starts with, from its pipeline's defaults, and the rules a
`session.update` follows."""

import time

from modelship.openai.protocol.realtime import (
    SESSION_OBJECTS,
    TURN_DETECTION,
    AudioInput,
    AudioOutput,
    RealtimeError,
    RealtimeSession,
    SemanticVad,
    ServerVad,
    SessionAudio,
    SessionFields,
    server_vad,
)
from modelship.utils import random_uuid
from modelship.utils.config_schema import AssistantPipelineConfig

# the top-level session fields a session.update sets as given
UPDATABLE = (
    "instructions",
    "output_modalities",
    "max_output_tokens",
    "reasoning",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
)
_INPUT_FIELDS = ("format", "transcription", "turn_detection")
_OUTPUT_FIELDS = ("format", "voice", "speed")
# the fields of one kind of turn detection that the other doesn't take
_SERVER_VAD_ONLY = frozenset({"threshold", "prefix_padding_ms", "silence_duration_ms", "idle_timeout_ms"})
_SEMANTIC_ONLY = frozenset({"eagerness"})


def _turn_detection(config: AssistantPipelineConfig) -> ServerVad | SemanticVad | None:
    """A new session's turn detection: server VAD on a pipeline with stt, unless its defaults say otherwise."""
    if config.assistant_config.stt is None:
        return None
    defaults = config.assistant_config.session_defaults
    given = defaults.audio.input if defaults is not None and defaults.audio is not None else None
    if given is None or "turn_detection" not in given.model_fields_set:
        return server_vad()
    if given.turn_detection is None:
        return None
    return TURN_DETECTION.validate_python(given.turn_detection.model_dump())


def new_session(config: AssistantPipelineConfig, lifetime_s: float) -> RealtimeSession:
    """The settings a session on *config* opens with, expiring after *lifetime_s*: a realtime session with an llm,
    a transcription session without, each from the pipeline's session defaults."""
    assistant = config.assistant_config
    kind = "realtime" if assistant.llm is not None else "transcription"
    defaults = assistant.session_defaults
    fields = defaults.model_dump(exclude_none=True, exclude={"audio"}) if defaults is not None else {}
    audio = SessionAudio(input=AudioInput(turn_detection=_turn_detection(config)))
    if kind == "realtime":
        fields.setdefault("output_modalities", ["audio"] if assistant.tts is not None else ["text"])
        audio.output = AudioOutput()
    fields["audio"] = audio
    return RealtimeSession(
        type=kind,
        object=SESSION_OBJECTS[kind],
        id=f"sess_{random_uuid()[:20]}",
        model=config.name,
        expires_at=int(time.time() + lifetime_s),
        **fields,
    )


def no_tts(config: AssistantPipelineConfig, param: str) -> RealtimeError | None:
    """The error for asking, at *param*, for audio output from a pipeline without a tts model; None with one."""
    if config.assistant_config.tts is not None:
        return None
    return RealtimeError(
        message=f"Pipeline {config.name!r} has no tts model to speak with; ask for text output.",
        code="invalid_value",
        param=param,
    )


def _audio_input(update: SessionFields) -> AudioInput | None:
    """The input audio settings *update* gives, if it gives any."""
    audio = update.audio
    return audio.input if audio is not None and "input" in audio.model_fields_set else None


def check_update(
    state: RealtimeSession, update: SessionFields, config: AssistantPipelineConfig, buffered: bool
) -> RealtimeError | None:
    """Why *update* can't apply to the session *state*, or None. *buffered* says whether the input audio buffer holds
    audio, which a change of format needs cleared."""
    if update.type is not None and update.type != state.type:
        return RealtimeError(
            message=f"This session is a {state.type} session; its type can't change.",
            code="invalid_value",
            param="session.type",
        )
    asks_audio = update.output_modalities is not None and "audio" in update.output_modalities
    if asks_audio and (error := no_tts(config, "session.output_modalities")) is not None:
        return error
    given = _audio_input(update)
    return _check_audio_input(state, given, config, buffered) if given is not None else None


def _misplaced_field(detection: ServerVad | SemanticVad) -> str | None:
    """A field *detection* was given that belongs to the other kind of turn detection, if any."""
    other = _SEMANTIC_ONLY if isinstance(detection, ServerVad) else _SERVER_VAD_ONLY
    return next((name for name in detection.model_extra or {} if name in other), None)


def _check_audio_input(
    state: RealtimeSession, given: AudioInput, config: AssistantPipelineConfig, buffered: bool
) -> RealtimeError | None:
    param = "session.audio.input.turn_detection"
    detection = given.turn_detection
    if isinstance(detection, ServerVad | SemanticVad):
        if (field := _misplaced_field(detection)) is not None:
            return RealtimeError(
                message=f"Unknown parameter: '{param}.{field}': {detection.type} doesn't take it.",
                code="unknown_parameter",
                param=f"{param}.{field}",
            )
        if isinstance(detection, ServerVad) and detection.idle_timeout_ms is not None:
            return RealtimeError(
                message="idle_timeout_ms is not supported yet.",
                code="invalid_value",
                param=f"{param}.idle_timeout_ms",
            )
        if config.assistant_config.stt is None:
            return RealtimeError(
                message=f"Pipeline {config.name!r} has no stt model to take audio.",
                code="invalid_value",
                param=param,
            )
    changes_format = "format" in given.model_fields_set and given.format != state.audio.input.format
    if changes_format and buffered:
        return RealtimeError(
            message="Clear the input audio buffer before changing its format.",
            code="invalid_value",
            param="session.audio.input.format",
        )
    return None


def apply_update(state: RealtimeSession, update: SessionFields) -> bool:
    """Sets the fields *update* gives on *state*, which `check_update` passed; whether it gave input audio settings,
    which turn detection has to follow."""
    given = _audio_input(update)
    if given is not None:
        for name in _INPUT_FIELDS:
            if name in given.model_fields_set:
                setattr(state.audio.input, name, getattr(given, name))
    audio_output = update.audio.output if update.audio is not None else None
    if audio_output is not None and state.audio.output is not None:
        for name in _OUTPUT_FIELDS:
            if name in audio_output.model_fields_set:
                setattr(state.audio.output, name, getattr(audio_output, name))
    for name in UPDATABLE:
        if name in update.model_fields_set:
            setattr(state, name, getattr(update, name))
    return given is not None
