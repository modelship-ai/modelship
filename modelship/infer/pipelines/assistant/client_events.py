"""Parsing the client's frames into Realtime client events, and the OpenAI error each one that doesn't parse gets."""

import json

from pydantic import ValidationError

from modelship.openai.protocol.realtime import CLIENT_EVENTS, ClientEvent, RealtimeError

# the union members pydantic names in a validation error's location; they aren't part of the client's param path
_UNION_TAGS = frozenset(
    {
        "user",
        "system",
        "assistant",
        "message",
        "function_call",
        "function_call_output",
        "item_reference",
        "input_text",
        "input_audio",
        "output_text",
        "output_audio",
        "audio/pcm",
        "audio/pcmu",
        "audio/pcma",
        "str",
        "VoiceId",
        "server_vad",
        "semantic_vad",
    }
)


def parse_event(text: str, accepted: list[str]) -> ClientEvent | RealtimeError:
    """The client event in the text frame *text*, or the error the client gets for it. *accepted* are the event types
    the session takes: a type outside them is the error, whatever else is wrong with the event."""
    try:
        event: ClientEvent = CLIENT_EVENTS.validate_json(text)
    except ValidationError as e:
        kind = _raw_field(text, "type")
        error = unsupported_type(kind, accepted) if kind is not None and kind not in accepted else _invalid(e, accepted)
        error.event_id = _raw_field(text, "event_id")
        return error
    if event.type not in accepted:
        error = unsupported_type(event.type, accepted)
        error.event_id = event.event_id
        return error
    return event


def unsupported_type(given: str, accepted: list[str]) -> RealtimeError:
    """The error for an event of type *given*, which the session doesn't take."""
    supported = ", ".join(repr(kind) for kind in accepted)
    return RealtimeError(
        message=f"Invalid value: {given!r}. Supported values are: {supported}.", code="invalid_value", param="type"
    )


def _param(loc: tuple[int | str, ...]) -> str:
    """A validation error's location as the client's param path, such as `item.content[0].type`."""
    parts: list[str] = []
    for part in loc:
        if isinstance(part, int) and parts:
            parts[-1] += f"[{part}]"
        elif part not in _UNION_TAGS:
            parts.append(str(part))
    return ".".join(parts)


def _missing(param: str) -> RealtimeError:
    return RealtimeError(
        message=f"Missing required parameter: {param!r}.", code="missing_required_parameter", param=param
    )


def _invalid(error: ValidationError, accepted: list[str]) -> RealtimeError:
    """The error for a client event that isn't valid, from the first problem pydantic found in it."""
    problem = error.errors()[0]
    if not problem["loc"]:
        if problem["type"] == "json_invalid":
            return RealtimeError(message="The event is not valid JSON.")
        if problem["type"] == "union_tag_invalid":
            return unsupported_type(str(problem.get("ctx", {}).get("tag")), accepted)
        if problem["type"] == "union_tag_not_found":
            return _missing("type")
        return RealtimeError(message='An event is a JSON object with a "type".')
    # the location's first part is the event's type, which isn't part of the param path
    param = _param(problem["loc"][1:])
    if problem["type"] == "missing":
        return _missing(param)
    return RealtimeError(message=f"{param}: {problem['msg']}", code="invalid_value", param=param)


def _raw_field(text: str, name: str) -> str | None:
    """Field *name* of an event that didn't validate, if it is a string."""
    try:
        raw = json.loads(text)
    except ValueError:
        return None
    found = raw.get(name) if isinstance(raw, dict) else None
    return found if isinstance(found, str) else None
