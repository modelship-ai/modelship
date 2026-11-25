"""A realtime session's calls to its pipeline's models, from the pipeline replica: each through the model's Serve
handle with the client's headers, identity and the disconnect registry, and a call left early stopped in the
replica too."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from modelship.infer.pipelines.assistant.model_calls import ModelCalls
from modelship.openai.protocol import SttCapabilities, SttStreamOptions, SttStreamText


class _Routes:
    """The model table: "qwen" served by *handle*, "loading" not served yet."""

    def __init__(self) -> None:
        self.qwen = MagicMock()

    def handle(self, model: str):
        return self.qwen if model == "qwen" else None

    def state(self, model: str) -> str:
        return {"qwen": "serving", "loading": "starting"}.get(model, "unknown")


def _calls(registry) -> tuple[ModelCalls, MagicMock]:
    routes = _Routes()
    return ModelCalls(routes, {"x-device": "kitchen"}, registry, "caller"), routes.qwen  # type: ignore[arg-type]


def _registry() -> MagicMock:
    registry = MagicMock()
    registry.set.remote = AsyncMock()
    return registry


class _Stream:
    """A model call's stream: *events*, then nothing until it's cancelled."""

    def __init__(self, *events) -> None:
        self.events = events
        self.cancelled = False

    async def __aiter__(self):
        for event in self.events:
            yield event
        await asyncio.Event().wait()

    def cancel(self) -> None:
        self.cancelled = True


class _One:
    def __init__(self, item) -> None:
        self.item = item

    async def __aiter__(self):
        yield self.item


def test_a_models_state_comes_from_the_table():
    calls, _ = _calls(_registry())
    assert [calls.model_state(m) for m in ("qwen", "loading", "nope")] == ["serving", "starting", "unknown"]


@pytest.mark.asyncio
async def test_an_answer_calls_the_model_with_the_connections_plumbing():
    registry = _registry()
    calls, handle = _calls(registry)
    assert calls.respond("nope", MagicMock(), "s1") is None
    handle.respond.options.return_value.remote.return_value = _Stream({"type": "response.created"})
    request = MagicMock()
    answer = calls.respond("qwen", request, "s1")
    assert answer is not None
    assert await answer.__anext__() == {"type": "response.created"}
    handle.respond.options.assert_called_once_with(stream=True, session_id="s1")
    (sent, headers, sent_registry, _req_id, identity), _ = handle.respond.options.return_value.remote.call_args
    assert (sent, headers, sent_registry, identity) == (request, {"x-device": "kitchen"}, registry, "caller")
    await answer.aclose()


@pytest.mark.asyncio
async def test_closing_an_answer_early_cancels_the_call_and_tells_the_replica():
    registry = _registry()
    calls, handle = _calls(registry)
    stream = _Stream({"type": "response.created"})
    handle.respond.options.return_value.remote.return_value = stream
    answer = calls.respond("qwen", MagicMock(), "s1")
    assert answer is not None
    await answer.__anext__()
    await answer.aclose()
    (_, _, _, req_id, _), _ = handle.respond.options.return_value.remote.call_args
    assert stream.cancelled
    registry.set.remote.assert_awaited_once_with(req_id)


@pytest.mark.asyncio
async def test_it_asks_a_model_what_it_can_do_as_speech_to_text():
    calls, handle = _calls(_registry())
    whisper = SttCapabilities(streaming=False, reports_language=True)
    handle.stt_capabilities.remote = AsyncMock(return_value=whisper)
    assert await calls.stt_capabilities("qwen") == whisper
    assert await calls.stt_capabilities("nope") is None


@pytest.mark.asyncio
async def test_a_turn_is_transcribed_with_the_connections_plumbing():
    calls, handle = _calls("registry")
    transcript = MagicMock()
    handle.transcribe.options.return_value.remote.return_value = _One(transcript)
    request = MagicMock()
    assert await calls.transcribe("qwen", b"RIFF", request, "s1") is transcript
    handle.transcribe.options.assert_called_once_with(stream=True, session_id="s1")
    (audio, sent, headers, registry, _req_id, identity), _ = handle.transcribe.options.return_value.remote.call_args
    expected = (b"RIFF", request, {"x-device": "kitchen"}, "registry", "caller")
    assert (audio, sent, headers, registry, identity) == expected
    assert await calls.transcribe("nope", b"RIFF", request, "s1") is None


@pytest.mark.asyncio
async def test_a_transcription_cancelled_mid_call_stops_the_model():
    registry = _registry()
    calls, handle = _calls(registry)
    stream = _Stream()
    handle.transcribe.options.return_value.remote.return_value = stream
    task = asyncio.ensure_future(calls.transcribe("qwen", b"RIFF", MagicMock(), "s1"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (_, _, _, _, req_id, _), _ = handle.transcribe.options.return_value.remote.call_args
    assert stream.cancelled
    registry.set.remote.assert_awaited_once_with(req_id)


@pytest.mark.asyncio
async def test_speech_is_asked_for_with_the_connections_plumbing():
    calls, handle = _calls("registry")
    speech = MagicMock()
    handle.speak.options.return_value.remote.return_value = _One(speech)
    request = MagicMock()
    assert await calls.speak("qwen", request, "s1") is speech
    handle.speak.options.assert_called_once_with(stream=True, session_id="s1")
    (sent, headers, registry, _req_id, identity), _ = handle.speak.options.return_value.remote.call_args
    assert (sent, headers, registry, identity) == (request, {"x-device": "kitchen"}, "registry", "caller")
    assert await calls.speak("nope", request, "s1") is None


@pytest.mark.asyncio
async def test_speech_cancelled_mid_call_stops_the_model():
    registry = _registry()
    calls, handle = _calls(registry)
    stream = _Stream()
    handle.speak.options.return_value.remote.return_value = stream
    task = asyncio.ensure_future(calls.speak("qwen", MagicMock(), "s1"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (_, _, _, req_id, _), _ = handle.speak.options.return_value.remote.call_args
    assert stream.cancelled
    registry.set.remote.assert_awaited_once_with(req_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["transcribe_push", "transcribe_finish"])
async def test_a_streams_audio_goes_to_the_replica_its_key_pins(method):
    calls, handle = _calls(_registry())
    text = SttStreamText(text="hi", audio_ms=160.0)
    call = getattr(handle, method)
    call.options.return_value.remote = AsyncMock(return_value=text)
    options = SttStreamOptions(language="de")
    assert await getattr(calls, method)("nope", "s1:t1", b"\0\0", 24000, options) is None
    assert await getattr(calls, method)("qwen", "s1:t1", b"\0\0", 24000, options) == text
    call.options.assert_called_once_with(session_id="s1:t1")
    (key, pcm, rate, sent, _req_id, identity), _ = call.options.return_value.remote.call_args
    assert (key, pcm, rate, sent, identity) == ("s1:t1", b"\0\0", 24000, options, "caller")


@pytest.mark.asyncio
async def test_closing_a_stream_goes_to_the_replica_its_key_pins():
    calls, handle = _calls(_registry())
    handle.transcribe_close.options.return_value.remote = AsyncMock(return_value=None)
    await calls.transcribe_close("nope", "s1:t1")
    await calls.transcribe_close("qwen", "s1:t1")
    handle.transcribe_close.options.assert_called_once_with(session_id="s1:t1")
    handle.transcribe_close.options.return_value.remote.assert_awaited_once_with("s1:t1")
