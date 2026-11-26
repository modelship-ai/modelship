"""A session's settings (`session_config`) and the responses its `Responder` starts: which settings a response takes
from its own parameters and which from the session, the LLM request it sends, the voice a spoken response says it in,
and the errors it starts with instead."""

import asyncio

import pytest

from modelship.infer.pipelines.assistant.conversation import Conversation
from modelship.infer.pipelines.assistant.output.playback import Playback
from modelship.infer.pipelines.assistant.output.responder import Responder
from modelship.infer.pipelines.assistant.output.response import ActiveResponse
from modelship.infer.pipelines.assistant.session_config import apply_update, check_update, new_session
from modelship.openai.protocol.realtime import (
    AudioPcm,
    InputAudio,
    InputText,
    ItemReference,
    ResponseParams,
    SessionFields,
    UserMessage,
)
from modelship.utils.config_schema import AssistantPipelineConfig

VOICE = AssistantPipelineConfig.model_validate(
    {"name": "voice", "type": "assistant", "assistant_config": {"stt": "w", "llm": "qwen", "tts": "kokoro"}}
)
CHAT = AssistantPipelineConfig.model_validate({"name": "chat", "type": "assistant", "assistant_config": {"llm": "q"}})


class _Calls:
    """The LLM: records each request and answers with no events."""

    def __init__(self) -> None:
        self.requests: list = []

    def respond(self, model, request, session_id):
        self.requests.append((model, request, session_id))

        async def answer():
            return
            yield

        return answer()


async def _sent(event: dict) -> None:
    pass


def _responder(config: AssistantPipelineConfig = VOICE, calls: _Calls | None = None):
    state = new_session(config, 3600)
    conversation = Conversation()
    responder = Responder(config, calls or _Calls(), state, conversation, Playback(), {}, _sent)  # type: ignore[arg-type]
    return responder, state, conversation


class TestSessionConfig:
    def test_a_new_session_follows_the_pipeline(self):
        voice, chat = new_session(VOICE, 3600), new_session(CHAT, 3600)
        assert (voice.type, voice.output_modalities) == ("realtime", ["audio"])
        assert voice.audio.input.turn_detection is not None
        assert (chat.output_modalities, chat.audio.input.turn_detection) == (["text"], None)

    def test_an_update_cant_change_the_sessions_type(self):
        state = new_session(VOICE, 3600)
        error = check_update(state, SessionFields(type="transcription"), VOICE, buffered=False)
        assert error is not None and error.param == "session.type"

    def test_audio_output_needs_a_tts_model(self):
        state = new_session(CHAT, 3600)
        error = check_update(state, SessionFields(output_modalities=["audio"]), CHAT, buffered=False)
        assert error is not None and error.param == "session.output_modalities"

    def test_a_new_format_needs_the_buffer_cleared(self):
        state = new_session(VOICE, 3600)
        update = SessionFields.model_validate({"audio": {"input": {"format": {"type": "audio/pcmu"}}}})
        assert check_update(state, update, VOICE, buffered=False) is None
        error = check_update(state, update, VOICE, buffered=True)
        assert error is not None and error.param == "session.audio.input.format"

    def test_apply_sets_what_the_update_gives_and_says_whether_input_changed(self):
        state = new_session(VOICE, 3600)
        assert apply_update(state, SessionFields(instructions="Be brief.")) is False
        assert state.instructions == "Be brief."
        update = SessionFields.model_validate({"audio": {"input": {"turn_detection": None}}})
        assert apply_update(state, update) is True
        assert state.audio.input.turn_detection is None


class TestResponder:
    @pytest.mark.asyncio
    async def test_the_request_takes_the_responses_settings_over_the_sessions(self):
        calls = _Calls()
        responder, state, conversation = _responder(calls=calls)
        state.instructions, state.max_output_tokens = "Session.", 100
        conversation.insert(UserMessage(type="message", role="user", content=[InputText(type="input_text", text="hi")]))
        params = ResponseParams(instructions="Response.", output_modalities=["text"])
        active = responder.start(params, None)
        assert isinstance(active, ActiveResponse)
        await active.task
        model, request, session_id = calls.requests[0]
        assert (model, session_id) == ("qwen", state.id)
        assert request.instructions == "Response." and request.max_output_tokens == 100
        assert request.stream and request.store is False
        assert active.response.status == "failed"

    @pytest.mark.asyncio
    async def test_it_waits_for_the_turns_committed_before_it(self):
        calls = _Calls()
        responder, _, _ = _responder(calls=calls)
        release = asyncio.Event()
        pending = asyncio.ensure_future(release.wait())
        active = responder.start(ResponseParams(output_modalities=["text"]), pending)
        assert isinstance(active, ActiveResponse)
        await asyncio.sleep(0.01)
        assert calls.requests == []
        release.set()
        await active.task
        assert len(calls.requests) == 1

    @pytest.mark.asyncio
    async def test_a_spoken_response_takes_its_own_format_and_voice(self):
        responder, state, _ = _responder()
        params = ResponseParams.model_validate({"audio": {"output": {"format": {"type": "audio/pcmu"}, "voice": "af"}}})
        active = responder.start(params, None)
        assert isinstance(active, ActiveResponse)
        assert active.response.audio is not None and active.response.audio.output is not None
        assert active.response.audio.output.format.type == "audio/pcmu"
        assert active.response.audio.output.voice == "af"
        # the session's own output stays as it was
        assert state.audio.output is not None and isinstance(state.audio.output.format, AudioPcm)
        await active.abandon()

    def test_audio_from_a_pipeline_without_tts_is_an_error(self):
        responder, _, _ = _responder(CHAT)
        error = responder.start(ResponseParams(output_modalities=["audio"]), None)
        assert not isinstance(error, ActiveResponse)
        assert error.param == "response.output_modalities"

    def test_an_input_reference_to_a_missing_item_is_an_error(self):
        responder, _, _ = _responder()
        params = ResponseParams(input=[ItemReference(type="item_reference", id="nope")], output_modalities=["text"])
        error = responder.start(params, None)
        assert not isinstance(error, ActiveResponse)
        assert (error.code, error.param) == ("item_not_found", "response.input[0].id")

    def test_audio_inside_an_input_item_is_an_error(self):
        responder, _, _ = _responder()
        item = UserMessage(type="message", role="user", content=[InputAudio(type="input_audio")])
        error = responder.start(ResponseParams(input=[item], output_modalities=["text"]), None)
        assert not isinstance(error, ActiveResponse)
        assert error.param == "response.input[0].content[0]"
