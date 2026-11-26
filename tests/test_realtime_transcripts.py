"""Transcribing a session's committed turns: the language the stt context picks and remembers, the transcription
queue's order and events, each kind of turn audio, and recorded input taking a turn out of the buffer."""

import asyncio

import pytest

from modelship.infer.pipelines.assistant.conversation import Conversation
from modelship.infer.pipelines.assistant.input.input_buffer import InputAudioBuffer
from modelship.infer.pipelines.assistant.input.transcripts import (
    RecordedTurn,
    SilentTurn,
    SttContext,
    Transcript,
    TranscriptionQueue,
)
from modelship.infer.pipelines.assistant.input.turn_input import RecordedInput
from modelship.infer.pipelines.assistant.session_config import new_session
from modelship.openai.protocol import (
    ErrorInfo,
    ErrorResponse,
    SttCapabilities,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranscriptionUsageAudio,
)
from modelship.openai.protocol.realtime import AudioTranscription, InputAudio, UserMessage
from modelship.utils.config_schema import AssistantPipelineConfig

DETECTS = SttCapabilities(streaming=False, reports_language=True)
SCRIBE = AssistantPipelineConfig.model_validate(
    {"name": "scribe", "type": "assistant", "assistant_config": {"stt": "w"}}
)


class _Calls:
    """The stt model: answers each transcription with the next of *answers*, recording each request."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests: list = []

    async def transcribe(self, model, wav, request, session_id):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _context(calls=None, capabilities: SttCapabilities = DETECTS) -> SttContext:
    return SttContext(calls or _Calls(), "w", capabilities, new_session(SCRIBE, 3600))  # type: ignore[arg-type]


def _item() -> UserMessage:
    return UserMessage(type="message", role="user", content=[InputAudio(type="input_audio")])


class TestSttContext:
    def test_the_sessions_language_wins_over_the_one_detected(self):
        context = _context()
        context.detected = "de"
        assert context.language() == "de"
        context._session.audio.input.transcription = AudioTranscription(language="fr")
        assert context.language() == "fr"

    def test_it_detects_only_without_a_language_and_with_a_model_that_reports_one(self):
        assert _context().detects_language()
        assert not _context(capabilities=SttCapabilities(streaming=False, reports_language=False)).detects_language()
        context = _context()
        context.detected = "de"
        assert not context.detects_language()

    def test_it_remembers_the_first_language_reported(self):
        context = _context()
        context.remember_language("unknown")
        context.remember_language("")
        assert context.detected is None
        context.remember_language("de")
        context.remember_language("fr")
        assert context.detected == "de"


class TestRecordedTurn:
    @pytest.mark.asyncio
    async def test_a_transcript_and_the_language_it_reports(self):
        calls = _Calls(TranscriptionResponseVerbose(language="de", duration=1.0, text="hallo"))
        context = _context(calls)
        assert await RecordedTurn(b"wav", 1.0).transcribe(context) == Transcript(text="hallo")
        assert calls.requests[0].response_format == "verbose_json"
        assert context.detected == "de"

    @pytest.mark.asyncio
    async def test_with_a_language_it_asks_for_plain_json(self):
        calls = _Calls(TranscriptionResponse(text="hi", usage=TranscriptionUsageAudio(seconds=1)))
        context = _context(calls)
        context.detected = "en"
        await RecordedTurn(b"wav", 1.0).transcribe(context)
        assert calls.requests[0].response_format == "json" and calls.requests[0].language == "en"

    @pytest.mark.asyncio
    async def test_the_models_error(self):
        calls = _Calls(ErrorResponse(error=ErrorInfo(message="bad audio", type="invalid_request_error")))
        transcript = await RecordedTurn(b"wav", 1.0).transcribe(_context(calls))
        assert transcript.text is None and transcript.error is not None
        assert transcript.error["message"] == "bad audio"

    @pytest.mark.asyncio
    async def test_a_failed_call(self):
        transcript = await RecordedTurn(b"wav", 1.0).transcribe(_context(_Calls(RuntimeError("down"))))
        assert transcript.error is not None and transcript.error["code"] == "server_error"


class TestTranscriptionQueue:
    @pytest.mark.asyncio
    async def test_turns_are_transcribed_in_commit_order(self):
        conversation, sent = Conversation(), []
        release = asyncio.Event()

        class _Slow(SilentTurn):
            async def transcribe(self, context):
                await release.wait()
                return Transcript(text="first")

        async def send(event):
            sent.append(event)

        queue = TranscriptionQueue(_context(), conversation, send)
        first, second = _item(), _item()
        conversation.insert(first)
        conversation.insert(second)
        queue.add(first, _Slow(1.0))
        last = queue.add(second, SilentTurn(0.5))
        await asyncio.sleep(0)
        assert not sent
        release.set()
        await last
        completed = [e for e in sent if e["type"].endswith("transcription.completed")]
        assert [e["item_id"] for e in completed] == [first.id, second.id]
        assert completed[0]["usage"] == {"type": "duration", "seconds": 1.0}
        assert queue.last is last

    @pytest.mark.asyncio
    async def test_a_failed_transcription_sends_the_failed_event(self):
        conversation, sent = Conversation(), []

        async def send(event):
            sent.append(event)

        item = _item()
        conversation.insert(item)
        calls = _Calls(RuntimeError("down"))
        await TranscriptionQueue(_context(calls), conversation, send).add(item, RecordedTurn(b"wav", 1.0))
        assert [e["type"] for e in sent] == [
            "conversation.item.input_audio_transcription.failed",
            "conversation.item.done",
        ]

    @pytest.mark.asyncio
    async def test_a_cancelled_transcription_abandons_its_audio(self):
        abandoned = []

        class _Hangs(SilentTurn):
            async def transcribe(self, context):
                await asyncio.Event().wait()
                return Transcript(text="")

            def abandon(self):
                abandoned.append(True)

        async def send(event):
            pass

        queue = TranscriptionQueue(_context(), Conversation(), send)
        task = queue.add(_item(), _Hangs(1.0))
        await asyncio.sleep(0)
        queue.cancel()
        await queue.wait(1.0)
        assert task.cancelled() and abandoned == [True]


class TestRecordedInput:
    def test_take_cuts_the_turn_out_of_the_buffer(self):
        buffer = InputAudioBuffer()
        buffer.append(bytes(16000 * 2), 16000)
        audio = RecordedInput(buffer).take(250.0, 750.0, "turn")
        assert isinstance(audio, RecordedTurn)
        assert audio.seconds == 0.5 and buffer.start_ms == 750.0
