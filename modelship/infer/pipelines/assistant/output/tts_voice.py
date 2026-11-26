"""`TtsVoice`: the pipeline's tts model saying one response's answer, in the voice, speed and audio format the
response asked for. It makes each spoken message's `Speech`, counts the audio sent into the client's playback, and
keeps what each message said, which truncation reads."""

from collections.abc import Callable

from modelship.infer.pipelines.assistant.calls import PipelineCalls
from modelship.infer.pipelines.assistant.codecs import G711_RATE, bytes_per_second, from_wav
from modelship.infer.pipelines.assistant.output.playback import Playback
from modelship.infer.pipelines.assistant.output.speech import SendAudio, Speech, Spoken
from modelship.logging import get_logger
from modelship.openai.protocol import ErrorResponse, RawSpeechResponse, SpeechRequest
from modelship.openai.protocol.realtime import AudioG711, AudioPcm, RealtimeError, Voice, VoiceId

logger = get_logger("realtime")

# a spoken answer goes out in pieces of this much audio
AUDIO_DELTA_SECONDS = 0.5


class TtsVoice:
    """*model* speaking through *calls* for session *session_id*, in *voice* at *speed*, its audio sent as
    *audio_format*. *playback* counts the audio sent, and *spoken* gets each message's record by item id."""

    def __init__(
        self,
        calls: PipelineCalls,
        model: str,
        session_id: str,
        audio_format: AudioPcm | AudioG711,
        voice: Voice | None,
        speed: float,
        playback: Playback,
        spoken: dict[str, Spoken],
    ):
        self._calls = calls
        self._model = model
        self._session_id = session_id
        self._kind = audio_format.type
        self._rate = audio_format.rate if isinstance(audio_format, AudioPcm) else G711_RATE
        self._voice = voice.id if isinstance(voice, VoiceId) else voice
        self._speed = speed
        self._playback = playback
        self._spoken = spoken
        self._bytes_per_second = bytes_per_second(self._kind, self._rate)
        self._step = int(self._bytes_per_second * AUDIO_DELTA_SECONDS) // 2 * 2

    def speech(self, item_id: str, send: SendAudio, cancelled: Callable[[], bool]) -> Speech:
        """The speech of message *item_id*, sending its audio through *send*; *cancelled* says whether its response
        was cancelled, after which the audio sent no longer plays."""

        async def played(audio: bytes) -> bool:
            if not await send(audio):
                return False
            # audio a cancel overtook doesn't play: the client stopped
            if not cancelled():
                self._playback.sent(len(audio) / self._bytes_per_second)
            return True

        speech = Speech(self.speak, self.encode, played, self._step, self._bytes_per_second)
        self._spoken[item_id] = speech.spoken
        return speech

    async def speak(self, text: str) -> bytes | RealtimeError:
        """The tts model's WAV of *text*, or why it couldn't say it."""
        request = SpeechRequest(
            input=text, model=self._model, voice=self._voice, speed=self._speed, response_format="wav"
        )
        try:
            result = await self._calls.speak(self._model, request, self._session_id)
        except Exception:
            logger.exception("Realtime session %s: speaking with %r failed", self._session_id, self._model)
            result = None
        if isinstance(result, RawSpeechResponse):
            return result.audio
        if isinstance(result, ErrorResponse):
            error = result.error
            return RealtimeError(message=error.message, type=error.type, code=error.code)
        return RealtimeError(
            message=f"Pipeline model {self._model!r} didn't speak the answer.", type="server_error", code="server_error"
        )

    def encode(self, wav: bytes) -> bytes:
        """The tts model's *wav* in the response's audio format."""
        return from_wav(wav, self._kind, self._rate)
