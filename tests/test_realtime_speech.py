"""Speaking a realtime answer: the text cut into pieces as the LLM writes it, the pieces spoken one at a time, what
was said in how much audio, the client's playback, and the tts model's WAV turned into the client's output format."""

import asyncio
import io
import warnings
import wave

import numpy as np
import pytest

from modelship.infer.pipelines.assistant.codecs import from_wav, read_wav, resample, to_pcm16
from modelship.infer.pipelines.assistant.output.chunker import FIRST_CHARS, MAX_CHARS, Chunker, speakable
from modelship.infer.pipelines.assistant.output.playback import TAIL_SECONDS, Playback
from modelship.infer.pipelines.assistant.output.speech import Speech
from modelship.openai.protocol.realtime import RealtimeError

LONG_SENTENCE = (
    "I turned on the kitchen lights, set the oven to two hundred degrees for the roast, and reminded you that the"
    " guests arrive at seven, so you have roughly an hour and a half to finish preparing everything before they"
    " come over for dinner tonight."
)


def _pieces(text: str, step: int) -> list[str]:
    """The pieces of *text* written *step* characters at a time."""
    chunker = Chunker()
    pieces = []
    for start in range(0, len(text), step):
        pieces += chunker.push(text[start : start + step])
    return pieces + chunker.flush()


class TestChunker:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # a short answer is spoken whole
            ("Sure, done.", ["Sure, done."]),
            # a first sentence under 20 characters joins the next
            (
                "Sure, done. The kitchen lights are on now. Anything else?",
                ["Sure, done. The kitchen lights are on now.", "Anything else?"],
            ),
            # the first piece ends at a clause; the next at a sentence
            (
                "The kitchen lights are on, and the oven is preheating. Dinner is at seven.",
                ["The kitchen lights are on,", "and the oven is preheating.", "Dinner is at seven."],
            ),
            # abbreviations, initials, decimals and numbered items don't end a sentence
            (
                "Dr. Smith and J. R. Tolkien met at 9.30 a.m. on Monday. It cost $1,000 per seat, e.g. a lot.",
                ["Dr. Smith and J. R. Tolkien met at 9.30 a.m. on Monday.", "It cost $1,000 per seat, e.g. a lot."],
            ),
            ("Here is the list:\n1. Buy milk and eggs.\n2. Call the plumber.", None),
            # closing quotes stay with their sentence
            (
                'The guard looked up. He said "stop right there now." Then he left the room quietly.',
                ["The guard looked up.", 'He said "stop right there now."', "Then he left the room quietly."],
            ),
            # CJK punctuation cuts without spaces
            ("今天天气很好，我们去公园散步吧。然后我们一起去吃一顿很好吃的晚饭，好吗？", None),
        ],
    )
    def test_pieces_end_at_sentences_and_clauses(self, text, expected):
        whole, by_char, by_word = _pieces(text, len(text)), _pieces(text, 1), _pieces(text, 4)
        assert whole == by_char == by_word
        assert "".join("".join(whole).split()) == "".join(text.split())
        if expected is not None:
            assert whole == expected

    def test_a_numbered_item_is_never_spoken_alone(self):
        pieces = _pieces("Here is the list:\n1. Buy milk and eggs.\n2. Call the plumber.", 1)
        assert pieces == ["Here is the list:\n1. Buy milk and eggs.", "2. Call the plumber."]

    def test_the_first_piece_without_a_boundary_ends_at_a_word_by_80_characters(self):
        text = "and then we walked along the river for a while talking about nothing in particular until the sun set"
        first = _pieces(text, 1)[0]
        assert len(first) <= FIRST_CHARS
        assert text.startswith(first + " ")

    def test_a_long_sentence_is_cut_at_its_last_clause_before_200_characters(self):
        pieces = _pieces(LONG_SENTENCE, 3)
        assert pieces == [
            "I turned on the kitchen lights,",
            "set the oven to two hundred degrees for the roast, and reminded you that the guests arrive at seven,",
            "so you have roughly an hour and a half to finish preparing everything before they come over for dinner"
            " tonight.",
        ]

    def test_a_long_run_without_punctuation_is_cut_at_words_and_no_piece_passes_its_limit(self):
        text = " ".join(["word"] * 150)
        pieces = _pieces(text, 7)
        assert len(pieces[0]) <= FIRST_CHARS
        assert all(len(piece) <= MAX_CHARS for piece in pieces)
        assert all(piece.split(" ") == ["word"] * len(piece.split(" ")) for piece in pieces)
        assert sum(len(piece.split(" ")) for piece in pieces) == 150

    def test_a_run_without_spaces_is_cut_at_the_limit(self):
        pieces = _pieces("x" * 450, 9)
        assert [len(piece) for piece in pieces] == [FIRST_CHARS, MAX_CHARS, 450 - FIRST_CHARS - MAX_CHARS]

    def test_a_boundary_at_the_end_of_the_text_waits_for_what_follows(self):
        chunker = Chunker()
        assert chunker.push("The kitchen lights are on.") == []
        assert chunker.push(" The") == ["The kitchen lights are on."]
        assert chunker.flush() == ["The"]

    def test_nothing_is_left_once_flushed(self):
        chunker = Chunker()
        chunker.push("Hello there")
        assert chunker.flush() == ["Hello there"]
        assert chunker.flush() == []


class TestSpeakable:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("**Sure**, the `kitchen` lights are _on_.", "Sure, the kitchen lights are on."),
            ("## Steps\n- Buy milk\n> quoted", "Steps Buy milk quoted"),
            ("See [the docs](https://example.com) or ![a cat](cat.png).", "See the docs or a cat."),
            ("Call snake_case_name now.", "Call snake_case_name now."),
            ("| a | b |", "a b"),
            ("1. First\n2. Second", "1. First 2. Second"),
        ],
    )
    def test_markdown_is_stripped(self, text, expected):
        assert speakable(text) == expected

    @pytest.mark.parametrize("text", ["---", "```python", "**", " . "])
    def test_nothing_to_say_is_empty(self, text):
        assert speakable(text) == ""


def _wav(samples: np.ndarray, rate: int, channels: int = 1) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2").tobytes())
    return out.getvalue()


class _Speaker:
    """A tts model speaking each text as 1 ms of audio per character, at 1000 Hz; *hold* keeps the first call
    waiting until it is released, and *fail* fails the call that many calls in."""

    def __init__(self, hold: bool = False, fail: int | None = None):
        self.texts: list[str] = []
        self.release = asyncio.Event()
        if not hold:
            self.release.set()
        self.fail = fail
        self.cancelled = False

    async def __call__(self, text: str) -> bytes | RealtimeError:
        self.texts.append(text)
        if len(self.texts) == 1:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.fail is not None and len(self.texts) > self.fail:
            return RealtimeError(message="voice broke", type="server_error", code="server_error")
        return _wav(np.full(len(text), 1000), 1000)


def _speech(speaker: _Speaker, step: int = 4) -> tuple[Speech, list[bytes]]:
    sent: list[bytes] = []

    async def send(audio: bytes) -> bool:
        sent.append(audio)
        return True

    return Speech(speaker, lambda wav: from_wav(wav, "audio/pcm", 1000), send, step, 2000), sent


class TestSpeech:
    @pytest.mark.asyncio
    async def test_the_first_piece_goes_alone_and_the_waiting_ones_go_in_one_call(self):
        speaker = _Speaker(hold=True)
        speech, sent = _speech(speaker)
        speech.write("Sure, I can help with that. The kitchen lights are on. ")
        await asyncio.sleep(0)
        speech.write("The oven is preheating now. Dinner is at seven tonight.")
        speaker.release.set()
        assert await speech.finish() is None
        assert speaker.texts == [
            "Sure, I can help with that.",
            "The kitchen lights are on. The oven is preheating now. Dinner is at seven tonight.",
        ]
        audio = b"".join(sent)
        assert len(audio) == 2 * sum(len(text) for text in speaker.texts)
        assert max(len(chunk) for chunk in sent) == 4

    @pytest.mark.asyncio
    async def test_cjk_pieces_are_joined_without_spaces(self):
        speaker = _Speaker(hold=True)
        speech, _ = _speech(speaker)
        speech.write("今天的天气非常好，我们一起去公园散步吧。然后")
        await asyncio.sleep(0)
        speech.write("我们一起去吃一顿很好吃的晚饭吧。之后我们再一起回家去休息一下好吗？还有别的事情吗？")
        speaker.release.set()
        await speech.finish()
        assert speaker.texts == [
            "今天的天气非常好，我们一起去公园散步吧。",
            "然后我们一起去吃一顿很好吃的晚饭吧。之后我们再一起回家去休息一下好吗？还有别的事情吗？",
        ]

    @pytest.mark.asyncio
    async def test_joined_pieces_stay_within_200_characters(self):
        speaker = _Speaker(hold=True)
        speech, _ = _speech(speaker)
        speech.write("This first sentence goes alone. ")
        await asyncio.sleep(0)
        speech.write("This is a sentence of about forty chars. " * 8)
        speaker.release.set()
        await speech.finish()
        assert len(speaker.texts) > 2
        assert all(len(text) <= MAX_CHARS for text in speaker.texts)

    @pytest.mark.asyncio
    async def test_markdown_is_not_spoken_and_a_piece_with_nothing_to_say_is_skipped(self):
        speaker = _Speaker()
        speech, _ = _speech(speaker)
        speech.write("**Sure**, the kitchen lights are on.\n---\n")
        await speech.finish()
        assert speaker.texts == ["Sure, the kitchen lights are on."]

    @pytest.mark.asyncio
    async def test_a_failed_piece_stops_the_speech(self):
        speaker = _Speaker(fail=0)
        speech, sent = _speech(speaker)
        speech.write("The kitchen lights are on now. The oven is preheating. ")
        failure = await speech.finish()
        assert failure is not None
        assert failure.message == "voice broke"
        assert speaker.texts == ["The kitchen lights are on now."]
        assert sent == []

    @pytest.mark.asyncio
    async def test_stopping_cancels_the_call_in_flight(self):
        speaker = _Speaker(hold=True)
        speech, sent = _speech(speaker)
        speech.write("The kitchen lights are on now. And")
        await asyncio.sleep(0)
        await speech.stop()
        assert speaker.cancelled
        assert sent == []


TRAIN = "The train leaves at seven forty. You don't need to book a seat, but it's cheaper if you do."


class TestSpoken:
    async def _spoken(self, *writes: str):
        # the fake speaker says 1 ms of audio a character
        speech, _ = _speech(_Speaker())
        for text in writes:
            speech.write(text)
        await speech.finish()
        return speech.spoken

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("audio_end_ms", "heard"),
        [
            (0, ""),
            (16, "The train leaves"),
            (32, "The train leaves at seven forty."),
            (61, "The train leaves at seven forty. You don't need to book a"),
            (90, TRAIN),
        ],
    )
    async def test_truncating_keeps_each_call_heard_and_the_cut_calls_words_in_proportion(self, audio_end_ms, heard):
        spoken = await self._spoken(TRAIN)
        assert spoken.ms == 90
        assert spoken.truncate(audio_end_ms, TRAIN) == heard
        assert spoken.ms == audio_end_ms

    @pytest.mark.asyncio
    async def test_the_transcript_keeps_its_markdown_and_spacing(self):
        text = "**Sure**, the  kitchen lights are on.\n\nThe oven is hot."
        spoken = await self._spoken(text[:20], text[20:])
        assert spoken.truncate(spoken.ms - 1, text) == "**Sure**, the  kitchen lights are on.\n\nThe oven is"

    @pytest.mark.asyncio
    async def test_a_piece_with_nothing_to_say_is_kept_only_when_audio_after_it_was_heard(self):
        text = "The kitchen lights are on now.\n---\n"
        spoken = await self._spoken(text)
        assert spoken.truncate(spoken.ms, text) == "The kitchen lights are on now."
        # a table's rule: a piece of its own, with nothing to say
        text = "| --- | --- | --- | --- |\nThe kitchen lights are on now."
        spoken = await self._spoken(text)
        assert spoken.truncate(0, text) == ""

    @pytest.mark.asyncio
    async def test_text_written_but_not_spoken_is_dropped(self):
        speaker = _Speaker(hold=True)
        speech, _ = _speech(speaker)
        speech.write("The kitchen lights are on now. And the oven")
        speaker.release.set()
        await asyncio.sleep(0.01)
        await speech.stop()
        assert speech.spoken.truncate(speech.spoken.ms, "The kitchen lights are on now. And the oven") == (
            "The kitchen lights are on now."
        )


class TestPlayback:
    def test_audio_plays_for_its_length_after_the_audio_before_it(self):
        now = [10.0]
        playback = Playback(lambda: now[0])
        assert not playback.playing()
        # three half seconds sent at once play one after another
        for _ in range(3):
            playback.sent(0.5)
        now[0] = 11.4 + TAIL_SECONDS
        assert playback.playing()
        now[0] = 11.5 + TAIL_SECONDS
        assert not playback.playing()

    def test_audio_sent_after_a_gap_plays_from_when_it_is_sent(self):
        now = [10.0]
        playback = Playback(lambda: now[0])
        playback.sent(0.5)
        now[0] = 20.0
        playback.sent(0.5)
        now[0] = 20.4 + TAIL_SECONDS
        assert playback.playing()

    def test_a_stop_ends_it(self):
        playback = Playback(lambda: 10.0)
        playback.sent(5)
        playback.stop()
        assert not playback.playing()


def _tone(frequency: float, rate: int, seconds: float = 0.5) -> np.ndarray:
    return (np.sin(2 * np.pi * frequency * np.arange(int(rate * seconds)) / rate) * 10000).astype(np.int16)


def _level(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


class TestOutputAudio:
    def test_g711_encoding_matches_audioop(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            audioop = pytest.importorskip("audioop")
        samples = np.arange(-32768, 32768, dtype=np.int32)
        wav = _wav(samples, 8000)
        pcm = samples.astype("<i2").tobytes()
        assert from_wav(wav, "audio/pcmu", 24000) == audioop.lin2ulaw(pcm, 2)
        assert from_wav(wav, "audio/pcma", 24000) == audioop.lin2alaw(pcm, 2)

    @pytest.mark.parametrize("kind", ["audio/pcmu", "audio/pcma"])
    def test_g711_round_trips_within_its_step(self, kind):
        tone = _tone(440, 8000)
        decoded = np.frombuffer(to_pcm16(from_wav(_wav(tone, 8000), kind, 24000), kind), dtype="<i2")
        assert np.max(np.abs(decoded.astype(np.int32) - tone)) <= 1024

    def test_downsampling_keeps_a_tone_it_can_hold_and_removes_one_it_cannot(self):
        kept = resample(_tone(1000, 24000), 24000, 8000)
        removed = resample(_tone(6000, 24000), 24000, 8000)
        assert len(kept) == len(removed) == 4000
        assert _level(kept) == pytest.approx(_level(_tone(1000, 8000)), rel=0.05)
        assert _level(removed) < 0.05 * _level(_tone(6000, 24000))

    def test_upsampling_keeps_the_tone(self):
        up = resample(_tone(440, 16000), 16000, 48000)
        assert len(up) == 24000
        assert _level(up) == pytest.approx(_level(_tone(440, 48000)), rel=0.02)

    def test_pcm_at_the_models_rate_passes_unchanged(self):
        tone = _tone(440, 24000)
        assert from_wav(_wav(tone, 24000), "audio/pcm", 24000) == tone.astype("<i2").tobytes()

    def test_g711_is_sent_at_8_khz(self):
        assert len(from_wav(_wav(_tone(440, 24000), 24000), "audio/pcmu", 24000)) == 4000

    def test_a_stereo_wav_is_mixed_to_mono(self):
        frames = np.array([[1000, 3000]] * 10, dtype=np.int16).reshape(-1)
        samples, rate = read_wav(_wav(frames, 16000, channels=2))
        assert (rate, samples.tolist()) == (16000, [2000] * 10)

    def test_a_wav_that_is_not_16_bit_is_refused(self):
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(1)
            wav.setframerate(8000)
            wav.writeframes(b"\x80" * 8)
        with pytest.raises(ValueError, match="8-bit"):
            read_wav(out.getvalue())
