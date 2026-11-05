"""Unit coverage for gateway-level transcription/translation response formats
(`text`/`srt`/`vtt`), rendered in `modelship.openai.utils.audio`."""

import pytest
from fastapi.responses import Response

from modelship.openai.protocol import (
    ErrorResponse,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranscriptionSegment,
    TranscriptionUsageAudio,
)
from modelship.openai.utils.audio import _timestamp, render_transcript, transcript_render_plan


def _seg(seg_id: int, start: float, end: float, text: str) -> TranscriptionSegment:
    return TranscriptionSegment(id=seg_id, start=start, end=end, text=text)


def _verbose(segments: list[TranscriptionSegment]) -> TranscriptionResponseVerbose:
    return TranscriptionResponseVerbose(language="en", duration=2.5, text="Hello world", segments=segments)


class TestTimestamp:
    @pytest.mark.parametrize(
        ("seconds", "srt", "vtt"),
        [
            (0, "00:00:00,000", "00:00:00.000"),
            (1.5, "00:00:01,500", "00:00:01.500"),
            (61.25, "00:01:01,250", "00:01:01.250"),
            (3661.5, "01:01:01,500", "01:01:01.500"),
            (36000, "10:00:00,000", "10:00:00.000"),
        ],
    )
    def test_formats_hours_minutes_seconds_millis(self, seconds, srt, vtt):
        assert _timestamp(seconds, ",") == srt
        assert _timestamp(seconds, ".") == vtt


class TestRenderTranscript:
    def test_srt_golden(self):
        segments = [_seg(0, 0.0, 1.0, "Hello"), _seg(1, 1.0, 2.5, "world")]
        assert render_transcript(segments, "srt") == (
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n2\n00:00:01,000 --> 00:00:02,500\nworld\n"
        )

    def test_vtt_golden(self):
        segments = [_seg(0, 0.0, 1.0, "Hello"), _seg(1, 1.0, 2.5, "world")]
        assert render_transcript(segments, "vtt") == (
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n\n00:00:01.000 --> 00:00:02.500\nworld\n"
        )

    def test_collapses_blank_lines_and_strips_each_line(self):
        segments = [_seg(0, 0.0, 1.0, "  line one\n\n  line two  ")]
        assert render_transcript(segments, "srt") == "1\n00:00:00,000 --> 00:00:01,000\nline one\nline two\n"


class TestRenderPlan:
    @pytest.mark.parametrize("fmt", ["json", "verbose_json"])
    def test_passthrough_formats_have_no_renderer(self, fmt):
        internal, render = transcript_render_plan(fmt, "m")
        assert internal == fmt
        assert render is None

    def test_text_asks_for_json_and_renders_plain(self):
        internal, render = transcript_render_plan("text", "m")
        assert internal == "json"
        assert render is not None
        result = render(TranscriptionResponse(text="hi there", usage=TranscriptionUsageAudio(seconds=1)))
        assert isinstance(result, Response)
        assert result.media_type == "text/plain; charset=utf-8"
        assert result.body == b"hi there"

    @pytest.mark.parametrize("fmt", ["srt", "vtt"])
    def test_subtitle_formats_ask_for_verbose(self, fmt):
        internal, render = transcript_render_plan(fmt, "m")
        assert internal == "verbose_json"
        assert render is not None
        result = render(_verbose([_seg(0, 0.0, 1.0, "Hello"), _seg(1, 1.0, 2.5, "world")]))
        assert isinstance(result, Response)
        assert result.media_type == "text/plain; charset=utf-8"
        assert b"-->" in result.body
        assert result.body.startswith(b"WEBVTT") == (fmt == "vtt")

    @pytest.mark.parametrize("fmt", ["srt", "vtt"])
    def test_no_segments_is_a_400(self, fmt):
        _, render = transcript_render_plan(fmt, "stt-model")
        assert render is not None
        result = render(_verbose([]))
        assert isinstance(result, ErrorResponse)
        assert result.error.type == "invalid_request_error"
        assert result._http_status == 400
        assert "stt-model" in result.error.message
