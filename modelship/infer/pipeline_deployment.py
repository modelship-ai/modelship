"""A pipeline's audio app: one Serve deployment for each pipeline that takes audio. Experimental.

It keeps no session state. The gateway owns each `/v1/realtime` session and sends the app the audio to
scan, so any replica serves any call.
"""

import asyncio

from ray import serve

from modelship.logging import configure_logging, get_logger
from modelship.utils.config_schema import PipelineConfig
from modelship.utils.downloads import locked_download
from modelship.utils.vad import SAMPLE_RATE, VadScanner, silero_vad_source

logger = get_logger("pipeline")


@serve.deployment
class PipelineDeployment:
    async def __init__(self, config: PipelineConfig):
        configure_logging()
        self.config = config
        self.scanner = VadScanner(await locked_download(silero_vad_source(), config.name))
        logger.info("Pipeline %s: audio app ready", config.name)

    async def scan(
        self, audio: bytes, rate: int, threshold: float, silence_duration_ms: int
    ) -> list[tuple[int, int | None]]:
        """The speech in *audio* (16-bit mono PCM at *rate*), as (start, end) in ms from its start; the end is
        None for speech that runs on to its end, or whose trailing silence is shorter than
        *silence_duration_ms*."""
        import soxr

        from modelship.utils.audio import from_pcm16

        samples = from_pcm16(audio)
        if rate != SAMPLE_RATE:
            samples = soxr.resample(samples, rate, SAMPLE_RATE)
        found = await asyncio.to_thread(self.scanner.scan, samples, threshold, silence_duration_ms)
        return [(segment.start_ms, segment.end_ms) for segment in found]
