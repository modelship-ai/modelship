import asyncio
import contextlib
import os
import threading
from collections.abc import AsyncGenerator, Callable
from typing import Any

import numpy as np

from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import ModelshipModelConfig, RawRequestProxy
from modelship.infer.sherpa_onnx.family.kokoro import KokoroFrontend
from modelship.infer.sherpa_onnx.registry import REGISTRY, SherpaOnnxRegistryEntry, registry_name
from modelship.logging import get_logger
from modelship.openai.protocol import ErrorResponse, RawSpeechResponse, SpeechRequest
from modelship.openai.utils.audio import sse_stream_speech
from modelship.utils import base_request_id
from modelship.utils.audio import to_pcm16, to_wav
from modelship.utils.phonemes.tokens import TokenTable

logger = get_logger("infer.sherpa_onnx")


class SherpaOnnxInfer(BaseInfer):
    """In-process sherpa-onnx TTS loader, generic across `OfflineTtsConfig`
    families. v1 scope: the registry only curates kokoro, CPU only."""

    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        self.tts: Any = None
        self.entry: SherpaOnnxRegistryEntry | None = None
        self.frontend: KokoroFrontend | None = None

    def shutdown(self) -> None:
        self.tts = None
        if self.frontend is not None:
            self.frontend.close()
            self.frontend = None

    def __del__(self):
        self.shutdown()

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.tts, self.entry, self.frontend = await loop.run_in_executor(None, self._load)

    def _load(self) -> tuple[Any, SherpaOnnxRegistryEntry, KokoroFrontend]:
        import sherpa_onnx

        assert self.model_config.model is not None
        bundle_dir = self.model_config._resolved_path
        if bundle_dir is None:
            raise ValueError(
                f"sherpa_onnx deployment '{self.model_config.name}' has no resolved bundle directory. "
                f"Check driver logs for resolution errors."
            )
        entry = REGISTRY[registry_name(self.model_config.model)]
        assert isinstance(entry, SherpaOnnxRegistryEntry)

        cfg = sherpa_onnx.OfflineTtsConfig()
        family_cfg = getattr(cfg.model, entry.family)
        for slot, rel_path in entry.files.items():
            setattr(family_cfg, slot, os.path.join(bundle_dir, rel_path))
        for slot, rel_path in entry.dirs.items():
            setattr(family_cfg, slot, os.path.join(bundle_dir, rel_path))
        # CPU only: CoreML fails under Ray's forked actor processes (macOS TCC
        # check on the coremlcompiler XPC call).
        cfg.model.provider = "cpu"
        cfg.model.num_threads = max(1, round(self.model_config.num_cpus))

        logger.info("loading sherpa_onnx model %r for '%s'", self.model_config.model, self.model_config.name)
        tts = sherpa_onnx.OfflineTts(cfg)
        if tts.num_speakers != len(entry.voice_names):
            raise ValueError(
                f"sherpa_onnx deployment '{self.model_config.name}': bundle reports {tts.num_speakers} speakers, "
                f"registry entry {self.model_config.model!r} expects {len(entry.voice_names)}. The bundle on disk "
                f"doesn't match this registry entry."
            )
        tokens = TokenTable.load(os.path.join(bundle_dir, entry.files["tokens"]))
        chinese_lexicon = os.path.join(bundle_dir, entry.chinese_lexicon) if entry.chinese_lexicon else None
        frontend = KokoroFrontend(tokens, chinese_lexicon)
        return tts, entry, frontend

    async def warmup(self) -> None:
        pass

    def _resolve_sid(self, voice: str | None) -> int:
        assert self.entry is not None
        names = self.entry.voice_names
        if voice in names:
            return names.index(voice)
        if voice is not None and voice.isdigit() and int(voice) < len(names):
            return int(voice)
        config = self.model_config.sherpa_onnx_config
        default = config.default_voice if config and config.default_voice else names[0]
        if voice is None:
            logger.info("no voice for '%s', using the default %r", self.model_config.name, default)
        else:
            logger.info("unknown voice %r for '%s', using the default %r", voice, self.model_config.name, default)
        return names.index(default)

    async def create_speech(
        self, request: SpeechRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | RawSpeechResponse | AsyncGenerator[str, None]:
        request_id = f"tts-{base_request_id(raw_request)}"
        assert self.entry is not None
        sid = self._resolve_sid(request.voice)
        voice = self.entry.voice_names[sid]

        if "stream_format" in request.model_fields_set and request.stream_format == "sse":
            logger.info("%s: voice=%s sid=%d stream=sse", request_id, voice, sid)
            return self._stream(request.input, voice, sid, request.speed)

        logger.info("%s: voice=%s sid=%d", request_id, voice, sid)
        wav = await self._generate(request.input, voice, sid, request.speed)
        return RawSpeechResponse(audio=wav)

    def _gen_config(self, text: str, voice: str, sid: int, speed: float) -> Any:
        import sherpa_onnx

        assert self.frontend is not None
        gen = sherpa_onnx.GenerationConfig()
        gen.sid = sid
        gen.speed = speed
        gen.phoneme_codepoints = self.frontend.chunks(text, voice)
        return gen

    async def _generate(self, text: str, voice: str, sid: int, speed: float) -> bytes:
        def run(stop: threading.Event) -> Any:
            gen = self._gen_config(text, voice, sid, speed)
            return self.tts.generate("", gen, lambda _samples, _progress: 0 if stop.is_set() else 1)

        result = await self.run_blocking(run, exclusive=False)
        samples = np.asarray(result.samples, dtype=np.float32)
        return to_wav(samples, result.sample_rate)

    async def _stream(self, text: str, voice: str, sid: int, speed: float) -> AsyncGenerator[str, None]:
        async with contextlib.aclosing(self._generate_chunks(text, voice, sid, speed)) as chunks:
            async for event in sse_stream_speech(chunks):
                yield event

    async def _generate_chunks(
        self, text: str, voice: str, sid: int, speed: float
    ) -> AsyncGenerator[tuple[bytes, int], None]:
        sample_rate = self.tts.sample_rate

        def run(emit: Callable[[bytes], None], stop: threading.Event) -> None:
            def on_chunk(samples: Any, _progress: float) -> int:
                emit(to_pcm16(np.asarray(samples, dtype=np.float32)))
                return 0 if stop.is_set() else 1

            gen = self._gen_config(text, voice, sid, speed)
            self.tts.generate("", gen, on_chunk)

        async with contextlib.aclosing(self.stream_blocking(run, exclusive=False)) as chunks:
            async for pcm in chunks:
                yield pcm, sample_rate
