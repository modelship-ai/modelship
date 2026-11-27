"""End-of-turn classification with Smart Turn v3.2 (pipecat-ai/smart-turn, BSD 2-Clause) on onnxruntime, for the
realtime pipeline's semantic VAD. `SmartTurnModel` is one model shared by every session of a replica: it hears a
turn's last 8 s of audio and says how likely the speaker is done. `SmartTurnLoader` builds it on first use."""

from __future__ import annotations

import asyncio

import numpy as np

from modelship.utils.sources import FileSource, check_file_source
from modelship.utils.whisper_features import SAMPLE_RATE, SAMPLES, log_mel

# pipecat-ai/smart-turn-v3 smart-turn-v3.2-cpu.onnx (int8) at revision f766f81, mirrored unmodified with its LICENSE.
_SMART_TURN_URL = (
    "https://github.com/modelship-ai/model-bundles/releases/download/turn/smart-turn/smart-turn-v3.2-cpu-2bb02631.onnx"
)
_SMART_TURN_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
_SMART_TURN_DEST = "turn/smart-turn-v3.2-cpu-2bb02631.onnx"


def smart_turn_source() -> FileSource:
    """The pinned Smart Turn model, for `download_model_source`."""
    return check_file_source(_SMART_TURN_URL, _SMART_TURN_SHA256, _SMART_TURN_DEST)


def last_8s(samples: np.ndarray) -> np.ndarray:
    """The last 8 s of 16 kHz *samples*, shorter audio padded with silence before it, as Smart Turn hears a turn."""
    if len(samples) >= SAMPLES:
        return samples[-SAMPLES:]
    return np.pad(samples, (SAMPLES - len(samples), 0))


class SmartTurnModel:
    """Smart Turn v3.2 on onnxruntime, one model for every session. Thread-safe."""

    def __init__(self, model_path: str) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        # Smart Turn's own settings for CPU inference
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        # each check allocates its tensors anew, so no memory stays reserved between checks
        options.enable_cpu_mem_arena = False
        self._session = onnxruntime.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

    def end_probability(self, pcm16: bytes, rate: int) -> float:
        """How likely the speaker of *pcm16*, a turn's 16-bit mono PCM at *rate*, has finished it."""
        import soxr

        from modelship.utils.audio import from_pcm16

        samples = from_pcm16(pcm16)
        if rate != SAMPLE_RATE:
            samples = soxr.resample(samples, rate, SAMPLE_RATE).astype(np.float32)
        features = log_mel(last_8s(samples))[None]
        (probability,) = self._session.run(None, {"input_features": features})
        return float(np.asarray(probability).item())


class SmartTurnLoader:
    """The replica's one `SmartTurnModel`, built from *model_path* when a session first asks for semantic VAD, so a
    replica whose sessions never do holds no model. A build that fails is tried again on the next ask."""

    def __init__(self, model_path: str) -> None:
        self._path = model_path
        self._model: SmartTurnModel | None = None
        self._lock = asyncio.Lock()

    async def load(self) -> SmartTurnModel:
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(SmartTurnModel, self._path)
            return self._model
