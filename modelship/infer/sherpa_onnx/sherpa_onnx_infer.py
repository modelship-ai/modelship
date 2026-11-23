"""What the sherpa_onnx loader's text-to-speech and speech-to-text models share: in-process, on the CPU, from a
registry bundle. The model-specific parts are in `family/`."""

import asyncio
from abc import abstractmethod

from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.infer.sherpa_onnx.registry import REGISTRY, AsrEntry, SherpaOnnxRegistryEntry, registry_name
from modelship.logging import get_logger

logger = get_logger("infer.sherpa_onnx")

# CoreML fails under Ray's forked actor processes (macOS TCC check on the coremlcompiler XPC call).
PROVIDER = "cpu"


class SherpaOnnxInfer(BaseInfer):
    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        assert model_config.model is not None
        self.registry_entry: SherpaOnnxRegistryEntry | AsrEntry = REGISTRY[registry_name(model_config.model)]

    @property
    def threads(self) -> int:
        return max(1, round(self.model_config.num_cpus))

    def bundle_dir(self) -> str:
        path = self.model_config._resolved_path
        if path is None:
            raise ValueError(
                f"sherpa_onnx deployment '{self.model_config.name}' has no resolved bundle directory. "
                f"Check driver logs for resolution errors."
            )
        return path

    async def start(self) -> None:
        logger.info("loading sherpa_onnx model %r for '%s'", self.model_config.model, self.model_config.name)
        await asyncio.get_running_loop().run_in_executor(None, self._load)

    @abstractmethod
    def _load(self) -> None:
        """Loads the model from `bundle_dir()`; runs in a worker thread."""

    async def warmup(self) -> None:
        pass
