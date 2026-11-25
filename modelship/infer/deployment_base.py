"""The calls the gateway and pipelines make on a model's or a pipeline's app, under the same names whatever serves
them. Each one refuses here; `ModelDeployment` and `PipelineDeployment` serve their own usecase's."""

from collections.abc import AsyncIterator
from typing import Any

from modelship.deploy.request_router import AffinityRouter, Routing
from modelship.infer.base_infer import _NOT_SUPPORTED
from modelship.infer.base_pipeline import refused_realtime
from modelship.infer.pipelines.relay import RealtimeFrame, RealtimeOpening
from modelship.openai.protocol import SttCapabilities


class DeploymentBase:
    # the model's or the pipeline's config
    config: Any

    @classmethod
    def routing(cls, config: Any) -> Routing:
        """The request router of *config*'s app."""
        return Routing(AffinityRouter)

    async def generate(self, request, request_headers, disconnect_registry, request_id=None, identity=None):
        yield _NOT_SUPPORTED

    async def respond(
        self, request, request_headers, disconnect_registry, request_id=None, identity=None, response_id=None
    ):
        yield _NOT_SUPPORTED

    async def embed(self, request, request_headers, disconnect_registry, request_id=None, identity=None):
        yield _NOT_SUPPORTED

    async def stt_capabilities(self) -> SttCapabilities | None:
        """What this model can do as speech to text; None when it doesn't transcribe."""
        return None

    async def transcribe(
        self, audio_data, request, request_headers, disconnect_registry, request_id=None, identity=None
    ):
        yield _NOT_SUPPORTED

    async def translate(
        self, audio_data, request, request_headers, disconnect_registry, request_id=None, identity=None
    ):
        yield _NOT_SUPPORTED

    async def speak(self, request, request_headers, disconnect_registry, request_id=None, identity=None):
        yield _NOT_SUPPORTED

    async def imagine(self, request, request_headers, disconnect_registry, request_id=None, identity=None):
        yield _NOT_SUPPORTED

    async def edit_image(
        self, image_data, mask_data, request, request_headers, disconnect_registry, request_id=None, identity=None
    ):
        yield _NOT_SUPPORTED

    async def vary_image(
        self, image_data, request, request_headers, disconnect_registry, request_id=None, identity=None
    ):
        yield _NOT_SUPPORTED

    async def realtime(self, opening: RealtimeOpening) -> AsyncIterator[RealtimeFrame]:
        """A `/v1/realtime` session's frames, while it lasts."""
        async for frame in refused_realtime(self.config.name):
            yield frame

    async def realtime_event(self, key: str, frames: list[str | bytes]) -> bool:
        """Hands the client's *frames* to session *key*; False when this replica doesn't hold it."""
        return False

    async def realtime_close(self, key: str) -> None:
        """Tells session *key* that its client left."""
