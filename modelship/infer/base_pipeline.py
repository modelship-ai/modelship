"""What every pipeline type implements, as `BaseInfer` for a model's loader: each type serves its own usecase's
calls, and refuses the rest."""

import json
from collections.abc import AsyncIterator

from modelship.infer.pipelines.relay import RealtimeClosed, RealtimeFrame, RealtimeOpening
from modelship.openai.protocol.realtime import RealtimeError, error_event
from modelship.utils.config_schema import PipelineConfig

# RFC 6455 policy violation: the session can't open as asked
CLOSE_REFUSED = 1008


async def refused_realtime(name: str) -> AsyncIterator[RealtimeFrame]:
    """The stream of a `/v1/realtime` session on something that doesn't hold one."""
    error = RealtimeError(message=f"{name!r} doesn't serve /v1/realtime.", code="model_not_supported", param="model")
    yield json.dumps(error_event(error))
    yield RealtimeClosed(CLOSE_REFUSED)


class BasePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    async def start(self) -> None:
        """Loads what the pipeline runs before it takes calls."""

    async def shutdown(self) -> None:
        """Ends what the pipeline holds."""

    def realtime(self, opening: RealtimeOpening) -> AsyncIterator[RealtimeFrame]:
        """Holds a `/v1/realtime` session until it ends, streaming its frames for the gateway to relay."""
        return refused_realtime(self.config.name)

    async def realtime_event(self, key: str, frames: list[str | bytes]) -> bool:
        """Hands the client's *frames* to session *key*, in order; False when this replica doesn't hold it."""
        return False

    async def realtime_close(self, key: str) -> None:
        """Tells session *key* that its client left."""
