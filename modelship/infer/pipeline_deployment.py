"""A pipeline's app: one Serve deployment per pipeline, which runs the pipeline's type and calls its gateway's models.
Experimental.

An assistant holds each `/v1/realtime` session in the replica that took it; the gateway relays the client's
WebSocket to it, and `StickyRouter` keeps every call of a session on that replica.
"""

from collections.abc import AsyncIterator

from ray import serve

from modelship.deploy.request_router import Routing, StickyRouter
from modelship.infer.base_pipeline import BasePipeline
from modelship.infer.deployment_base import DeploymentBase
from modelship.infer.pipelines.relay import RealtimeFrame, RealtimeOpening
from modelship.infer.pipelines.routes import ModelRoutes
from modelship.logging import configure_logging, get_logger
from modelship.metrics import stamp_gateway
from modelship.utils.config_schema import AssistantPipelineConfig, PipelineConfig, parse_pipeline_app_name

logger = get_logger("pipeline")


def load_pipeline(config: PipelineConfig, routes: ModelRoutes) -> BasePipeline:
    """What runs *config*'s type."""
    if isinstance(config, AssistantPipelineConfig):
        from modelship.infer.pipelines.assistant.assistant_pipeline import AssistantPipeline

        return AssistantPipeline(config, routes)
    raise ValueError(f"pipeline {config.name!r}: no pipeline type {type(config).__name__}")


@serve.deployment
class PipelineDeployment(DeploymentBase):
    @classmethod
    def routing(cls, config: PipelineConfig) -> Routing:
        return Routing(StickyRouter, pins=("realtime", "realtime_event"), unpins=("realtime_close",))

    async def __init__(self, config: PipelineConfig):
        configure_logging()
        self.config = config
        context = serve.get_replica_context()
        parsed = parse_pipeline_app_name(context.app_name)
        gateway = parsed[0] if parsed is not None else ""
        stamp_gateway(gateway)
        self.routes = ModelRoutes(gateway, context.replica_id.unique_id)
        self.routes.start()
        self.pipeline = load_pipeline(config, self.routes)
        await self.pipeline.start()
        logger.info("Pipeline %s: ready", config.name)

    async def __del__(self):
        if routes := getattr(self, "routes", None):
            await routes.close()
        if pipeline := getattr(self, "pipeline", None):
            await pipeline.shutdown()

    async def realtime(self, opening: RealtimeOpening) -> AsyncIterator[RealtimeFrame]:
        async for frame in self.pipeline.realtime(opening):
            yield frame

    async def realtime_event(self, key: str, frames: list[str | bytes]) -> bool:
        return await self.pipeline.realtime_event(key, frames)

    async def realtime_close(self, key: str) -> None:
        await self.pipeline.realtime_close(key)
