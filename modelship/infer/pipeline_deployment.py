"""A pipeline's audio app: one Serve deployment for each pipeline that takes audio. Experimental.

It keeps no session state. The gateway owns each `/v1/realtime` session and sends the app the audio to
scan, so any replica serves any call.
"""

from ray import serve

from modelship.logging import configure_logging, get_logger
from modelship.utils.config_schema import PipelineConfig

logger = get_logger("pipeline")


@serve.deployment
class PipelineDeployment:
    def __init__(self, config: PipelineConfig):
        configure_logging()
        self.config = config
        logger.info("Pipeline %s: audio app ready", config.name)
