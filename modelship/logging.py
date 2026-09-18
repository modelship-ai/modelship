import contextvars
import json
import logging
import os
import socket
from logging.handlers import SysLogHandler
from urllib.parse import urlparse

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
# Caller-identity fields for log correlation (see modelship.openai.auth.identity_key).
# identity_tier records which resolution tier produced the identity ("header" /
# "api_key" / "unscoped") so an unexpected shift to "unscoped" — e.g. a fronting
# proxy that stopped setting the trusted header — is visible in logs.
identity_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("identity", default=None)
identity_tier_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("identity_tier", default=None)

_configured = False


class RequestContextFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get(None)
        record.identity = identity_var.get(None)
        record.identity_tier = identity_tier_var.get(None)
        return True


class ModelshipJsonFormatter(logging.Formatter):
    def format(self, record):
        msg = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "pid": record.process,
        }
        req_id = getattr(record, "request_id", None)
        if req_id:
            msg["request_id"] = req_id
        identity = getattr(record, "identity", None)
        if identity:
            msg["identity"] = identity
        identity_tier = getattr(record, "identity_tier", None)
        if identity_tier:
            msg["identity_tier"] = identity_tier
        if record.exc_info and record.exc_info[1] is not None:
            msg["exception"] = self.formatException(record.exc_info)
        return json.dumps(msg)


class ModelshipTextFormatter(logging.Formatter):
    def format(self, record):
        ts = self.formatTime(record, self.datefmt)
        req_id = getattr(record, "request_id", None)
        req_part = f" [{req_id}]" if req_id else ""
        identity = getattr(record, "identity", None)
        id_part = f" id={identity}" if identity else ""
        base = f"[{ts}] {record.levelname:<8} {record.name}{req_part}{id_part} | {record.getMessage()}"
        if record.exc_info and record.exc_info[1] is not None:
            base += "\n" + self.formatException(record.exc_info)
        return base


TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_LIB_LOGGERS = (
    "ray",
    "ray.serve",
    "vllm",
    "transformers",
    "diffusers",
    "flashinfer",
    "huggingface_hub",
    "pywhispercpp",
)

# Env vars that libraries check internally when creating their own loggers.
# Setting these ensures the level sticks even when a library re-configures
# its loggers after our configure_logging() call (e.g. vLLM's init_logger).
_LIB_ENV_VARS = {
    "RAY_LOG_LEVEL": "ray",
    "RAY_SERVE_LOG_LEVEL": "ray",
    "VLLM_LOGGING_LEVEL": "vllm",
    "TRANSFORMERS_VERBOSITY": "transformers",
    "DIFFUSERS_VERBOSITY": "diffusers",
    "HF_HUB_VERBOSITY": "huggingface_hub",
}

# HuggingFace-style libraries expect lowercase verbosity values
# ("debug"/"info"/"warning"/"error"/"critical") rather than the standard
# logging-module names.
_LOWERCASE_LEVEL_LIBS = frozenset({"transformers", "diffusers", "huggingface_hub"})

# Above CRITICAL — effectively off, until the app drops to DEBUG/TRACE.
_LIB_SILENT_LEVEL = logging.CRITICAL + 10


def _resolve_app_level(level_name: str) -> int:
    name = level_name.upper()
    return TRACE if name == "TRACE" else getattr(logging, name, logging.INFO)


def compute_lib_level(app_level: int) -> tuple[int, str]:
    """Silent above DEBUG; at DEBUG or TRACE, floor at DEBUG — no library defines lower."""
    if app_level <= logging.DEBUG:
        return logging.DEBUG, "DEBUG"
    return _LIB_SILENT_LEVEL, "CRITICAL"  # name must be one library env vars recognize


def get_lib_log_config() -> tuple[int, str]:
    """Return (lib_level_int, lib_level_name) for the currently-configured app level."""
    return compute_lib_level(logging.getLogger("modelship").getEffectiveLevel())


def propagate_lib_log_env(level_name: str | None = None) -> None:
    """Set library-native log env vars from MSHIP_LOG_LEVEL.

    Must run BEFORE the libraries are imported so their module-level loggers
    pick up the right level. Uses setdefault so explicit user overrides win.
    Safe to call multiple times.
    """
    name = (level_name or os.environ.get("MSHIP_LOG_LEVEL", "INFO")).upper()
    app_level = _resolve_app_level(name)
    _, lib_level_name = compute_lib_level(app_level)
    for env_var, lib_name in _LIB_ENV_VARS.items():
        val = lib_level_name.lower() if lib_name in _LOWERCASE_LEVEL_LIBS else lib_level_name
        os.environ.setdefault(env_var, val)

    # Opt out of vLLM's import-time logging.config.dictConfig(), which closes Ray
    # Serve's MemoryHandler and crashes replica recovery (see TestVllmConfigureLoggingOptOut).
    # We own vLLM's level via setLevel + VLLM_LOGGING_LEVEL, so its dictConfig is pure
    # downside. setdefault so an explicit user value wins.
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

    # Superseded by _DownloadProgressLogger (infer/sources/hf.py), which bypasses this env var.


def _parse_syslog_target(target: str) -> SysLogHandler:
    """Parse a syslog URI and return a configured SysLogHandler.

    Supported formats:
        syslog://host:port       — UDP (default)
        syslog+tcp://host:port   — TCP
        syslog://host            — UDP, port 514
    """
    tcp = target.startswith("syslog+tcp://")
    url = target.replace("syslog+tcp://", "syslog://", 1)
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 514
    socktype = socket.SOCK_STREAM if tcp else socket.SOCK_DGRAM
    return SysLogHandler(address=(host, port), socktype=socktype)


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.environ.get("MSHIP_LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("MSHIP_LOG_FORMAT", "text").lower()
    log_target = os.environ.get("MSHIP_LOG_TARGET", "console").lower()

    app_level = _resolve_app_level(level_name)
    lib_level, _ = compute_lib_level(app_level)

    root_logger = logging.getLogger("modelship")
    root_logger.setLevel(app_level)
    root_logger.propagate = False

    if root_logger.handlers:
        return

    handler = _parse_syslog_target(log_target) if log_target.startswith("syslog") else logging.StreamHandler()
    handler.setLevel(app_level)

    if log_format == "json":
        handler.setFormatter(ModelshipJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    else:
        handler.setFormatter(ModelshipTextFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    handler.addFilter(RequestContextFilter())
    root_logger.addHandler(handler)

    # OpenTelemetry: add an OTLP log exporter as a second handler when configured.
    otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otel_endpoint:
        _setup_otel(root_logger, otel_endpoint, app_level)

    # Give libraries modelship's formatter, each via its own handler instance —
    # not modelship's own handler object, which e.g. ray.init(logging_level=...)
    # reaches into and mutates in place, corrupting modelship's own formatting too
    # since Handler objects are shared by reference, not copied.
    for name in _LIB_LOGGERS:
        lib_logger = logging.getLogger(name)
        lib_logger.handlers.clear()
        lib_logger.propagate = False
        lib_logger.setLevel(lib_level)
        lib_handler = _parse_syslog_target(log_target) if log_target.startswith("syslog") else logging.StreamHandler()
        lib_handler.setLevel(lib_level)
        lib_handler.setFormatter(handler.formatter)
        lib_handler.addFilter(RequestContextFilter())
        lib_logger.addHandler(lib_handler)
    propagate_lib_log_env(level_name)


# Set by the vLLM loader; the engine and worker processes it starts inherit it.
VLLM_CHILD_ENV = "MSHIP_VLLM_CHILD"


def configure_vllm_child_logging() -> None:
    """vLLM general plugin, run in every process that loads vLLM's plugins; only
    the engine and worker processes a replica starts lack modelship's logging."""
    if os.environ.get(VLLM_CHILD_ENV):
        configure_logging()


# pyright: reportMissingImports=false
def _setup_otel(root_logger: logging.Logger, endpoint: str, level: int) -> None:
    """Attach an OpenTelemetry log exporter to *root_logger*.

    Requires the ``opentelemetry-sdk`` and ``opentelemetry-exporter-otlp``
    packages (install via ``uv sync --extra otel``).
    """
    try:
        from opentelemetry.exporter.otlp.proto.grpc.log_exporter import (
            OTLPLogExporter,
        )
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    except ImportError:
        root_logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry packages are not installed. "
            "Install with: uv sync --extra otel"
        )
        return

    resource = Resource.create({SERVICE_NAME: "modelship"})
    provider = LoggerProvider(resource=resource)
    exporter = OTLPLogExporter(endpoint=endpoint, insecure=not endpoint.startswith("https"))
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

    otel_handler = LoggingHandler(level=level, logger_provider=provider)
    root_logger.addHandler(otel_handler)

    # Enable Ray's native OTel tracing so spans from Ray workers are exported too.
    os.environ.setdefault("RAY_TRACING_ENABLED", "1")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"modelship.{name}")
