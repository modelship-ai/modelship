"""Modelship Prometheus metrics — all exported via Ray's metrics agent.

When MSHIP_METRICS=true, metrics are defined using Ray's metrics APIs so they
flow through the same Ray metrics agent port as ray_*, serve_*, and vllm:*
metrics.  When disabled, no-op objects are exported so call sites need zero
conditional logic.

Two construction paths exist because metrics are emitted from two kinds of
process:

- ``ray.serve.metrics`` — only valid inside a Ray Serve replica (the gateway and
  the model-deployment actor). Used for the request/inference/model-load metrics.
- ``ray.util.metrics`` — the generic Ray metrics API, valid in any Ray worker or
  the driver. Used for the HA control-plane metrics emitted by the deploy
  coordinator (a plain detached actor), the state store (used inside the
  coordinator and the deploy driver), and the deploy driver itself — none of
  which run in a Serve replica context, so ``ray.serve.metrics`` would not bind.

Both APIs expose the same ``inc`` / ``set`` / ``observe`` surface, so the no-op
stubs below cover either and call sites stay identical.
"""

import time

from modelship.utils.runtime_env import env_setting

_ENABLED = env_setting("MSHIP_METRICS").lower() == "true"

# ---------------------------------------------------------------------------
# No-op metric stubs (used when metrics are disabled)
# ---------------------------------------------------------------------------


class _NoOpCounter:
    def inc(self, value=1.0, tags=None):
        pass

    def set_default_tags(self, tags):
        pass


class _NoOpGauge:
    def set(self, value, tags=None):
        pass

    def set_default_tags(self, tags):
        pass


class _NoOpHistogram:
    def observe(self, value, tags=None):
        pass

    def set_default_tags(self, tags):
        pass


# ---------------------------------------------------------------------------
# Latency bucket boundaries (in seconds)
# ---------------------------------------------------------------------------

_REQUEST_LATENCY_BOUNDARIES = [0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60]
# State-store ops are sub-second on a healthy backend; the long tail flags a slow
# or failing redis/file volume that would silently break self-heal durability.
_STATE_STORE_BOUNDARIES: list[float] = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]


# ---------------------------------------------------------------------------
# Per-gateway stamping
# ---------------------------------------------------------------------------
# Metrics emitted from a Serve replica (gateway + model actors) carry a `gateway`
# tag. We can't use Ray's set_default_tags for it: these metrics are built at
# import (no replica context), so the serve wrapper never registers its system
# tags, then errors when set_default_tags later runs inside a replica. Instead
# stamp_gateway records the name once per process and the wrapper injects it.

# Serve-metric keys carrying a `gateway` tag — wrapped so emissions stamp it.
_GATEWAY_SCOPED_KEYS = frozenset(
    {
        "request_total",
        "request_duration_seconds",
        "request_errors_total",
        "client_disconnects_total",
        "stream_chunks_total",
        "gateway_reconciles_total",
        "gateway_watch_errors_total",
        "generation_duration_seconds",
        "tts_generation_duration_seconds",
        "image_generation_duration_seconds",
        "transcription_duration_seconds",
        "embedding_duration_seconds",
        "resource_cleanup_errors_total",
    }
)

_GATEWAY = {"name": ""}  # mutated once per process by stamp_gateway


class _GatewayScopedMetric:
    """Wraps a Ray metric to stamp the `gateway` tag (set once via stamp_gateway)
    onto every emission, so no call site passes it."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        # Fires only on a miss (inc/set/observe are defined). Delegate any other
        # metric attr (name, description, Ray internals) to the wrapped metric;
        # guard _inner so a pre-__init__ lookup raises instead of recursing.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def _tags(self, tags):
        merged = dict(tags) if tags else {}
        merged["gateway"] = _GATEWAY["name"]
        return merged

    def inc(self, value=1.0, tags=None):
        self._inner.inc(value, self._tags(tags))

    def set(self, value, tags=None):
        self._inner.set(value, self._tags(tags))

    def observe(self, value, tags=None):
        self._inner.observe(value, self._tags(tags))


def stamp_gateway(gateway_name: str) -> None:
    """Record the gateway name once (from a replica's ``__init__``) so every
    gateway-scoped metric stamps it. No call site passes the gateway tag."""
    _GATEWAY["name"] = gateway_name


class _CreatedStampedCounter:
    def __init__(self, inner, created):
        self._inner = inner
        self._created = created
        self._seen = set()

    def inc(self, value=1.0, tags=None):
        key = frozenset((tags or {}).items())
        if key not in self._seen:
            self._seen.add(key)
            self._created.inc(time.time(), tags)
        self._inner.inc(value, tags)


def _build_metrics():
    """Construct real or no-op metric objects based on MSHIP_METRICS."""

    if not _ENABLED:
        return {
            # Gateway
            "request_total": _NoOpCounter(),
            "request_duration_seconds": _NoOpHistogram(),
            "request_errors_total": _NoOpCounter(),
            "client_disconnects_total": _NoOpCounter(),
            "stream_chunks_total": _NoOpCounter(),
            # Gateway HA routing (watch loop)
            "gateway_reconciles_total": _NoOpCounter(),
            "gateway_watch_errors_total": _NoOpCounter(),
            # Inference timing
            "generation_duration_seconds": _NoOpHistogram(),
            "tts_generation_duration_seconds": _NoOpHistogram(),
            "image_generation_duration_seconds": _NoOpHistogram(),
            "transcription_duration_seconds": _NoOpHistogram(),
            "embedding_duration_seconds": _NoOpHistogram(),
            # Resource cleanup
            "resource_cleanup_errors_total": _NoOpCounter(),
        }

    from ray.serve.metrics import Counter, Histogram

    # Ray's type stubs over-constrain tag_keys (Tuple[str] instead of variable-length
    # tuples) and boundaries (List[float] vs int literals). Suppressed with type: ignore.
    metrics = {
        # -- Gateway layer --
        "request_total": Counter(
            "modelship_request_total",
            description="Total inference requests by model and endpoint.",
            tag_keys=("model", "endpoint", "status", "gateway"),  # type: ignore[arg-type]
        ),
        "request_duration_seconds": Histogram(
            "modelship_request_duration_seconds",
            description="End-to-end request latency (gateway to response) in seconds.",
            boundaries=_REQUEST_LATENCY_BOUNDARIES,
            tag_keys=("model", "endpoint", "gateway"),  # type: ignore[arg-type]
        ),
        "request_errors_total": Counter(
            "modelship_request_errors_total",
            description="Total inference errors by model, endpoint, and error type.",
            tag_keys=("model", "endpoint", "error_type", "gateway"),  # type: ignore[arg-type]
        ),
        "client_disconnects_total": Counter(
            "modelship_client_disconnects_total",
            description="Total client disconnects during inference.",
            tag_keys=("model", "endpoint", "gateway"),  # type: ignore[arg-type]
        ),
        "stream_chunks_total": Counter(
            "modelship_stream_chunks_total",
            description="Total streaming chunks emitted.",
            tag_keys=("model", "gateway"),  # type: ignore[arg-type]
        ),
        # -- Gateway HA routing (watch loop) --
        "gateway_reconciles_total": Counter(
            "modelship_gateway_reconciles_total",
            description="Routing reconciles applied by a gateway replica from the coordinator.",
            tag_keys=("gateway",),
        ),
        "gateway_watch_errors_total": Counter(
            "modelship_gateway_watch_errors_total",
            description="Errors in a gateway replica's coordinator watch loop (retried).",
            tag_keys=("gateway",),
        ),
        # -- Inference timing --
        "generation_duration_seconds": Histogram(
            "modelship_generation_duration_seconds",
            description="Chat/text generation latency in seconds.",
            boundaries=_REQUEST_LATENCY_BOUNDARIES,
            tag_keys=("model", "gateway"),  # type: ignore[arg-type]
        ),
        "tts_generation_duration_seconds": Histogram(
            "modelship_tts_generation_duration_seconds",
            description="TTS inference latency in seconds.",
            boundaries=_REQUEST_LATENCY_BOUNDARIES,
            tag_keys=("model", "gateway"),  # type: ignore[arg-type]
        ),
        "image_generation_duration_seconds": Histogram(
            "modelship_image_generation_duration_seconds",
            description="Image generation latency in seconds.",
            boundaries=_REQUEST_LATENCY_BOUNDARIES,
            tag_keys=("model", "gateway"),  # type: ignore[arg-type]
        ),
        "transcription_duration_seconds": Histogram(
            "modelship_transcription_duration_seconds",
            description="Speech-to-text latency in seconds.",
            boundaries=_REQUEST_LATENCY_BOUNDARIES,
            tag_keys=("model", "gateway"),  # type: ignore[arg-type]
        ),
        "embedding_duration_seconds": Histogram(
            "modelship_embedding_duration_seconds",
            description="Embedding inference latency in seconds.",
            boundaries=_REQUEST_LATENCY_BOUNDARIES,
            tag_keys=("model", "gateway"),  # type: ignore[arg-type]
        ),
        # -- Resource cleanup --
        "resource_cleanup_errors_total": Counter(
            "modelship_resource_cleanup_errors_total",
            description="Errors during resource cleanup (engine shutdown, memory release).",
            tag_keys=("model", "component", "gateway"),  # type: ignore[arg-type]
        ),
    }
    return {k: _GatewayScopedMetric(v) if k in _GATEWAY_SCOPED_KEYS else v for k, v in metrics.items()}


def _build_util_metrics():
    """Construct the HA control-plane metrics via ``ray.util.metrics``.

    These are emitted by the deploy coordinator, the state store, and the deploy
    driver — none of which run inside a Serve replica, so they cannot use
    ``ray.serve.metrics``. The generic Ray metrics API works in any worker/driver
    and exports through the same agent with the same ``ray_`` prefix.
    """

    if not _ENABLED:
        return {
            # Gateway coordinator
            "gateway_replicas_behind": _NoOpGauge(),
            "model_up": _NoOpGauge(),
            # State store
            "state_store_operations_total": _NoOpCounter(),
            "state_store_operation_duration_seconds": _NoOpHistogram(),
            # Deploy coordinator
            "deploy_requests_total": _NoOpCounter(),
            "deploy_models_changed_total": _NoOpCounter(),
            "model_load_failures_total": _NoOpCounter(),
            "backend_deaths_total": _NoOpCounter(),
        }

    from ray.util.metrics import Counter, Gauge, Histogram

    return {
        # -- Gateway coordinator --
        "gateway_replicas_behind": Gauge(
            "modelship_gateway_replicas_behind",
            description="Running gateway replicas not on the gateway coordinator's current routing generation.",
            tag_keys=("gateway",),
        ),
        "model_up": Gauge(
            "modelship_model_up",
            description="1 while the deployment a gateway routes a model to can serve it, 0 while it can't and isn't deploying.",
            tag_keys=("gateway", "model"),
        ),
        # -- State store (durable HA state: deploy versions, conversations) --
        "state_store_operations_total": Counter(
            "modelship_state_store_operations_total",
            description="State-store operations by backend, op, and result.",
            tag_keys=("backend", "op", "result"),  # result: ok | error
        ),
        "state_store_operation_duration_seconds": Histogram(
            "modelship_state_store_operation_duration_seconds",
            description="State-store operation latency in seconds.",
            boundaries=_STATE_STORE_BOUNDARIES,
            tag_keys=("backend", "op"),
        ),
        # -- Deploy coordinator --
        "deploy_requests_total": _CreatedStampedCounter(
            Counter(
                "modelship_deploy_requests_total",
                description="Deploy requests finished, by outcome.",
                tag_keys=("gateway", "outcome"),
            ),
            Counter(
                "modelship_deploy_requests_created_seconds_total",
                description="Unix time each deploy_requests_total series was created.",
                tag_keys=("gateway", "outcome"),
            ),
        ),
        "deploy_models_changed_total": _CreatedStampedCounter(
            Counter(
                "modelship_deploy_models_changed_total",
                description="Models changed by a successful deploy request, by action.",
                tag_keys=("gateway", "action"),
            ),
            Counter(
                "modelship_deploy_models_changed_created_seconds_total",
                description="Unix time each deploy_models_changed_total series was created.",
                tag_keys=("gateway", "action"),
            ),
        ),
        "model_load_failures_total": _CreatedStampedCounter(
            Counter(
                "modelship_model_load_failures_total",
                description="Failed model loads reported by model replicas.",
                tag_keys=("gateway", "model", "loader", "fatal"),
            ),
            Counter(
                "modelship_model_load_failures_created_seconds_total",
                description="Unix time each model_load_failures_total series was created.",
                tag_keys=("gateway", "model", "loader", "fatal"),
            ),
        ),
        "backend_deaths_total": _CreatedStampedCounter(
            Counter(
                "modelship_backend_deaths_total",
                description="Backend process deaths reported by model replicas.",
                tag_keys=("gateway", "model"),
            ),
            Counter(
                "modelship_backend_deaths_created_seconds_total",
                description="Unix time each backend_deaths_total series was created.",
                tag_keys=("gateway", "model"),
            ),
        ),
    }


_metrics = _build_metrics()
_util_metrics = _build_util_metrics()

# -- Gateway --
REQUEST_TOTAL = _metrics["request_total"]
REQUEST_DURATION_SECONDS = _metrics["request_duration_seconds"]
REQUEST_ERRORS_TOTAL = _metrics["request_errors_total"]
CLIENT_DISCONNECTS_TOTAL = _metrics["client_disconnects_total"]
STREAM_CHUNKS_TOTAL = _metrics["stream_chunks_total"]

# -- Gateway HA routing (watch loop) --
GATEWAY_RECONCILES_TOTAL = _metrics["gateway_reconciles_total"]
GATEWAY_WATCH_ERRORS_TOTAL = _metrics["gateway_watch_errors_total"]

# -- Inference timing --
GENERATION_DURATION_SECONDS = _metrics["generation_duration_seconds"]
TTS_GENERATION_DURATION_SECONDS = _metrics["tts_generation_duration_seconds"]
IMAGE_GENERATION_DURATION_SECONDS = _metrics["image_generation_duration_seconds"]
TRANSCRIPTION_DURATION_SECONDS = _metrics["transcription_duration_seconds"]
EMBEDDING_DURATION_SECONDS = _metrics["embedding_duration_seconds"]

# -- Resource cleanup --
RESOURCE_CLEANUP_ERRORS_TOTAL = _metrics["resource_cleanup_errors_total"]

# -- HA control plane (ray.util.metrics — non-Serve emitters) --
# Gateway coordinator
GATEWAY_REPLICAS_BEHIND = _util_metrics["gateway_replicas_behind"]
MODEL_UP = _util_metrics["model_up"]
# State store
STATE_STORE_OPERATIONS_TOTAL = _util_metrics["state_store_operations_total"]
STATE_STORE_OPERATION_DURATION_SECONDS = _util_metrics["state_store_operation_duration_seconds"]
# Deploy coordinator
DEPLOY_REQUESTS_TOTAL = _util_metrics["deploy_requests_total"]
DEPLOY_MODELS_CHANGED_TOTAL = _util_metrics["deploy_models_changed_total"]
MODEL_LOAD_FAILURES_TOTAL = _util_metrics["model_load_failures_total"]
BACKEND_DEATHS_TOTAL = _util_metrics["backend_deaths_total"]
