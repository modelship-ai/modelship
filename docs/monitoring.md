# Monitoring & Logging

Each Ray node exposes Prometheus metrics on one port via Ray's metrics agent. When enabled, all of a node's metrics — Ray cluster, Ray Serve, vLLM engine, and custom Modelship metrics — are available on that node's scrape endpoint.

## Logging

Modelship uses a centralized logging system with structured output and request correlation. All application logs go through the `modelship.*` logger hierarchy, separate from library logs (Ray, vLLM, etc.).

### Configuration

| Env Var | Default | Description |
|---|---|---|
| `MSHIP_LOG_LEVEL` | `INFO` | App log level. Set to `TRACE` for request/response payloads, `DEBUG` for detailed diagnostics. Library logs are silent unless it is `DEBUG` or `TRACE`, which set them to `DEBUG`. |
| `MSHIP_LOG_FORMAT` | `text` | `text` for human-readable output, `json` for structured JSON lines (for log aggregation with ELK/Loki/Splunk). |
| `MSHIP_LOG_TARGET` | `console` | Log target. `console` writes to stderr; syslog URIs ship logs to a remote syslog server (see below). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | When set, logs are also exported to an OpenTelemetry collector via OTLP (see below). |

`MSHIP_LOG_FORMAT` (`--log-format`) is set once, on `mship start`, for every actor on the cluster. The level, target and OTLP endpoint (`--log-level`, `--log-target`, `--otel-endpoint`) belong to a node: set them on `mship start` or `mship join`, and every actor on that node uses them, library levels included. `mship deploy` and `stop` read these env vars only for their own output. Ray Serve's logs follow the node too, with two exceptions that use the head's level: Serve's HTTP proxy on every node, and a replica's first startup lines, logged before modelship's code runs in it.

### Log Levels

Library logs (Ray, vLLM, transformers) are silent unless the level is `DEBUG` or `TRACE`:

| Level | App logs (`modelship.*`) | Library logs |
|---|---|---|
| `TRACE` | Request/response payloads (audio bytes, transcription text, chat messages, etc.) | `DEBUG` |
| `DEBUG` | Detailed diagnostics, per-chunk details | `DEBUG` |
| `INFO` (default) | Startup, deployment, request summaries | silent |
| `WARNING` | Warnings only | silent |

### Request Correlation

Every API request is assigned a unique request ID that appears in all log lines for that request — both in the API gateway process and in the model deployment actor. This allows tracing a request end-to-end across Ray actor boundaries.

Text format example:
```
[2025-04-09 14:06:54] INFO     modelship.api [a1b2c3d4] | chat_completion model=llama messages=3 stream=True max_tokens=512
```

JSON format example:
```json
{"timestamp": "2025-04-09T14:06:54", "level": "INFO", "logger": "modelship.api", "message": "chat_completion model=llama messages=3 stream=True max_tokens=512", "request_id": "a1b2c3d4", "pid": 12345}
```

### Logger Names

| Logger | Scope |
|---|---|
| `modelship.startup` | Application initialization and shutdown |
| `modelship.api` | API gateway endpoints |
| `modelship.infer` | Base inference layer |
| `modelship.infer.deployment` | Ray Serve model deployment actor |
| `modelship.infer.vllm` | vLLM inference backend |
| `modelship.infer.diffusers` | Diffusers inference backend |
| `modelship.infer.diffusers.image` | Diffusers image generation |

### Syslog

Ship logs to a remote syslog server instead of stderr. Useful for centralized logging on bare-metal or Unraid setups without extra infrastructure.

```bash
# UDP (default)
mship start --log-target syslog://192.168.1.50:514

# TCP (reliable delivery)
mship start --log-target syslog+tcp://192.168.1.50:514

# Via environment variable
MSHIP_LOG_TARGET=syslog://192.168.1.50:514 mship start
```

Supported URI formats:

| URI | Protocol | Notes |
|---|---|---|
| `syslog://host:port` | UDP | Default, fire-and-forget |
| `syslog+tcp://host:port` | TCP | Reliable delivery |
| `syslog://host` | UDP | Port defaults to 514 |

The syslog target replaces the console handler — logs go to the syslog server only. The `--log-format` setting still applies (text or JSON).

### OpenTelemetry

Export logs to an OpenTelemetry collector via OTLP. Unlike syslog, OTel is additive — logs still go to the console (or syslog) handler, and are also shipped to the collector.

First, install the optional dependencies:

```bash
uv sync --extra otel
```

Then configure the endpoint:

```bash
# Via CLI
mship start --otel-endpoint http://collector:4317

# Via environment variable
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317 mship start
```

When OTel is enabled:
- Logs are exported via `BatchLogRecordProcessor` with `OTLPLogExporter` (gRPC)
- The service name is set to `modelship`
- `RAY_TRACING_ENABLED=1` is set automatically so Ray workers also export traces
- HTTPS endpoints are detected from the URI scheme; all others use insecure connections

If the `opentelemetry-sdk` and `opentelemetry-exporter-otlp` packages are not installed, a warning is logged and OTel is skipped.

## Architecture

```
Prometheus  ──scrape──>  Ray Metrics Agent (:8079)
                              |
                              |-- ray_node_*          Ray cluster: GPU, CPU, memory
                              |-- ray_serve_*         Ray Serve: HTTP requests, latency, replicas
                              |-- ray_vllm_*          vLLM engine: KV cache, TTFT, tokens, queue
                              |-- ray_modelship_*         Custom: per-model latency, errors
```

> **Note:** All metrics are prefixed with `ray_` by Ray's metrics agent. vLLM metric names are also sanitized (`:` → `_`), so e.g. the vLLM-native `vllm:kv_cache_usage_perc` becomes `ray_vllm_kv_cache_usage_perc`.

## Enabling Metrics

Metrics are enabled by default. Set `MSHIP_METRICS=false` to disable:

```bash
docker run --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token \
  -v ./models.yaml:/models.yaml \
  -v ./models-cache:/.cache \
  -p 8000:8000 -p 8079:8079 -p 8265:8265 \
  ghcr.io/modelship-ai/modelship:latest-cuda start --config /models.yaml --ray-dashboard-host 0.0.0.0 --enable-ray-auth
```

> The dashboard always starts and binds `127.0.0.1` by default — `--ray-dashboard-host 0.0.0.0` above is what makes the published `8265` port actually reachable. Only do this on a trusted/private network (see [troubleshooting.md](troubleshooting.md)). `--enable-ray-auth` makes the now-reachable dashboard require a bearer token — retrieve it with `docker exec <container> cat /home/modelship/.ray/auth_token`, or set your own with `-e MSHIP_RAY_AUTH_TOKEN`.

| Env Var | Default | Description |
|---|---|---|
| `MSHIP_METRICS` | `true` | Master toggle for modelship's own metrics on the whole cluster (`--no-metrics` on `mship start`); also pins the Ray metrics export port. |
| `MSHIP_METRICS_PORT` | `8079` on `start`, random on `join` | This node's metrics port (`--metrics-port`). |

`MSHIP_METRICS=false` turns modelship's own metrics into no-ops and leaves port 8079 free. Ray's
per-node exporter has no off switch — it keeps serving Ray's system metrics on a random port.

## Connecting to Prometheus

Add Modelship as a scrape target in your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: modelship
    scrape_interval: 15s
    static_configs:
      - targets: ["<modelship-host>:8079"]
```

In a multi-node cluster every node serves only its own metrics, so Prometheus scrapes each node. The head lists every node's address and port in a service discovery file, which Prometheus can read when it runs on the head's host:

```yaml
scrape_configs:
  - job_name: modelship
    file_sd_configs:
      - files: ["/tmp/ray/prom_metrics_service_discovery.json"]
```

Otherwise pin `--metrics-port` on every node and list the nodes as static targets. The Helm chart pins it, and its PodMonitor scrapes each pod's `metrics` port.

The alert rules expect the scrape job to be named `modelship`. The examples above use it, and the chart's PodMonitor sets it on every target whatever the release or `nameOverride`.

## Connecting to Grafana

A pre-built Grafana dashboard is included at [`docs/grafana-dashboard.json`](grafana-dashboard.json).

To import it:

1. Open Grafana and go to **Dashboards > Import**
2. Upload `grafana-dashboard.json` or paste its contents
3. Select your Prometheus datasource when prompted

At the top the dashboard has **Gateway**, **Model**, and **Node** dropdowns. The Gateway dropdown filters every `ray_modelship_*` panel to a single gateway (or All); the vLLM/Ray-Serve/GPU rows stay cluster-wide since those metrics aren't per-gateway (see [The `gateway` dimension](#the-gateway-dimension)). The Model dropdown also filters the vLLM panels.

The dashboard has 9 rows:

| Row | What it shows | Metric sources |
|---|---|---|
| **Overview** | Request rate, error rate, in-flight requests, models loaded, client disconnects | `ray_modelship_*`, `ray_serve_*` |
| **Latency** | Gateway P50/P95/P99, per-model latency, per-usecase latency (generate, TTS, image, STT, embed), per-model time to first token | `ray_modelship_*` |
| **vLLM Engine** | KV cache usage, TTFT, inter-token latency, token throughput, queue depth, preemptions, prefix cache hit rate | `ray_vllm_*` |
| **GPU & System** | GPU utilization, GPU memory, CPU, system memory (cluster aggregate) | `ray_node_*` |
| **Ray Serve** | Health check latency, request count, deployment processing latency, HTTP request latency | `ray_serve_*` |
| **Operational** | Model load time, load failures, resource cleanup errors, streaming chunks/s | `ray_modelship_*`, `ray_serve_*` |
| **Alerts** | Error rate %, KV cache usage, queue depth, TTFT P99, client disconnects, preemptions, GPU memory | `ray_modelship_*`, `ray_vllm_*`, `ray_node_*` |
| **Cluster / HA** | Model app status, gateway reconciles, Serve requests per model, gateway replicas behind the routing generation, gateway watch errors, state-store latency/errors, models changed by deploys | `ray_modelship_*`, `ray_serve_*` |
| **Per-Node Resources** | GPU utilization/memory, CPU, system memory broken out by node (filtered by the Node dropdown) | `ray_node_*` |

> **Deploying via the Helm chart:** set `grafanaDashboard.enabled=true` to ship this dashboard as a ConfigMap the Grafana sidecar auto-imports (and `prometheusRule.enabled=true` for the alert rules below) — no manual import needed. See the chart README.

## Alerting

A standalone Prometheus alerting rules file is included at [`docs/prometheus-alerts.yml`](prometheus-alerts.yml). The Grafana dashboard also has a dedicated **Alerts** row with threshold lines on the key panels.

### Importing Alert Rules

Add the rules file to your Prometheus config:

```yaml
rule_files:
  - /path/to/prometheus-alerts.yml
```

Then reload Prometheus (`kill -HUP <pid>` or `POST /-/reload` if `--web.enable-lifecycle` is set).

### Alert Reference

#### Critical (page-worthy)

| Alert | Condition | For | Description |
|---|---|---|---|
| `ModelshipHighErrorRate` | Errors > 5% of requests, cancelled ones excluded | 5m | Significant portion of requests are failing |
| `ModelshipHTTP5xxRatio` | 5xx > 5% of a gateway's HTTP requests, `/health` and `/readyz` excluded | 5m | Gateway is answering server errors |
| `ModelshipModelDown` | `ray_modelship_model_up` == 0 | 5m | A gateway has no running replica of a model to route to; the model answers 503 |
| `ModelshipKVCacheExhausted` | KV cache usage > 95% | 5m | Requests will queue or be preempted |
| `ModelshipModelLoadFailure` | A model replica failed to load in the last 15m | 0m | A model can't come up; its replica log has the reason |
| `ModelshipBackendCrashLooping` | A model's backend died 3+ times in 15m | 0m | `llama-server` or vLLM's engine keeps crashing |
| `ModelshipTargetDown` | `up{job="modelship"}` == 0 | 5m | Prometheus cannot scrape a node |
| `ModelshipStateStoreErrors` | Any state-store op errors | 5m | Durable HA state failing — self-heal at risk |
| `ModelshipDeployJobFailed` | `kube_job_failed` (condition true) > 0 | 0m | Deploy Job failed; last upgrade may not be applied (needs kube-state-metrics) |

#### Warning (investigate)

| Alert | Condition | For | Description |
|---|---|---|---|
| `ModelshipHighP99Latency` | P99 of one model's successful requests of one operation > 30s | 5m | End-to-end latency is very high |
| `ModelshipModelHighErrorRate` | One model's errors > 5% of its requests, cancelled ones excluded, above 0.05 requests/s | 5m | A model is failing while the cluster-wide rate may look fine |
| `ModelshipHighQueueDepth` | Waiting requests > 10 | 5m | vLLM engine is falling behind |
| `ModelshipPreemptions` | Preemption rate > 0 | 5m | GPU memory pressure causing request eviction |
| `ModelshipClientDisconnects` | Cancelled requests > 1/min | 5m | Clients timing out or dropping connections |
| `ModelshipGPUMemoryPressure` | Available GPU memory < 1 GB | 5m | GPU is nearly out of memory |
| `ModelshipHighTTFT` | TTFT P99 > 5s | 5m | Users waiting too long for first token |
| `ModelshipModelHighTTFT` | One model's time to first token P99 > 5s | 5m | Streaming requests to a model wait too long, on any loader that streams |
| `ModelshipAppUnhealthy` | `ray_serve_application_status` == 2 (UNHEALTHY) | 5m | A Serve application has an unhealthy deployment |
| `ModelshipDeployStuck` | `ray_serve_application_status` == 5 (DEPLOYING) | 30m | A Serve application is waiting for capacity or a slow model load |
| `ModelshipDeployFailed` | A deploy request failed in the last 15m | 0m | The request was rolled back; the gateway serves its last committed models |
| `ModelshipGatewayRoutingDivergence` | `gateway_replicas_behind` > 0 | 10m | A gateway replica is routing from a stale table |
| `ModelshipRayWorkerDown` | Ray worker pod not running or crash-looping | 5m | Cluster capacity degraded (needs kube-state-metrics); a worker hosting no replica is NotReady by design |

### Tuning Thresholds

All thresholds are starting points. Adjust based on your deployment:

- **Error rate**: 5% is aggressive — if you run small models that occasionally OOM, raise to 10%.
- **P99 latency**: 30s works for chat completions with long outputs. For embeddings or TTS, consider lowering to 5-10s by adding rules for their `operation`.
- **Queue depth**: 10 assumes a single vLLM instance. Scale proportionally with replicas.
- **KV cache**: 95% is the danger zone. If you use prefix caching heavily, 90% may be more appropriate.
- **TTFT**: 5s is generous. For interactive chat, consider 2-3s.
- **GPU memory**: 1 GB threshold assumes you're not running anything else on the GPU. Raise if you have shared workloads.

## Health and readiness

Two endpoints are always available regardless of the metrics toggle.

**`/health`** — cheap liveness probe. Returns 200 as soon as the gateway is up (before model deployments finish):

```bash
curl http://localhost:8000/modelship/health
# {"status": "ok", "uptime_s": 12.3}
```

**`/readyz`** — readiness. Returns 200 when at least one model is expected and every expected model has a deployment the gateway routes to; 503 with the same JSON body while any model is still pending. **A gateway with no models configured at all is never ready — `/readyz` stays 503 indefinitely**, even though `models_expected` and `models_pending` are both empty (there is nothing to be "pending" on, but nothing to serve either, so it never reports ready). This matters for a genuinely model-less control-plane node (e.g. a k8s head-only install with an empty `models.yaml`) or an external LB/Ingress health check pointed at `/readyz` before any model is configured — both stay permanently unhealthy by design, not by bug. Bodies carry full state so a single poll tells you what's loaded and what's outstanding:

```bash
curl http://localhost:8000/modelship/readyz
# 200 when ready:
# {
#   "status": "ok",
#   "ready": true,
#   "uptime_s": 420.5,
#   "models_loaded":   ["kokoro", "llm", "nomic-embed", "whisper"],
#   "models_expected": ["llm", "kokoro", "whisper", "nomic-embed"],
#   "models_pending":  []
# }
```

Use `/health` for Kubernetes liveness probes and `/readyz` for readiness probes — `/readyz` returning 503 prevents a service from flipping traffic onto the pod before models are loaded.

## Modelship Metrics Reference

Custom metrics are exported through Ray's metrics agent with a `ray_` prefix. Per-model/per-gateway metrics use `ray.serve.metrics` (they're emitted inside Serve replicas); the HA control-plane metrics use `ray.util.metrics` (emitted by the gateway coordinator, state store, and deploy coordinator, which are not Serve replicas).

### The `gateway` dimension

Every per-model and per-gateway metric below carries a `gateway` tag identifying which gateway emitted it, so multiple gateways sharing one Ray cluster stay distinguishable (the Grafana dashboard exposes a **Gateway** dropdown built on it). The tag is stamped automatically from `MSHIP_GATEWAY_NAME` — the deploy driver sets it and forwards it to every replica via `runtime_env`, so no call site passes it explicitly.

Cluster-scoped metrics are **not** per-gateway, because the thing they measure is shared cluster-wide: the state store (one shared backend), and all inherited `ray_vllm_*` / `ray_serve_*` / `ray_node_*` metrics (engine/cluster level). The dashboard's Gateway dropdown therefore filters the `ray_modelship_*` per-model panels but leaves the vLLM/GPU/Ray-Serve panels cluster-wide.

### Gateway

| Metric | Type | Tags | Description |
|---|---|---|---|
| `ray_modelship_request_total` | Counter | `model`, `operation`, `status`, `gateway` | Requests by model and API method (`status`: `ok`, `error`, `cancelled` when the client left first) |
| `ray_modelship_request_duration_seconds` | Histogram | `model`, `operation`, `status`, `gateway` | End-to-end request latency |
| `ray_modelship_request_errors_total` | Counter | `model`, `operation`, `error_type`, `gateway` | Errors: `validation_error`, `inference_error`, `stream_error` (a failure after the stream started), `model_not_found`, `model_not_ready`, `http_<status>` (rejected before dispatch), `unhandled` |
| `ray_modelship_stream_chunks_total` | Counter | `model`, `gateway` | Streaming chunks emitted |

`operation` is the API method: `create_chat_completion`, `create_response` (background runs included, counted when they end), `create_response_ws` (one per WebSocket turn), `compact_response`, `create_embeddings`, `create_transcriptions`, `create_translations`, `create_speech`, `create_image`, `create_image_edit`, `create_image_variation`. A request for a model the gateway doesn't serve, and a WebSocket frame that isn't a valid request, is counted with `model=""`.

### Model Deployment

| Metric | Type | Tags | Description |
|---|---|---|---|
| `ray_modelship_model_load_failures_total` | Counter | `model`, `loader`, `fatal`, `gateway` | Failed model loads, reported by the replica to the deploy coordinator (`fatal`: `true` when the deploy doesn't retry it) |
| `ray_modelship_model_load_failures_created_seconds_total` | Counter | `model`, `loader`, `fatal`, `gateway` | Unix time each `model_load_failures_total` series was created |
| `ray_modelship_backend_deaths_total` | Counter | `model`, `gateway` | Deaths of a model's backend process (`llama-server`, vLLM's engine), reported by its replica |
| `ray_modelship_backend_deaths_created_seconds_total` | Counter | `model`, `gateway` | Unix time each `backend_deaths_total` series was created |

### Inference Timing

| Metric | Type | Tags | Description |
|---|---|---|---|
| `ray_modelship_generation_duration_seconds` | Histogram | `model`, `gateway` | Chat/text generation latency |
| `ray_modelship_tts_generation_duration_seconds` | Histogram | `model`, `gateway` | Text-to-speech latency |
| `ray_modelship_image_generation_duration_seconds` | Histogram | `model`, `gateway` | Image generation latency |
| `ray_modelship_transcription_duration_seconds` | Histogram | `model`, `gateway` | Speech-to-text latency |
| `ray_modelship_embedding_duration_seconds` | Histogram | `model`, `gateway` | Embedding latency |
| `ray_modelship_time_to_first_token_seconds` | Histogram | `model`, `gateway` | From a streaming request reaching its replica to its first content, reasoning or tool-call token (`vllm`, `llama_server`) |

### Resource Cleanup

| Metric | Type | Tags | Description |
|---|---|---|---|
| `ray_modelship_resource_cleanup_errors_total` | Counter | `model`, `component`, `gateway` | Errors during engine/model cleanup |

### HA Control Plane

These cover the multi-node / HA machinery deployed by the Helm chart (gateway coordinator, pluggable state store, gateway watch loop). The first eight carry a `gateway` tag; the rest are cluster-scoped.

| Metric | Type | Tags | Description |
|---|---|---|---|
| `ray_modelship_gateway_reconciles_total` | Counter | `gateway` | Routing reconciles a gateway replica applied from the coordinator |
| `ray_modelship_gateway_watch_errors_total` | Counter | `gateway` | Errors in a gateway replica's coordinator watch loop (retried) |
| `ray_modelship_gateway_replicas_behind` | Gauge | `gateway` | Running gateway replicas not on the gateway coordinator's current routing generation |
| `ray_modelship_model_up` | Gauge | `gateway`, `model` | 1 while the deployment the gateway routes a model to can serve it, 0 when it can't; not set while that deployment is missing or deploying with no running replica |
| `ray_modelship_deploy_requests_total` | Counter | `gateway`, `outcome` | Deploy requests finished (`outcome`: `succeeded`, `failed`, `cancelled`), including those cancelled while queued |
| `ray_modelship_deploy_requests_created_seconds_total` | Counter | `gateway`, `outcome` | Unix time each `deploy_requests_total` series was created |
| `ray_modelship_deploy_models_changed_total` | Counter | `gateway`, `action` | Models changed by a successful deploy request (`action`: `add`, `remove`, `rescale`) |
| `ray_modelship_deploy_models_changed_created_seconds_total` | Counter | `gateway`, `action` | Unix time each `deploy_models_changed_total` series was created |
| `ray_modelship_state_store_operations_total` | Counter | `backend`, `op`, `result` | State-store ops (`op`: `get`/`set`/`delete`; `result`: `ok`/`error`) |
| `ray_modelship_state_store_operation_duration_seconds` | Histogram | `backend`, `op` | State-store operation latency |

The deploy coordinator's four counters (`deploy_requests_total`, `deploy_models_changed_total`, `model_load_failures_total`, `backend_deaths_total`) start each series at its first value, which `increase()` doesn't count. Each has a `_created_seconds_total` counter with the same tags holding the Unix time the series was created, so the alerts and the Models Changed panel count a series created within their window in full.

## Built-in Metrics from vLLM and Ray

These are automatically available when `MSHIP_METRICS=true` — no additional configuration needed.

### vLLM (`ray_vllm_*`)

vLLM metrics are routed through Ray's metrics agent via `RayPrometheusStatLogger`. The native `vllm:` prefix is sanitized to `ray_vllm_`. Their `model_name` label is the modelship model name.

- `ray_vllm_num_requests_running` / `ray_vllm_num_requests_waiting` — queue depth
- `ray_vllm_kv_cache_usage_perc` — KV cache utilization (0-1)
- `ray_vllm_time_to_first_token_seconds` — TTFT histogram
- `ray_vllm_inter_token_latency_seconds` — ITL histogram
- `ray_vllm_e2e_request_latency_seconds` — end-to-end latency histogram
- `ray_vllm_request_queue_time_seconds` — time spent waiting in queue
- `ray_vllm_prompt_tokens_total` / `ray_vllm_generation_tokens_total` — token throughput counters
- `ray_vllm_num_preemptions_total` — memory pressure signal
- `ray_vllm_prefix_cache_hits_total` / `ray_vllm_prefix_cache_queries_total` — cache efficiency

Full reference: [vLLM Metrics Documentation](https://docs.vllm.ai/en/stable/design/metrics/)

### Ray Serve (`ray_serve_*`)

- `ray_serve_num_http_requests_total` — request count by route, method, status
- `ray_serve_http_request_latency_ms` — request latency histogram
- `ray_serve_handle_request_counter_total` — request count by deployment
- `ray_serve_deployment_processing_latency_ms` — per-replica processing time
- `ray_serve_health_check_latency_ms` — health check latency histogram
- `ray_serve_application_status` — status of each application: 0=UNKNOWN, 1=DEPLOY_FAILED, 2=UNHEALTHY, 3=NOT_STARTED, 4=DELETING, 5=DEPLOYING, 6=RUNNING. A model's application is named `<gateway>.<model>-<fingerprint>`.
- `ray_serve_replica_processing_queries` — requests each replica is processing
- `ray_serve_deployment_queued_queries` — requests queued at the caller (the gateway replica, for a model's application)
- `ray_serve_replica_startup_latency_ms` — time from a replica's creation to ready, including the wait for capacity
- `ray_serve_replica_initialization_latency_ms` — time a replica spent in its constructor (a model's whole load) and first health check

Full reference: [Ray Serve Monitoring](https://docs.ray.io/en/latest/serve/monitoring.html)

### Ray Cluster (`ray_*`)

- `ray_node_gpus_utilization` — GPU utilization by device
- `ray_node_gram_used` / `ray_node_gram_available` — GPU memory
- `ray_node_cpu_utilization` — CPU usage
- `ray_node_mem_used` / `ray_node_mem_total` — system memory

Full reference: [Ray Metrics](https://docs.ray.io/en/latest/cluster/metrics.html)
