# Architecture

## Overview

Modelship is a **FastAPI gateway** exposing an OpenAI-compatible API, built on [Ray Serve](https://docs.ray.io/en/latest/serve/) for deployment orchestration. Inference backends:

- **[vLLM](https://github.com/vllm-project/vllm)** — continuous batching + PagedAttention, GPU or CPU
- **llama-server** — proxied `llama-server` subprocess for quantized GGUF chat/embeddings/vision, CPU or GPU
- **[HuggingFace Diffusers](https://github.com/huggingface/diffusers)** — image generation via `AutoPipelineForText2Image`
- **stable-diffusion.cpp** — image generation for GGUF SD models, CPU everywhere plus Metal on Apple Silicon
- **sherpa-onnx** — TTS (Kokoro), CPU/CoreML
- **whisper.cpp** — STT via `pywhispercpp`, CPU everywhere plus Metal on Apple Silicon

## Request Lifecycle

1. Client sends a request to the FastAPI gateway (e.g. `POST /v1/chat/completions`)
2. The gateway identifies the target model from the request body and looks up the deployment serving it — `404` for a model the gateway doesn't have, `503` for a configured model with nothing serving yet
3. A `RequestWatcher` begins monitoring the client connection for disconnects
4. The request is forwarded to the model's Ray Serve deployment via a `RawRequestProxy` (serializable headers + cancellation event)
5. The deployment runs inference and streams the response back as JSON or SSE
6. On client disconnect mid-inference, the watcher fires the cancellation event, freeing GPU resources immediately

## Model Deployments

Each model in `models.yaml` becomes an isolated Ray Serve deployment (`ModelDeployment` actor):

- **Independent lifecycle** — one model crashing doesn't affect others
- **Per-model GPU budgeting** — `num_gpus` controls VRAM allocation (e.g. `0.7` for 70%)
- **One load per node** — a replica loading its model holds its node's lease (`DeployLeases`, pinned to the head node), so loads on the same node run one at a time and their memory spikes don't overlap; different nodes load in parallel. Downloading happens before the lease is taken
- **Additive by default** — `mship deploy` adds models to a running cluster without disrupting existing deployments. `--reconcile` instead makes the cluster match the config exactly (add/remove/replace); it never tears the cluster down
- **One deployment per model name** — a model name maps to exactly one deployment; scale it with `num_replicas` (or `autoscaling_config`), which Ray Serve load-balances across replicas natively. Changing a model's config replaces its deployment (`--replace-strategy`, default `blue_green`) rather than adding a second one alongside it
- **Routing derived from Serve** — a head-node actor (`ReplicaCoordinator`) computes each gateway's model → deployment table once a second from Ray Serve's application statuses and the gateway's effective config. A changed model keeps being served by its old deployment until the new one has a running replica; a deployment nothing uses is deleted 10 s later, and `mship deploy` waits for that. Nothing is stored, so a restarted coordinator recomputes the tables
- **One download per source** — a replica fetching weights holds a cluster-wide lease (`DownloadLeases`, pinned to the head node) on that HF repo or archive in its cache; replicas needing the same source wait their turn, and an already-cached source skips the lease. A lease that stops being renewed (its replica died mid-download) has that download's unfinished files removed on its node before anyone else gets the source
- **Multi-gateway support** — independent gateways can share a cluster via `--gateway-name`, each managing its own models and reachable under its own route (`/<slugified-gateway-name>/v1/...`), since every gateway shares the cluster's one HTTP proxy/port

### Inference Loaders

| Loader | Backend | Use cases | GPU required |
|--------|---------|-----------|--------------|
| `vllm` | vLLM engine | Chat/generation, embeddings, transcription, translation | No — GPU or CPU |
| `llama_server` | llama-server subprocess | Chat/generation, embeddings, vision (GGUF) | No — CPU or GPU (`n_gpu_layers` offload, fractional or whole) |
| `diffusers` | HuggingFace Diffusers | Image generation | Yes |
| `stable_diffusion_cpp` | stable-diffusion.cpp | Image generation (GGUF SD1.5/SDXL/SD-Turbo, Flux) | No — CPU everywhere, Metal on Apple Silicon |
| `whispercpp` | `pywhispercpp` | STT | No — CPU everywhere, Metal on Apple Silicon |
| `sherpa_onnx` | sherpa-onnx | TTS (Kokoro) | No — never touches CUDA |

Every deployment also requests a `mship_<loader>` Ray custom resource (see [Capability-aware scheduling](#capability-aware-scheduling)) alongside `num_gpus`/`GPU`.

### Identity-scoped vLLM prefix caching

vLLM's automatic prefix caching shares KV-cache blocks across every request on an engine by default — two callers sending the same prefix (e.g. a shared system prompt) get observably different time-to-first-token depending on cache state, leaking one caller's recent activity to another via timing. Every vLLM chat/Responses request is cache-salted with the caller's `identity_key()` (`modelship/openai/auth.py`) before it reaches the engine, so cache reuse is confined to one identity; a different identity always misses. This only isolates as well as identity resolution does — with no `MSHIP_API_KEYS`/`MSHIP_TRUSTED_IDENTITY_HEADER` configured, every caller shares one identity bucket. Salting trades cache hit rate, not memory, for isolation. Set `enable_prefix_caching: false` in `vllm_engine_kwargs` to disable prefix caching entirely.

## Responses API (`/v1/responses`)

Shaped natively per loader, not via a chat-completions round trip. `BaseInfer.create_response(request, raw_request)` is a hookable method (default: unsupported); `VllmInfer` and `LlamaServerInfer` implement it directly from their parsed `(reasoning, content, tool_calls)` output — the same `ParsedChatOutput` seam `/v1/chat/completions` uses.

- **Non-streaming** — `utils.responses.build_responses_items_from_parsed` maps the parsed tuple into `output[]` items; `protocol/responses/adapter.build_response_object` builds the envelope.
- **Streaming** — `ResponsesStreamTranslator` consumes loader-native typed chunks and emits the Responses event protocol (`response.created` → `output_item.added` → `output_text.delta`/`reasoning_summary_text.delta`/`function_call_arguments.delta` → `output_item.done` → `response.completed`). Output items open lazily on first delta and close at stream end.

**Supported** (on `vllm` and `llama_server` only): text, reasoning (`reasoning` output item), client-driven tool calling, server-side MCP tool execution, conversation state (`store`/`previous_response_id`, `GET`/`DELETE /v1/responses/{id}`, `/input_items`), `background` mode. **404s** on `diffusers` — no generic fallback. **Rejected with 400**: hosted built-in tools (`web_search`), OpenAI-hosted MCP connectors, `tool_choice` forcing a specific `mcp` tool, `background: true` + `store: false`. Encrypted reasoning (`reasoning.encrypted_content`) isn't implemented — server-side state covers the same need.

### Server-side MCP (`tools: [{"type": "mcp", ...}]`)

Discovered and executed entirely at the gateway (`modelship/openai/mcp/`), never touching a loader — intercepted before the Ray hop (`mcp.loop.wants_mcp`) and driven through `mcp.loop.run_mcp_response`. Works uniformly across loaders with zero loader changes, and composes with background mode and the WebSocket transport.

Loop: list the server's tools over streamable HTTP (official `mcp` SDK), expose them to the model as plain `function` tools, run turns, execute resulting `tools/call` calls itself, append results and loop — up to 10 turns or `max_tool_calls` — then return a completed response. A `require_approval` tool instead emits `mcp_approval_request` and ends the turn; the client resumes with `previous_response_id` + `mcp_approval_response`.

One logical response can span N loader turns, so a gateway-side stitcher (`mcp.loop.Stitcher`) owns the outer envelope: renumbers `sequence_number`/`output_index` per turn, accumulates output, rewrites matched tool-call events into `mcp_call` events (unmatched calls pass through as plain `function_call`). The client always sees one `response.created` and one terminal event.

Egress is permissive by default (plain HTTP, localhost, private IPs allowed) except cloud metadata endpoints, blocked unconditionally. Lock down with `MSHIP_MCP_ALLOWED_HOSTS` and `MSHIP_MCP_REQUIRE_HTTPS=true`.

### Background mode (`background: true`)

Returns a `status: "queued"` `ResponseObject` immediately; generation continues on a task on the dispatching gateway replica (not the model deployment). Poll `GET /v1/responses/{id}` until terminal (`completed`/`incomplete`/`failed`/`cancelled`), or `POST /v1/responses/{id}/cancel` (reuses the `DisconnectRegistry` actor). `DELETE` on an in-flight run implies cancel. Requires `store: true` (default) — `background: true` + `store: false` is a 400.

A detached run has no client connection to detect a dead worker, so the drain task heartbeats a separate `HeartbeatRegistry` actor every few seconds; a poller finding no live heartbeat for a non-terminal snapshot reports it `failed`. `background: true` + `stream: true` tees events into a short-lived per-response replay log (`MSHIP_RESPONSES_STREAM_BUFFER_TTL_S`, default 600s) as well as the durable snapshot; a disconnected client resumes via `GET .../{id}?stream=true&starting_after=<sequence_number>`. `redis://` (`MSHIP_STATE_STORE`) is the supported backend for production background use — `memory://` doesn't survive a restart.

### Conversation state

State lives in the gateway, not the loaders — `GET`/`DELETE` carry no model and can't be routed to a deployment. `resolve_history` prepends `previous_response_id`'s conversation into `input` before the Ray hop (a store outage is a clean 503 before any GPU work); `persist_response` tees output into the store on the way back. The loader only ever sees a flat `input`.

`modelship/openai/state/responses.py` stores one self-contained snapshot per response id, keyed `responses/<identity>/<response_id>` via the generic `modelship.state` store — continuing is a single read, and each turn's fresh id gives branching for free. The streaming write is the one asymmetry: the terminal `response.completed` event is re-parsed out of the SSE stream to recover the response object, persisted before forwarding so a store failure can downgrade the terminal event to `response.failed`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/modelship/v1", api_key="…")

resp = client.responses.create(model="reasoning-qwen", input="Which is larger, 9.11 or 9.9?")
print(resp.output_text)

with client.responses.stream(model="reasoning-qwen", input="Explain why, briefly.") as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
```

## GPU Allocation

Ray schedules deployments across available GPUs based on the `num_gpus` fraction each model requests — e.g. two models each requesting `num_gpus: 0.9` land on separate GPUs. Multi-slot vLLM deploys (`tp*pp > 1`) build a Ray Serve placement group instead (one whole-GPU bundle per slot, STRICT_PACK); fractional `num_gpus` combined with `tp*pp > 1` is rejected at config time.

### Capability-aware scheduling

`num_gpus`/`num_cpus` describe *how much* hardware a deployment needs, not *whether the node can run the loader at all*. Every node advertises a `mship_<loader>` Ray custom resource (e.g. `mship_vllm`) for each loader whose backend is installed — probed via `importlib.util.find_spec()`, plus a real-binary check for `llama_server` (`modelship/deploy/capabilities.py`). Every deployment requests its loader's resource alongside `num_gpus`/`GPU`, so it only schedules onto a node with that backend present; an unsatisfiable deploy pends, and Ray Serve's own slow-start warning names the missing resource.

This is what lets a `thin` (no-torch) coordinator deploy models onto `cuda`/`cpu` worker nodes, and what stops a `diffusers` model from landing on a `cpu`-image node without `diffusers` installed. `MSHIP_NODE_CAPABILITIES` (JSON) overrides a node's advertised set wholesale.

## Key Files

| File | Purpose |
|------|---------|
| `modelship/launcher.py` | Entry point behind `mship start` / `join` / `deploy` (`python -m modelship.launcher <command>`) — resolves cache root, checks Python version, detects accelerator, hands off to `driver.py` |
| `modelship/driver.py` | `start` / `join` / `deploy`: bring up or attach to Ray, then the deploy loop — additive by default, `--reconcile` to converge exactly |
| `modelship/openai/api.py` | FastAPI gateway with OpenAI endpoints |
| `modelship/openai/protocol/responses/` | `/v1/responses` schemas, chat adapter (`adapter.py`), streaming translator (`streaming.py`) |
| `modelship/state/` | Generic pluggable KV store (`memory://` via a detached Ray actor, `redis://`). Domain layers: `openai/state/responses.py`, `deploy/effective_config.py` |
| `modelship/infer/model_deployment.py` | Ray Serve deployment actor |
| `modelship/infer/infer_config.py` | Pydantic config models and protocols |
| `modelship/infer/downloads.py`, `download_leases.py` | Replica-side weight download under the cluster-wide lease; the lease actor and leftover cleanup |
| `modelship/infer/vllm/vllm_infer.py` | vLLM engine wrapper |
| `modelship/infer/llama_server/llama_server_infer.py` | llama-server subprocess proxy (GGUF chat/embed/vision) |
| `modelship/infer/diffusers/diffusers_infer.py` | Diffusers pipeline wrapper |
| `modelship/infer/stable_diffusion_cpp/stable_diffusion_cpp_infer.py` | stable-diffusion.cpp wrapper (CPU/Metal image gen) |
| `modelship/infer/whispercpp/` | whisper.cpp STT wrapper |
| `modelship/infer/sherpa_onnx/` | sherpa-onnx TTS wrapper |
| `config/models.yaml` | Model configuration |
