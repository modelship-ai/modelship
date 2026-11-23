# Model Configuration

Reference for `models.yaml`, the file `--config` takes. Each entry under `models:` defines one deployment; an optional [`pipelines:`](#pipelines-experimental) list sits beside it, and any other top-level key is an error. Unknown keys are rejected: a typo like `n_ctxx` is a validation error, not a silently ignored setting — in the file and in the [CLI flags](#deploying-from-the-command-line-no-config-file) alike.

## CLI Options

For a one-page summary of every command, see the [CLI reference](cli.md).

`mship start` starts a cluster on this machine — its head node, the API gateway and any models
given — and stays running. `mship join` adds this machine to a running cluster as a worker node
and stays running. `mship deploy` changes the models of the cluster running on this machine, or with
`--ray-dashboard-url` of a cluster elsewhere: it sends a deploy request and exits; with `--wait` it waits for
the request to succeed or fail and exits with the outcome; `mship deploy --cancel ID` cancels a deploy request
instead, rolling back what it submitted, and with `--wait` waits for the rollback. `mship stop` stops the nodes
`start` and `join` run on this machine. Each takes the arguments
marked for it (env vars work as fallbacks; CLI wins over env):

| Argument | Commands | Env Var | Default | Description |
|---|---|---|---|---|
| `--config` | start, deploy | — | — | Path to a models.yaml; one that doesn't exist is a hard error. `start` makes its gateway serve exactly these models; `deploy` adds them. Without it (or `--model`), both redeploy the gateway's committed models that are missing |
| `--gateway-name` | start, deploy | `MSHIP_GATEWAY_NAME` | `modelship` | Name for the API gateway app. Multiple gateways can coexist on one cluster, each mounted at `/<slugified-name>` (e.g. `modelship` → `/modelship/v1/...`) |
| `--gateway-min-replicas` | start | `MSHIP_GATEWAY_MIN_REPLICAS` | `1` | Fewest replicas each API gateway autoscales down to; 2 or more gives routing/ingress HA |
| `--gateway-max-replicas` | start | `MSHIP_GATEWAY_MAX_REPLICAS` | `4` | Most replicas each API gateway autoscales up to |
| `--gateway-target-ongoing-requests` | start | `MSHIP_GATEWAY_TARGET_ONGOING_REQUESTS` | `64` | Ongoing requests per gateway replica that autoscaling aims for; a streamed response counts until it ends |
| `--gateway-max-ongoing-requests` | start | `MSHIP_GATEWAY_MAX_ONGOING_REQUESTS` | `1024` | Most requests one gateway replica handles at once; more wait in the proxy |
| `--openai-api-port` | start, deploy | `MSHIP_OPENAI_API_PORT` | `8000` | Port for the OpenAI-compatible API |
| `--gcs-address` | join | `MSHIP_GCS_ADDRESS` | — | The head's GCS address as `host[:port]` (e.g. `mship-head:6380`) — its `--gcs-port`, `6380` when omitted. Reachable only from inside the cluster's private network. See [Multi-node without Kubernetes](multi-node-docker.md) |
| `--enable-ray-auth` | start | `MSHIP_RAY_AUTH` | `false` | Require Ray's token auth for the dashboard and cluster-internal RPC. The token is `MSHIP_RAY_AUTH_TOKEN` when set; otherwise Ray generates it at `~/.ray/auth_token` |
| `--gcs-port` | start | `MSHIP_GCS_PORT` | `6380` | Ray GCS server port — what `mship join --gcs-address` points at |
| `--ray-dashboard-host` | start | `MSHIP_RAY_DASHBOARD_HOST` | `127.0.0.1` | Ray dashboard bind address. The dashboard's job API runs arbitrary code, so bind beyond loopback only on a private network and with `--enable-ray-auth`; `start` warns without it |
| `--ray-dashboard-port` | start | `MSHIP_RAY_DASHBOARD_PORT` | `8265` | Ray dashboard port. Only needed to run multiple modelship heads on one host under `--network=host` |
| `--node-num-cpus` | start, join | `MSHIP_NODE_NUM_CPUS` | auto-detect | CPUs this node reserves |
| `--node-num-gpus` | start, join | `MSHIP_NODE_NUM_GPUS` | auto-detect | GPUs this node reserves. Refused at startup if it exceeds what the container can actually see |
| `--node-labels` | start, join | `MSHIP_NODE_LABELS` | — | Ray labels for this node as `KEY=VALUE[,KEY=VALUE...]`, which a model's `node_selector` matches. The `modelship/` prefix is reserved |
| `--prune-ray-sessions` | start, join | `MSHIP_PRUNE_RAY_SESSIONS` | `true` | At node startup, delete stale `session_*` dirs left under the Ray temp root by previous, no-longer-running nodes. A live node's session is always kept |
| `--reconcile` | deploy | — | `false` | Make the cluster match the config: add new models, remove dropped ones, replace changed ones (vs. the default additive union). With no `--config`, redeploys this gateway's committed models that are missing (self-heal) |
| `--replace-strategy` | deploy | — | `blue_green` | How to replace a changed model: `blue_green` (deploy new before dropping old, no request loss) or `stop_start` (drop old first, brief unavailability) |
| `--wait` | deploy | — | `false` | Wait for the deploy request's outcome and exit with it. `deploy` waits for the request to succeed or fail (`0` succeeded, `1` failed or cancelled); without it, `deploy` exits once the request is queued, and the outcome shows in the head's log. With `--cancel`, it waits for the rollback (`0` once cancelled, `1` if the request ends otherwise). A signal only stops the wait; `mship deploy --cancel` cancels |
| `--cancel` | deploy | — | — | Cancel the deploy request with this id, as `mship deploy` printed it, instead of sending one; takes no model options. A queued request is dropped; a running one is rolled back, models already up included. A request that has committed, or that failed and is already rolling back, can't be cancelled |
| `--ray-dashboard-url` | deploy | `MSHIP_RAY_DASHBOARD_URL` | — | Send the deploy (or `--cancel`) to the cluster whose Ray dashboard answers at this URL, e.g. `http://head:8265`, instead of the one on this machine. It runs as a Ray job on that cluster's head, with the head's env. Needs `MSHIP_RAY_AUTH_TOKEN` when the cluster runs with `--enable-ray-auth`. See [Deploy from another machine](multi-node-docker.md#deploy-from-another-machine) |
| `--timeout` | stop | — | `90` | Seconds a node may take to shut down before `mship stop` kills it and every process it started |
| `--cache-dir` | start, join | `MSHIP_CACHE_DIR` | `/.cache` | Base cache directory for model weights; may be shared storage |
| `--node-cache-dir` | start, join | `MSHIP_NODE_CACHE_DIR` | `$MSHIP_HOME/node-cache` | Node-local compile/JIT cache directory (vLLM, Triton, FlashInfer). Must not be shared storage |
| `--state-store` | start | `MSHIP_STATE_STORE` | `memory://` | Connection URI for the gateways' deploy versions + `/v1/responses` state (see [State store](#state-store-mship_state_store)) |
| `--log-level` | start, join | `MSHIP_LOG_LEVEL` | `INFO` | Log level of this node's actors: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`; library levels follow it |
| `--log-format` | start | `MSHIP_LOG_FORMAT` | `text` | `text` or `json`, for the whole cluster |
| `--log-target` | start, join | `MSHIP_LOG_TARGET` | `console` | `console` or syslog URI (e.g. `syslog://host:514`, `syslog+tcp://host:514`) for this node's actors |
| `--otel-endpoint` | start, join | `OTEL_EXPORTER_OTLP_ENDPOINT` | — | OpenTelemetry OTLP endpoint (e.g. `http://collector:4317`) for this node's actors |
| `--no-metrics` | start | `MSHIP_METRICS` | enabled | Disable modelship's own Prometheus metrics on the whole cluster. Ray's node exporter has no off switch and falls back to a random port |
| `--metrics-port` | start, join | `MSHIP_METRICS_PORT` | `8079` on start, random on join | This node's Prometheus metrics port. A joiner's random port is listed in the head's service-discovery file; pin it where the scraper targets a fixed port, e.g. a Kubernetes PodMonitor |
| `--no-preflight` | start, deploy | `MSHIP_PREFLIGHT` | enabled | Disable preflight hardware auto-sizing and its check that a model fits; models run on loader/library defaults plus explicit config. Useful for benchmarking |
| `--trusted-identity-header` | start, deploy | `MSHIP_TRUSTED_IDENTITY_HEADER` | — | Header name (e.g. `X-Consumer-Id`) a fronting credentials layer sets with a caller identity it already resolved and authorized. See [Trusted identity header](#trusted-identity-header) |
| `--max-request-body-bytes` | start, deploy | `MSHIP_MAX_REQUEST_BODY_BYTES` | `52428800` | Max request body size in bytes |
| `--responses-ttl-s` | start, deploy | `MSHIP_RESPONSES_TTL_S` | `2592000` | TTL in seconds for stored `/v1/responses` conversation state, at least `86400` (24 hours); `<=0` disables expiry |
| `--state-sweep-interval-s` | start | `MSHIP_STATE_SWEEP_INTERVAL_S` | `300` | Interval in seconds between expired-key sweeps in the in-memory state store |

`mship deploy` waits up to 5 minutes for the head: a Ray node on this machine (or the dashboard at
`--ray-dashboard-url`) and the deploy coordinator `mship start` creates. It exits `0` on success, `1` when the
deploy fails or is cancelled or the cluster refuses the Ray auth token, `2` on a usage error, `3` when no Ray
dashboard answered at `--ray-dashboard-url` within the wait (or it stopped answering mid-deploy), `4` when the
cluster still had no deploy coordinator, and `130` when a signal stops the wait.

### Deploying from the command line (no config file)

`--model` deploys models straight from the command line, no `models.yaml` needed. Each `--model REF` opens a model, and the `--model.<key>` flags after it, up to the next `--model`, set keys of that model's entry:

```bash
mship start --model lmstudio-community/Qwen3-8B-GGUF:'*Q4_K_M.gguf' \
              --model.loader llama_server --model.usecase generate --model.num-cpus 4 \
            --model kokoro-multi-lang-v1_0 \
              --model.loader sherpa_onnx --model.usecase tts
```

Every key of a `models:` entry has a flag, at any depth, named for its config path, hyphenated: `--model.num-gpus` is `num_gpus`, `--model.llama-server-config.n-ctx` is `llama_server_config: {n_ctx: …}`, and `--model.vllm-engine-kwargs.reasoning-config.reasoning-end-str` is three levels down. The `model` key itself is the `--model REF` that opens the entry. Other flags (`--wait`, `--gateway-name`, …) can go anywhere on the line. The entries are validated by the same schema as the file, so anything rejected in the file is rejected here, with the same message, naming the flag. `mship deploy --help` lists every flag with its type and default.

Values are read as YAML: the same text you would write after the colon in the file.

| Flag | Value |
|---|---|
| `--model.llama-server-config.n-ctx 8192` | `8192` |
| `--model.vllm-engine-kwargs.trust-remote-code true` | `true` |
| `--model.vllm-engine-kwargs.limit-mm-per-prompt '{image: 2}'` | a map |
| `--model.llama-server-config.tensor-split '[3, 1]'` | a list |
| `--model.llama-server-config.threads null` | `null` — an explicit value, not "left unset" |
| `--model.node-selector '{modelship/variant: vulkan}'` | a map |

That includes YAML's own quirks: bare `on`, `off`, `yes` and `no` are booleans, so quote them — `--model.llama-server-config.chat-template '"on"'` — when the string is what you mean.

A block's own flag sets the whole block as a YAML mapping, and deeper flags then set keys inside it, whatever their order on the line:

```bash
mship deploy --model lmstudio-community/Qwen3-8B-GGUF:'*Q4_K_M.gguf' \
               --model.loader llama_server --model.usecase generate \
               --model.llama-server-config '{n_ctx: 8192, parallel: 4}' \
               --model.llama-server-config.n-ctx 16384
```

deploys `llama_server_config: {n_ctx: 16384, parallel: 4}`.

Limits:

- **`--model` and `--config` are mutually exclusive.** A `--model.<key>` flag before the first `--model` is an error, and so is the same flag twice for one model.
- **A flag can set a key, not unset one.** Omitting it leaves the schema default; pass `null` for the fields that accept it.
- **`mship start` deploys exactly the models given,** as with a file. A later `mship deploy --model …` adds to the running cluster (the default additive merge).

#### Inferred names

With no `--model.name`, the name clients call the model by comes from the `model` reference — its basename (or the filename stem for a local weight file), minus GGUF and quantization decoration. The selector is ignored: it picks a quant, it doesn't identify the model.

| `--model` | Inferred `name` |
|---|---|
| `lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf` | `qwen3-8b` |
| `bartowski/Llama-3.3-70B-Instruct-GGUF:*Q8_0*-of-*.gguf` | `llama-3.3-70b-instruct` |
| `Qwen/Qwen3-8B` | `qwen3-8b` |
| `/models/qwen3-8b-instruct.Q4_K_M.gguf` | `qwen3-8b-instruct` |
| `~/models/Qwen3-8B/` | `qwen3-8b` |
| `base.en` | `base.en` |

Inference is deterministic, so re-running the same command is idempotent. It also means two different references can infer the same name — deploying `Qwen/Qwen3-8B` on `vllm` and then a `Qwen3-8B` GGUF on `llama_server` both resolve to `qwen3-8b`, and the second **replaces** the first, since a name maps to exactly one deployment. That is what you want when swapping a model's loader; pass `--model.name` to run both side by side.

### Cache directory structure

Under `MSHIP_CACHE_DIR` (default `/.cache`; may be shared storage):

| Subdir | Contents | Env var |
|---|---|---|
| `huggingface` | HF models and tokenizers | `HF_HOME`, `HF_HUB_CACHE` (take precedence over a node's own `HF_HOME`, `HF_HUB_CACHE` and `HUGGINGFACE_HUB_CACHE`) |
| `sherpa_onnx/<name>-<sha256 prefix>` | sherpa-onnx registry tarballs | — |
| `vad/<file>-<sha256 prefix>` | the Silero voice activity model, from `model-bundles` | — |
| `.mship-cache-id` | Identifies this cache, so nodes sharing it take turns downloading | — |

Under `MSHIP_NODE_CACHE_DIR` (default `$MSHIP_HOME/node-cache` — `/opt/mship/node-cache` in the image). Never put this on storage shared between nodes: vLLM keys its GPU peer-access cache by device index alone, so nodes would read each other's results.

| Subdir | Contents | Env var |
|---|---|---|
| `vllm` | vLLM compiled artifacts | `VLLM_CACHE_ROOT` |
| `vllm-config` | vLLM usage-stats config | `VLLM_CONFIG_ROOT` |
| `triton` | Triton JIT kernels | `TRITON_CACHE_DIR` |
| `flashinfer` | FlashInfer JIT kernels | `FLASHINFER_WORKSPACE_BASE` |

### Additive vs. reconcile deploys

By default, deploys add models to a running cluster without touching existing deployments — run repeatedly against different config files to compose a cluster incrementally. `--reconcile` instead makes the cluster match the config exactly (add/remove/replace, draining in-flight requests on replace); it never tears the cluster down.

```bash
mship start --config config/llm.yaml             # start the cluster with the LLMs
mship deploy --config config/tts.yaml             # add, doesn't touch the LLMs
mship deploy --config config/all.yaml --reconcile      # make the cluster match exactly
```

`mship start` always makes its gateway match the models it is given, and removes any others its state store still holds. A model added later with `mship deploy` lasts until the head's next start; start the head with no `--config` to bring back what the store holds instead.

### What a deploy changes

A deploy request is compared, model by model, with the gateway's committed version and with what Ray Serve holds. Each model gets one action:

| Action | When |
|--------|------|
| `add` | The gateway doesn't have the model |
| `replace` | A field other than `num_replicas` or `autoscaling_config` changed, so the model gets a new deployment |
| `rescale` | Only `num_replicas` or `autoscaling_config` differ from what Serve holds; the running deployment is changed in place |
| `redeploy` | The config is unchanged, but Serve has no such deployment or a failed one |
| `remove` | `deploy --reconcile` or `start`, and the config leaves the model out |
| `keep` | Nothing differs |

The actions run in a fixed order, so that what frees room runs before what needs it:

1. Deployments left behind by an earlier failed deploy are deleted.
2. Rescales that raise no replica limit. Under `blue_green`, a replaced model whose replica count drops has its old deployment scaled down here.
3. `stop_start` only: the old deployments of replaced and removed models are deleted.
4. New deployments come up (`add`, `replace`, `redeploy`).
5. Rescales that raise a replica limit.
6. [Pipelines](#pipelines-experimental): their audio deployments come up or rescale, after every model.
7. The gateway switches to the new models and pipelines, and the version is committed.
8. `blue_green` only: the old deployments of replaced and removed models and pipelines are deleted.

The head's log lists the request's actions by step, and `--wait` prints them with the outcome:

```
Deploy 3f2a9c1d04be's diff:
  2. rescale whisper: num_replicas: 2 -> 1
  4. replace qwen: llama_server_config.n_ctx: 4096 -> 8192
  6. switch the gateway and commit
  keep: kokoro
  kokoro: unchanged
  qwen: up
  whisper: rescaled
Deploy 3f2a9c1d04be succeeded; the gateway is on version 8.
```

Replica counts are compared with Serve, not with the committed version, so sending the same config again puts back a count that no longer matches it. A deploy with no config does that for every committed model, and redeploys the ones Serve no longer has.

A request that fails or is cancelled after step 2 gets its scale-downs undone too, which loads the stopped replicas again.

## Pipelines (experimental)

> **Experimental.** Pipelines deploy, and a client can open a `/v1/realtime` session on one and update it. Nothing else in a session works yet: every other event is answered with an `unsupported_event` error. Audio, text turns and tools arrive in later releases.

A pipeline runs an OpenAI Realtime session over models the gateway already serves: speech to text, a chat model, and text to speech. Clients pick one with `/v1/realtime?model=<name>`. Pipelines sit beside `models:` in the same file:

```yaml
models:
  - name: "qwen"
    model: "lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf"
    usecase: "generate"
    loader: "llama_server"

pipelines:
  - name: "assistant"
    llm: "qwen"
    session_defaults:
      instructions: "You are a helpful home assistant."
```

| Field | Type | Description |
|---|---|---|
| `name` | string | Name clients pass as `/v1/realtime?model=<name>`. No `#` |
| `num_replicas` | int | Replicas of the pipeline's audio deployment. Default `1`. It reserves no CPU or GPU |
| `stt` | string | Name of the speech-to-text model it transcribes with |
| `llm` | string | Name of the chat model that answers |
| `tts` | string | Name of the text-to-speech model that speaks the answer |
| `session_defaults` | object | Defaults for OpenAI's Realtime session fields; a client's `session.update` overrides any of them |

A pipeline needs at least one of `stt` and `llm`. `session_defaults` holds `instructions` (the system prompt), `output_modalities` (`[text]`, `[audio]` or both) and `max_output_tokens` (an integer or `inf`); more of OpenAI's session fields arrive as the pipeline learns to use them.

From the command line, `--pipeline NAME` opens a pipeline the way `--model REF` opens a model, on `mship start` and `mship deploy` alike, and its keys are `--pipeline.<key>` flags right after it:

```bash
mship deploy --model lmstudio-community/Qwen3-8B-GGUF:'*Q4_K_M.gguf' \
               --model.name qwen --model.loader llama_server --model.usecase generate \
             --pipeline assistant --pipeline.llm qwen \
               --pipeline.session-defaults '{instructions: "You are a helpful home assistant."}'
```

A `--model.*` flag must follow its own `--model`, and a `--pipeline.*` flag its own `--pipeline`: one placed after the other kind is an error.

A pipeline deploys with the models, through the same request:

- A pipeline with `stt` gets an audio deployment, named `<gateway>.<pipeline>-p<fingerprint>`. It takes no CPU or GPU and runs on any node whose variant isn't `thin`. It comes up after every model of the request, and a change to `num_replicas` rescales it in place. A pipeline with only `llm` has no deployment.
- Every other change to a pipeline (its models, `session_defaults`) reaches the gateway with the new version, and nothing restarts.
- `mship deploy` adds or replaces pipelines by name and leaves the others. `deploy --reconcile` and `mship start` remove the pipelines the request doesn't name.
- A deploy never checks the models a pipeline names. A session on a pipeline whose model the gateway doesn't have, or doesn't serve yet, is refused with an error naming that model.
- A pipeline can't take a model's name: `/v1/models` lists both, and a pipeline's card has `owned_by: "modelship-pipeline"`. A pipeline is listed once its audio deployment is up.

## Trusted Identity Header

modelship never authenticates callers. There is no login, API keys, permissions, or per-model access control, and none is planned; that belongs to whatever sits in front (nginx, Kong, LiteLLM, a custom credentials layer). `MSHIP_TRUSTED_IDENTITY_HEADER` lets that layer forward a caller identity it already resolved (a consumer/tenant id), used for log correlation and for scoping server-side state (see [Stateful responses](#stateful-responses)) — never for authorization.

The header's value is trusted **unconditionally** — no signature check. Safe only if both hold:

1. **The fronting layer unconditionally overwrites the header**, stripping any client-supplied copy. A layer that only sets it when absent lets a client send `X-Consumer-Id: someone-elses-id` and impersonate them.
2. **modelship is reachable only from that fronting layer** (network policy, private subnet, or same pod) — never directly from any client, who could otherwise set the header themselves.

For stronger guarantees than network isolation, add mTLS on that internal hop (service-mesh sidecar, Kong, a local proxy) so the peer's certificate — not just placement — proves the request's origin. modelship does not implement or verify certificates itself.

If it's unset (the default), or a request doesn't carry the header, the caller lands in a single shared identity bucket.

## Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Model identifier used in API requests. Maps to exactly one deployment — a repeated name with different settings is a validation error |
| `model` | string | HuggingFace repo ID, local path, or `repo:filename` (see [Model source](#model-source)) |
| `usecase` | string | `generate`, `embed`, `transcription`, `translation`, `tts`, or `image`. Defaults to `image` and is otherwise rejected for `diffusers`/`stable_diffusion_cpp`; for `sherpa_onnx`, the one its registry name gives (`tts` or `transcription`) |
| `loader` | string | `vllm`, `diffusers`, `llama_server`, `stable_diffusion_cpp`, `whispercpp`, `sherpa_onnx` |
| `num_gpus` | float \| int | Default `0`: the model gets no GPU, and the node's GPUs are hidden from it. Fractional `< 1` shares one GPU — supported by `vllm`, `diffusers`, `llama_server`, `stable_diffusion_cpp`, and (Metal-only) `whispercpp`; forced to `0` for `sherpa_onnx`. `whispercpp` with `num_gpus > 0` schedules only onto a `metal` node. Integer `≥ 1` requests that many whole GPUs (for `vllm`, auto-sets `tensor_parallel_size = num_gpus` unless tp/pp is already specified; left unset with tp × pp above 1, it defaults to tp × pp). `llama_server`/`whispercpp` reject a non-integer `≥ 1`; `stable_diffusion_cpp` takes at most `1`. See [Sharing one GPU](#sharing-one-gpu) |
| `num_cpus` | float | CPU units to allocate. Default `0.1` |
| `num_replicas` | int | Fixed Ray Serve replica count. Default `1`. Mutually exclusive with `autoscaling_config`. Above `1`, a `vllm` or `llama_server` model must [set its context length](#scaling-a-deployment) |
| `autoscaling_config` | object | Autoscale replicas with load instead of a fixed `num_replicas` (see [Autoscaling](#autoscaling)). A `vllm` or `llama_server` model with it must [set its context length](#scaling-a-deployment) |
| `max_ongoing_requests` | int | Per-replica Ray Serve concurrency cap (default: Ray Serve's own default). Streaming requests hold a slot for the whole generation, so a low cap throttles upstream of the engine |
| `node_selector` | map | Ray node labels the model's nodes must match, as `{key: value}`; a value can also be `!value`, `in(a,b)` or `!in(a,b)`. Every node has `modelship/variant`, `modelship/os` and `modelship/hostname`, plus its `--node-labels`, and on Kubernetes `ray.io/node-group` (the worker group's name); `mship info --<variant>` lists each node's. A `modelship/variant` key replaces the variants modelship picks for the loader. Default `{}`. See [Multi-node clusters](#multi-node-clusters) |
| `vllm_engine_kwargs` | object | vLLM engine options (see [vLLM Loader](#vllm-loader)) |
| `diffusers_config` | object | Diffusers pipeline options (see [Diffusers Loader](#diffusers-loader)) |
| `llama_server_config` | object | llama-server loader options (see [llama_server Loader](#llama_server-loader)) |
| `stable_diffusion_cpp_config` | object | stable-diffusion.cpp loader options (see [stable-diffusion.cpp Loader](#stable-diffusioncpp-loader)) |
| `whispercpp_config` | object | whisper.cpp loader options (see [whispercpp Loader](#whispercpp-loader)) |
| `sherpa_onnx_config` | object | sherpa-onnx loader options (see [sherpa_onnx Loader](#sherpa_onnx-loader)) |

For `sherpa_onnx`, `provider` is always `cpu` and the thread count is derived from `num_cpus`; neither is user-supplied.

## Sharing one GPU

A fractional `num_gpus` (`0 < n < 1`) lets two or more deploys share one physical GPU via Ray's own fractional scheduling — e.g. a `vllm` model at `0.7` and a `llama_server` model at `0.3` on the same card. Keep fractions on one GPU summing to **at most 0.9**: the remaining headroom covers each engine's fixed pre-allocation overhead (CUDA context, etc.) outside the declared share.

Each GPU-capable loader sizes and enforces its share differently:

| Loader | Enforcement |
|---|---|
| `vllm` | Hard — allocates `total VRAM × gpu_memory_utilization` (a fractional `num_gpus` sets this directly) and fails at startup if it doesn't fit |
| `diffusers` | Hard — `torch.cuda.set_per_process_memory_fraction` caps the process; allocating past it OOMs |
| `llama_server` | Static — preflight sizes `n_ctx`/`n_gpu_layers` to fit the declared share; llama.cpp allocates upfront and never grows |
| `whispercpp` | None (Metal only) — `num_gpus > 0` is a plain on/off switch, no VRAM knob to size |
| `stable_diffusion_cpp` | Soft — preflight sets sd.cpp's `--max-vram` budget inside the share; the CUDA context and driver allocations sit outside it, so preflight leaves room for them |
| `sherpa_onnx` | N/A — never touches a GPU; `num_gpus` is ignored (forced to `0`) |

Preflight sizes a fractional deploy from its **declared share of the GPU's total capacity**, not free VRAM at that moment; if free VRAM falls short, it logs a warning naming both numbers.

On a `cuda` node whose GPU is integrated (an NVIDIA DGX Spark's GB10), the GPU's memory is also system RAM, and NVIDIA's monitoring library can't report it. Preflight asks CUDA whether the GPU is integrated, the same check llama.cpp and vLLM make, and sizes deploys from `/proc/meminfo`: the GPU's total is the node's RAM, and what's free is `MemAvailable` less a reserve of 5% of RAM, Ray's out-of-memory headroom (5% at Ray's default `RAY_memory_usage_threshold` of 0.95) and 1 GiB. A whole-GPU `vllm` deploy gets `gpu_memory_utilization` set to that free share, at most 0.9, since 0.9 of the total would be more than is free. Fractions are of the node's RAM: keep them summing to **at most 0.85** there, so the OS and Ray stay under Ray's threshold.

On a `vulkan` node whose GPU is integrated (an AMD APU such as Strix Halo), the GPU's memory is system RAM. Preflight sizes a `llama_server` deploy, whole or fractional, against what is really free: on AMD, the free part of the BIOS carve-out plus the free GTT, capped by `MemAvailable` less the same reserve as on a DGX Spark. It reads `/proc/meminfo`, not the container's memory limit, because GTT allocations don't count against `docker run --memory`. llama.cpp's own free figure ignores RAM that other processes hold, so without this a model loaded next to a `vllm` CPU deploy would be sized as if that RAM were still free.

Sharp edges:
- No SM/compute isolation between co-tenants — they time-slice like any two CUDA processes sharing a device (fine for one large + one small model; not a MIG/MPS substitute if both are large).
- On a multi-GPU node you can't pin which physical GPU a fractional deploy lands on — Ray picks.
- If a co-tenant *frees* VRAM while vLLM is mid-startup (profiling its own footprint), vLLM's internal consistency check can abort; Ray Serve's replica restart recovers automatically.

## Model source

The `model:` field accepts three forms. For built-in loaders, modelship **validates** the source on the driver before any Ray actor spins up — auth failures, missing repos, bad selectors, and (on `vllm`) GGUF files all surface at startup, not inside a stuck deployment. Weight **download** happens per-replica, on whichever node hosts it (see [Multi-node clusters](#multi-node-clusters)) — the driver only resolves metadata and pins the revision every node fetches.

| Form | Example | Notes |
|---|---|---|
| HF repo ID | `Qwen/Qwen3-7B` | Downloaded with a universal filter (prefers `*.safetensors`, skips `*.bin` when both exist) |
| Local path | `/mnt/nfs/models/qwen-7b` | A directory of HF-format files, or a single file for llama.cpp/vllm GGUF |
| `repo:filename` | `lmstudio-community/Qwen2.5-7B-Instruct-GGUF:*Q4_K_M.gguf` | Glob selector; must match exactly one file (or a sharded set, e.g. `*-of-*.gguf`) |

The `:filename` selector also works against a **local directory**: if `model:` is a directory path containing `:`, the selector matches files inside it; the loader gets the resolved file's full path.

### Multi-node clusters

Each node downloads its own copy of whatever gets scheduled onto it — **not** the driver's `HF_HOME` shared out to workers. A thin/control-only node that never hosts a replica downloads nothing; a node that does host one pulls exactly that model, pinned to the revision the driver validated (byte-identical weights across nodes even if the upstream repo changes between deploys).

- **Shared storage (NFS/EFS) for `MSHIP_CACHE_DIR` is optional**, not required — mount it to dedupe across nodes; without it, each node downloads its own, correctly.
- Cache paths resolve on each node from its own `MSHIP_CACHE_DIR` and `MSHIP_NODE_CACHE_DIR`, so they may differ between nodes. `mship` and the images set both; if you start a node's Ray yourself (`ray start`), export them there, or replicas on that node refuse to start.
- Replicas downloading the same model into the same cache take turns: one fetches while the rest wait (logging `waiting for download lease …`), then find it cached. Nodes with separate caches download in parallel. A replica killed mid-download has its partial files removed before the next attempt. The first download after a minute without any waits about 30s before it starts.
- Every node that can host a model needs its own disk and egress (HF rate limits apply per node).
- Every node that can host a gated model needs `HF_TOKEN` in its own environment; replicas read it from the node, not the driver.
- A local-path `model:` is resolved on whichever node hosts the replica — the path must exist on every node that could host it; there's no cross-node copying.

A model also only schedules onto a node whose variant runs its loader on the device it asks for (`cpu` has no `diffusers`, and `cuda` runs `vllm` only on a GPU, for instance): every node labels itself with its variant (`modelship/variant`), and every deploy selects the variants that run its loader on its device. `node_selector` narrows that further, e.g. `{modelship/variant: vulkan}` when both a cuda and a vulkan node could run the model, or a label you gave a node with `--node-labels`. See [Architecture: Capability-aware scheduling](architecture.md#capability-aware-scheduling).

See [Multi-node without Kubernetes](multi-node-docker.md) for the full non-k8s cluster setup (auth, ports, co-location).

### Multi-variant GGUF repos

If `model:` points at an HF repo with more than one `.gguf` file and no `:filename` selector, modelship raises at startup listing the variants:

```
HF repo 'lmstudio-community/Qwen2.5-7B-Instruct-GGUF' contains 5 GGUF variants — pick one with the `:filename` syntax (glob supported, must match exactly one file):
  - Qwen2.5-7B-Instruct-Q2_K.gguf
  - Qwen2.5-7B-Instruct-Q4_K_M.gguf
  - Qwen2.5-7B-Instruct-Q5_K_M.gguf
  - Qwen2.5-7B-Instruct-Q8_0.gguf
  - Qwen2.5-7B-Instruct-fp16.gguf
Example: model: lmstudio-community/Qwen2.5-7B-Instruct-GGUF:*Q4_K_M.gguf
```

## vLLM Loader

Chat/generation, embeddings, transcription, and translation. Configured via `vllm_engine_kwargs`.

Two vLLM settings are **not** keys here, because modelship derives them and setting them is a config error: `model` (the engine always loads the resolved top-level `model:`) and `gpu_memory_utilization` (a fractional `num_gpus` for a shared GPU, else a preflight recommendation, else `0.9` — `0.4` on CPU, where vLLM reads it as a *host RAM* fraction instead of VRAM).

| Field | Type | Default | Description |
|---|---|---|---|
| `tensor_parallel_size` | int | `1` | GPUs for tensor parallelism |
| `pipeline_parallel_size` | int | `1` | GPUs for pipeline parallelism |
| `max_model_len` | int | auto | Max sequence length; must be positive if set. Left unset on GPU, vLLM fits the largest context its own post-profiling memory allows (the min across workers, so TP/PP are covered). On `num_gpus: 0`, preflight sizes it from host RAM instead, falling back to that same auto-fit when it declines (missing `config.json`, unreadable KV-cache geometry, etc.). Setting it explicitly overrides both, and is required with `num_replicas` above `1` or `autoscaling_config` ([Scaling a Deployment](#scaling-a-deployment)) |
| `dtype` | string | `auto` | `auto`, `float16`, `bfloat16` |
| `tokenizer` | string | model default | Custom tokenizer path |
| `trust_remote_code` | bool | `false` | Allow remote code execution |
| `quantization` | string | — | e.g. `awq`, `gptq` |
| `enable_auto_tool_choice` | bool | — | Enable automatic tool/function calling |
| `tool_call_parser` | string | — | e.g. `llama3_json`, `hermes` |
| `enable_reasoning` | bool | — | `false` turns reasoning parsing off |
| `reasoning_parser` | string | — | e.g. `qwen3`, `deepseek_r1` |
| `default_chat_template_kwargs` | dict | — | Default variables for the chat template, e.g. `{"enable_thinking": false}` for Qwen3. Only the keys the template reads have an effect |
| `thinking_token_budget` | int | — | Most tokens a model may spend thinking; `0` skips thinking. See [Limiting thinking](#limiting-thinking) |
| `reasoning_config` | dict | — | `reasoning_start_str` and `reasoning_end_str` for the budget. See [Limiting thinking](#limiting-thinking) |
| `enforce_eager` | bool | — | Disable CUDA graph capture |
| `kv_cache_dtype` | string | — | e.g. `fp8` |
| `enable_prefix_caching` | bool | vLLM default: on | Disable to turn off vLLM's automatic prefix caching entirely. Not usually needed — every request is already cache-salted per caller identity (see [Identity-scoped prefix caching](architecture.md#identity-scoped-vllm-prefix-caching)); this is for an operator who wants caching off regardless of identity |
| `max_num_batched_tokens` | int | — | vLLM scheduler batch cap |
| `max_num_seqs` | int | — | Max concurrent sequences |
| `limit_mm_per_prompt` | dict | — | Per-modality multimodal item cap, e.g. `{"image": 4}` |
| `mm_processor_kwargs` | dict | — | Forwarded to the HF processor (e.g. `min_pixels`/`max_pixels` for Qwen2.5-VL) |

> **GGUF is not supported on the `vllm` loader** — vLLM 0.24 dropped in-tree GGUF, so a `.gguf` source is rejected at driver preflight, unconditionally regardless of GPU vs. CPU. Use `loader: llama_server` for GGUF; `vllm` takes safetensors, or AWQ/GPTQ/FP8 quants.

### Tool calling and reasoning

The tool parser is the vLLM parser that reads a tool call as the model's chat template writes it. The reasoning parser is the vLLM parser that reads the reasoning the chat template writes, alongside the tool parser. A template that writes no reasoning gets the parser named in the model's `config.json` when the template shows it thinks, or else `deepseek_r1`, vLLM's plain `<think>` parser, when the template uses `<think>`. The startup log names the parser picked. `tool_call_parser` and `reasoning_parser` name one instead; `enable_auto_tool_choice: false` and `enable_reasoning: false` switch one off.

On a model that reasons, `response_format` and a forced `tool_choice` (`required` or a named function) take effect once the reasoning ends: the answer arrives in `content` or `tool_calls` with the reasoning beside it, and `max_tokens` covers both.

On a template with an `enable_thinking` variable, a request's `reasoning_effort` (`reasoning.effort` on `/v1/responses`) sets it: `none` turns thinking off, any other value turns it on. That overrides `default_chat_template_kwargs`, and a request's own `chat_template_kwargs` overrides both.

### Limiting thinking

`thinking_token_budget` caps how long a model thinks: at the budget, vLLM writes `reasoning_end_str` and the model answers. It is the default and the ceiling for a request's own `thinking_token_budget` on `/v1/chat/completions`; `/v1/responses` has no such field.

`reasoning_end_str` is the parser's closing marker (`</think>` for `qwen3`) unless set. A sentence ahead of the marker gives cleaner answers:

```yaml
vllm_engine_kwargs:
  thinking_token_budget: 1024
  reasoning_config:
    reasoning_end_str: "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
```

`reasoning_start_str` is for a parser that declares no marker of its own. A model with no reasoning parser, or without both strings, has no budget, and the startup log says so.

### CPU (no GPU required)

Installable via the `vllm-cpu` extra (paired with `num_gpus: 0`) for quantized chat with no GPU — safetensors, or AWQ/GPTQ/compressed-tensors quants (CPU backend supports AWQ/GPTQ on x86 plus INT8 W8A8); GGUF is rejected here too.

`gpu_memory_utilization` means *host RAM* fraction on CPU, not VRAM — modelship's default drops to `0.4` (vLLM's own `0.9` would try to reserve 90% of node RAM and fail at worker init). Since it isn't a settable key, preflight's recommendation (from actual free RAM and the model's weight footprint) always applies when it can compute one; the `0.4` fallback only kicks in when preflight declines (e.g. unreadable `config.json`). Set `max_model_len` explicitly for finer control. vLLM also reads `VLLM_CPU_KVCACHE_SPACE` (fixed GiB budget) and `VLLM_CPU_OMP_THREADS_BIND` (thread pinning) directly from the environment — vLLM-native, not modelship config.

Minimum config — preflight fills in the rest:

```yaml
models:
  - name: qwen-cpu
    model: Qwen/Qwen3-8B-AWQ
    usecase: generate
    loader: vllm
    num_gpus: 0
```

See `config/examples/vllm-cpu.yaml` for a complete example with tool calling.

### Examples

```yaml
models:
  - name: qwen                              # chat
    model: Qwen/Qwen3-0.6B
    usecase: generate
    loader: vllm
    num_gpus: 0.30
    vllm_engine_kwargs:
      max_model_len: 8192

  - name: llama                             # tool calling
    model: meta-llama/Llama-3.1-8B-Instruct
    usecase: generate
    loader: vllm
    num_gpus: 0.70
    vllm_engine_kwargs:
      enable_auto_tool_choice: true
      tool_call_parser: llama3_json

  - name: nomic-embed                       # embeddings
    model: nomic-ai/nomic-embed-text-v1.5
    usecase: embed
    loader: vllm
    num_gpus: 0.15
    vllm_engine_kwargs:
      trust_remote_code: true

  - name: whisper                           # transcription
    model: openai/whisper-large-v3-turbo
    usecase: transcription
    loader: vllm
    num_gpus: 0.15
    vllm_engine_kwargs:
      trust_remote_code: true
```

### Multi-GPU with tensor parallelism

`num_gpus: 2` is shorthand for "2 whole GPUs" — `tensor_parallel_size` is auto-derived. Setting both is redundant (a warning is logged); setting only `tensor_parallel_size`/`pipeline_parallel_size` is fine too. Each slot always owns one whole GPU.

```yaml
models:
  - name: llama-70b
    model: meta-llama/Llama-3.1-70B-Instruct
    usecase: generate
    loader: vllm
    num_gpus: 2
```

Multi-slot deploys always use vLLM's ray distributed executor: each TP/PP slot is its own Ray worker actor inside a Ray Serve placement group (`STRICT_PACK`, one whole-GPU bundle per slot, all on one node for NVLink).

> Fractional `num_gpus` (`< 1`) is **single-GPU only**. Combining it with `tensor_parallel_size > 1` or `pipeline_parallel_size > 1` is rejected at config time — Ray packs fractional placement-group bundles onto the same physical GPU, which breaks tensor parallelism. Share a GPU with `num_gpus: 0.x` and `tp: 1`; use whole-GPU integer `num_gpus` for TP.

## Diffusers Loader

HuggingFace Diffusers for image generation; any `AutoPipelineForText2Image`-compatible model works out of the box. Configured via `diffusers_config`:

| Field | Type | Default | Description |
|---|---|---|---|
| `dtype` | string | `float16` | `float16`, `bfloat16`, `float32` |
| `num_inference_steps` | int | `30` | Default denoising steps (overridable per request) |
| `guidance_scale` | float | `7.5` | Default classifier-free guidance scale (overridable per request) |

```yaml
models:
  - name: sd-turbo
    model: stabilityai/sd-turbo
    usecase: image
    loader: diffusers
    num_gpus: 0.35
    diffusers_config:
      dtype: "float16"
      num_inference_steps: 4
      guidance_scale: 0.0
```

## llama_server Loader

Runs GGUF models by launching a [`llama-server`](https://github.com/ggml-org/llama.cpp) subprocess and proxying its native OpenAI-compatible HTTP API. Chat templating, tool-call parsing, and reasoning parsing are all llama-server's own (`--jinja --reasoning-format auto`), not modelship's. `--parallel` request slots let concurrent requests actually overlap instead of serializing behind a single lock. Requires the unified `llama` binary discoverable via `MSHIP_LLAMA_SERVER_BIN` (see [development.md](development.md#llama-server-binary-llama_server-loader)); Docker images ship a pinned build under `/opt/mship/builds/<variant>/llama.cpp/`, and `mship` points `MSHIP_LLAMA_SERVER_BIN` at it.

`num_gpus` accepts `0` (CPU-only), a fraction `< 1` (shares one physical GPU), or a whole integer — see [Sharing one GPU](#sharing-one-gpu). Configured via `llama_server_config`:

| Field | Type | Default | Description |
|---|---|---|---|
| `n_ctx` | int | auto (preflight); `2048` when preflight declines | Per-slot context length. The launch command multiplies this by `parallel` for llama-server's total `-c`. Unset, preflight gives each slot as much of the model's own context length as fits. Preflight shells out to `llama fit-params`, which builds the real KV cache and compute buffers to solve `n_ctx`/`n_gpu_layers`/`tensor_split` together. On `num_gpus: 0` it fits against the node's free RAM (a container memory limit counts), less a reserve of 5% of the node's RAM plus 1 GiB, so each model is sized against what the ones loaded before it left. A model that preflight cannot fit into that, at its smallest context or at a set `n_ctx`, is not loaded and its deploy fails ([Troubleshooting](troubleshooting.md#could-not-be-sized-for-this-node)). A value is required with `num_replicas` above `1` or `autoscaling_config` ([Scaling a Deployment](#scaling-a-deployment)) |
| `n_batch` | int | `512` | Batch size for prompt processing |
| `n_gpu_layers` | int | auto (preflight); `-1` when preflight declines or answers negatively | Layers offloaded to GPU when `num_gpus > 0`; forced to `0` when `num_gpus` is `0`. Preflight records a count only when `fit-params` returns a non-negative one, so a `-ngl -1` answer leaves this default in place while its `-c` answer still applies. `-1` hits llama-server's own auto-fit-to-free-memory path — verified for any negative value on b11699 (`--help`'s documented `'auto'`/`'all'` string tokens aren't reachable through this int field) |
| `threads` | int | `None` (llama-server default: all cores) | Compute thread count (`--threads`). Preflight recommends `num_cpus` when the deploy reserves ≥1 whole CPU and it wouldn't undercut `parallel` |
| `parallel` | int | `1` | Concurrent request slots (`--parallel`). Also becomes `max_ongoing_requests`'s default when that's unset, so overflow queues in Ray Serve rather than inside llama-server |
| `chat_template` | string | — | Built-in template name (e.g. `chatml`) or a Jinja file path. Omit to use the GGUF's embedded template |
| `mmproj` | string | — | Multimodal projector file/repo ref for vision models — see [Vision](#vision-gguf) |
| `cache_reuse` | int | `0` | Min chunk size (tokens) for fuzzy KV-cache reuse via position-shifting (`--cache-reuse`). `0` means exact-prefix reuse only; raise it to also reuse chunks after a mid-prompt divergence (changed system prompt, swapped RAG doc) |
| `context_shift` | bool | `false` | Evict oldest tokens and keep generating when a slot's context fills, instead of erroring (`--context-shift`) |
| `cache_ram_mib` | int | `0` | Cap in MiB on llama-server's host-RAM cache of saved conversations (`--cache-ram`). `0` disables it, `-1` removes the cap |
| `ubatch_size` | int | `512` | Physical max batch size (`-ub`) — the largest single memory lever after context itself |
| `flash_attn` | `on`/`off`/`auto` | `auto` | Flash Attention use (`-fa`) |
| `cache_type_k` / `cache_type_v` | string | `f16` | KV cache quantization (`-ctk`/`-ctv`); also `f32`, `bf16`, `f8_e4m3` (not on a `vulkan` node: llama.cpp aborts), `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1` |
| `tensor_split` | list[float] | `None` | Proportional offload split across GPUs (`-ts`). Preflight recommends an uneven split on heterogeneous cards |
| `override_tensor` | list[string] | `None` | Where to keep matching weights (`-ot`), as `pattern=buffer` entries such as `exps=CPU`; used only with a GPU. When a MoE model doesn't fit on its GPU, preflight moves some layers' experts to the CPU with it. It shrinks the context to 512 before it moves any, so set `n_ctx` to keep more context at the cost of more experts on the CPU |
| `reasoning` | `on`/`off`/`auto` | `None` (llama-server default: `auto`, from the chat template) | Whether the model thinks (`--reasoning`) |
| `reasoning_budget` | int | `None` (llama-server default: unlimited) | Most tokens a model may spend thinking (`--reasoning-budget`); `0` skips thinking. The default and the ceiling for a request's own `thinking_budget_tokens` or `reasoning_budget_tokens` |
| `reasoning_budget_message` | string | — | Text written before the end-of-thinking marker when the budget runs out (`--reasoning-budget-message`). A sentence there gives cleaner answers, as on [vLLM](#limiting-thinking) |
| `reasoning_preserve` | bool | `None` (the chat template's own default) | `true` keeps the reasoning of past turns in the prompt (`--reasoning-preserve`), `false` drops it (`--no-reasoning-preserve`), on templates that support it |
| `media_allow_private_urls` | bool | `false` | Let llama-server fetch `image_url`/`audio_url` media from localhost and private (LAN) addresses (`--media-allow-private-urls`). Off, such a URL is refused; `data:` URIs and public URLs work either way |

On a GPU deploy, whatever doesn't fit on the GPU (layers, or a MoE model's experts) stays in RAM. `llama fit-params` sizes only the GPU, so preflight then asks it what the fitted launch keeps in RAM and refuses the deploy when that is more than the node has free, after the same reserve a CPU deploy keeps. Weights count even though llama.cpp maps them from the file: a model that pages its weights from disk is too slow to use. On a GPU that uses system RAM (a DGX Spark's GB10, an AMD iGPU), the GPU's part, a vision model's `mmproj` included, and the RAM part come out of the same free memory and are checked together.

Each slot keeps the prompt it last processed (`--cache-prompt`, always on), so a follow-up turn, or another chat with the same system prompt, only processes the new tokens. llama-server can also save a conversation to host RAM when an unrelated one takes its slot and restore it when it returns; `cache_ram_mib` caps that cache and defaults to `0` (off), because it grows after load, outside the memory `n_ctx` was sized against. Saved conversations take about as much RAM per token as the context itself. There is no persistent **on-disk** prompt cache; caching is in-memory only and doesn't survive a process restart, unlike modelship's disk cache for other loaders.

#### MLA models (DeepSeek, MiniCPM3)

llama.cpp caches the compressed latent for MLA architectures only when the GGUF ships split `attn_k_b`/`attn_v_b` projections. Conversions predating that support carry a fused `attn_kv_b` and fall back to full per-head K/V — for DeepSeek-V2-Lite that is 276,480 B/token instead of 31,104, so the same VRAM buys ~8.9x less context. `llama fit-params` builds the real KV cache either way, so `n_ctx` is sized correctly for whichever layout the file has — but the only way to recover the missing context is to use a newer conversion. Nothing in the model's metadata reveals this, so check the tensor names if a GGUF sizes far smaller than expected.

Minimum config — preflight fills in `n_ctx`, `n_gpu_layers`, `threads`:

```yaml
models:
  - name: "qwen-llama-server"
    model: "lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf"
    usecase: "generate"
    loader: "llama_server"
    num_gpus: 1
```

### Tool calling and reasoning gaps vs. the OpenAI spec

Both parsers are auto-detected from the model's chat template — no modelship-level override. Gaps are per-model-family, not per-loader, so test against the specific model in use:

- **Named-function forcing is unsupported.** `tool_choice: {"type": "function", "function": {"name": "X"}}` silently falls back to `auto` (llama-server logs a warning; modelship doesn't surface it as an error).

`response_format`/`json_schema` can combine with reasoning in the same request; `logprobs`/`top_logprobs` are forwarded and returned.

### Vision (GGUF)

Set `mmproj` to a multimodal projector file (local path or `repo:filename`) to enable image input. Requests with `image_url`/`input_image` content parts are rejected at the gateway when `mmproj` isn't configured.

```yaml
models:
  - name: "qwen-vl-llama-server"
    model: "lmstudio-community/Qwen3-VL-8B-Instruct-GGUF:*Q4_K_M.gguf"
    usecase: "generate"
    loader: "llama_server"
    llama_server_config:
      mmproj: "lmstudio-community/Qwen3-VL-8B-Instruct-GGUF:mmproj-Qwen3-VL-8B-Instruct-F16.gguf"
```

### Embeddings (GGUF)

```yaml
models:
  - name: nomic-embed-server
    model: "nomic-ai/nomic-embed-text-v1.5-GGUF:nomic-embed-text-v1.5.Q4_K_M.gguf"
    usecase: embed
    loader: llama_server
```

## stable-diffusion.cpp Loader

Runs [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s `sd-server` as a subprocess, found through `MSHIP_SD_SERVER_BIN` (which `mship bootstrap` provisions, see [development.md](development.md#sd-server-binary-stable_diffusion_cpp-loader)), for GGUF-quantized diffusion checkpoints (SD1.5, SDXL, SD-Turbo, Flux). It runs on the GPU on `cuda`, `vulkan` and `metal` nodes, and on the CPU everywhere. `usecase` defaults to `image`; serves `/v1/images/generations`, `/v1/images/edits`, `/v1/images/variations`.

`num_gpus` is `0` (CPU), a fraction below `1` (shares one GPU), or `1`. sd-server queues requests and generates one at a time. A client that disconnects cancels its request, queued or running; a running one stops at the end of its current sampling step.

On a fractional `num_gpus`, preflight passes sd.cpp a GPU budget (`max_vram`): the share of the GPU's total, less what sd.cpp doesn't count (1 GiB on `cuda`, 0.25 GiB on `vulkan` and `metal`). When the model's files plus 0.5 GiB fit that budget, it sets `auto_fit: false` and the weights stay on the GPU, with sd.cpp splitting a large image's compute to stay inside the budget. When they don't, it leaves sd.cpp's auto-fit on and logs a warning. Auto-fit puts what doesn't fit on the GPU in RAM, up to the free RAM less 2 GiB or 10%, and reads the rest from disk as it runs. It reads the node's free RAM, not a container's memory limit. A whole GPU keeps sd.cpp's own fitting.

Configured via `stable_diffusion_cpp_config`:

| Field | Type | Default | Description |
|---|---|---|---|
| `sample_steps` | int | `20` | Denoising steps (`--steps`) |
| `cfg_scale` | float | `7.0` | Classifier-free guidance scale (`--cfg-scale`) |
| `guidance` | float | sd.cpp's (`3.5`) | Distilled guidance, for models with a guidance input such as Flux (`--guidance`) |
| `sample_method` | string | `default` | Sampler (`--sampling-method`); `default` lets sd.cpp pick per architecture |
| `scheduler` | string | `default` | Denoiser sigma scheduler (`--scheduler`) |
| `wtype` | string | `default` | On-the-fly weight quantization (`--type`, e.g. `q4_0`, `q8_0`, `f16`); `default` keeps the file's |
| `n_threads` | int | `-1` | CPU threads (`--threads`); `-1` lets sd.cpp pick |
| `vae_tiling` | bool | `false` | Tile the VAE decode to cut peak memory (`--vae-tiling`; recommended by preflight on a CPU deploy with little free RAM) |
| `flash_attn` | string | sd.cpp's (`auto`) | `on`, `off`, `auto`, or module assignments such as `auto,vae=off` (`--flash-attn`) |
| `clip_on_cpu` / `vae_on_cpu` | bool | `false` | Run the text encoder / VAE on the CPU on a GPU deploy (`--backend te=cpu,vae=cpu,<gpu>`) |
| `offload_to_cpu` | bool | `false` | Keep weights in RAM and move them to the GPU when needed (`--offload-to-cpu`) |
| `max_vram` | float | set by preflight | GPU budget in GiB (`--max-vram`) |
| `auto_fit` | bool | set by preflight | sd.cpp's placement of weights by free memory (`--auto-fit`) |
| `diffusion_model_path` / `clip_l_path` / `clip_g_path` / `t5xxl_path` / `vae_path` | string | — | Standalone component paths for split checkpoints (pre-placed local paths) |

`MSHIP_LOG_LEVEL=TRACE` makes sd-server verbose. GGUF variants in an HF repo use the same `:filename` syntax as the llama_server loader (see [Model source](#model-source)). The first launch of a `metal` deploy on a machine compiles Metal's shaders, about 20 s.

```yaml
models:
  - name: sd21-turbo
    model: "gpustack/stable-diffusion-v2-1-turbo-GGUF:*Q4_1.gguf"
    usecase: image
    loader: stable_diffusion_cpp
    num_gpus: 0.25
    stable_diffusion_cpp_config:
      sample_steps: 4
      cfg_scale: 1.0
```

## whispercpp Loader

Runs [whisper.cpp](https://github.com/ggml-org/whisper.cpp) speech-to-text in-process via [`pywhispercpp`](https://github.com/absadiki/pywhispercpp) bindings — no subprocess, unlike `llama_server`. CPU-only on Linux, so a `num_gpus > 0` deploy schedules only onto a `metal` node. There, `num_gpus > 0` (including a fraction, to share the GPU) enables Metal offload — no separate VRAM knob, just a plain on/off switch, so the model's own footprint determines whether it fits its share.

Configured via `whispercpp_config`:

| Field | Type | Default | Description |
|---|---|---|---|
| `n_threads` | int | pywhispercpp's own default (`min(4, cores)`) | Compute thread count |
| `flash_attn` | bool | `false` | ggml flash attention |

`model:` accepts:
- A whisper.cpp model name (e.g. `base.en`, `large-v3-turbo`), shorthand for `ggerganov/whisper.cpp:ggml-<name>.bin` — pinned, validated at deploy time, and cached under `huggingface/` like any HF ref. The [repo's file list](https://huggingface.co/ggerganov/whisper.cpp/tree/main) shows every name.
- A local path or HF `repo:filename` pointing at a `ggml-*.bin` file, resolved like every other loader. Both GGUF and safetensors are rejected with a pointer at the right form.

```yaml
models:
  - name: "whisper-cpp-base"
    model: "base.en"
    usecase: "transcription"
    loader: "whispercpp"
    num_cpus: 2
```

See `config/examples/whispercpp.yaml` for every `model:` form and a Metal example.

## sherpa_onnx Loader

Runs [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) in-process via its Python bindings: text to speech with Kokoro, and offline speech to text with Parakeet. CPU only — never touches CUDA or CoreML, so `num_gpus` is ignored entirely (any value accepted at config time, forced to `0` at deploy with a warning if nonzero) and reserves no GPU capacity another deploy could use.

`model:` is not an HF repo or path — it's a name from a curated built-in registry (or a local directory whose basename matches one). Each name maps to a tarball with a pinned sha256 in modelship's own [`model-bundles`](https://github.com/modelship-ai/model-bundles) releases (unmodified copies of the sherpa-onnx project's files), downloaded and cached under `<cache_root>/sherpa_onnx/<name>-<sha256 prefix>/` on first use.

| Name | `usecase` | Notes |
|---|---|---|
| `kokoro-en-v0_19` | `tts` | 11 speakers. English only. Includes `af_bella` |
| `kokoro-multi-lang-v1_0` | `tts` | 54 speakers. English, Mandarin, Japanese, and more. Includes `af_bella` and `af_heart` |
| `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` | `transcription` | NVIDIA's Parakeet TDT 0.6b v3 (CC-BY-4.0): 25 European languages, punctuated, with word timestamps. About 30× faster than real time on 2 threads |

`usecase` must be the one the name's row gives.

```yaml
models:
  - name: "kokoro"
    model: "kokoro-en-v0_19"
    usecase: "tts"
    loader: "sherpa_onnx"
    num_cpus: 2
```

| `sherpa_onnx_config` key | Default | Description |
|---|---|---|
| `default_voice` | the bundle's first voice | `tts` models only. The voice for a `/v1/audio/speech` request that names no voice, or one the bundle doesn't have (for example OpenAI's `alloy`). Must be one of the bundle's voices. A voice the bundle has, or a numeric speaker id within range, is always used as given |

```yaml
models:
  - name: "kokoro"
    model: "kokoro-multi-lang-v1_0"
    usecase: "tts"
    loader: "sherpa_onnx"
    sherpa_onnx_config:
      default_voice: "af_heart"
```

A speech-to-text model serves `/v1/audio/transcriptions` in every `response_format` and with `stream`:

- It finds the language itself and takes none: a request's `language` and `prompt` are accepted and ignored. `verbose_json` reports `language` as the request's, or `unknown`.
- `verbose_json` has a segment per sentence, and word timestamps when `timestamp_granularities` includes `word`.
- Audio longer than 30 s is cut at its pauses into pieces of at most 30 s, found with [Silero VAD](https://github.com/snakers4/silero-vad), which the model downloads beside its bundle. With `stream`, each piece is a `transcript.text.delta`.

```yaml
models:
  - name: "parakeet"
    model: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    usecase: "transcription"
    loader: "sherpa_onnx"
    num_cpus: 2
```

See `config/examples/sherpa-onnx.yaml` for the local-directory form.

## Scaling a Deployment

Use `num_replicas` to run several identical copies. The gateway sends a conversation's turns to the same replica while it isn't much busier than the others ([Conversation-affinity routing](architecture.md#conversation-affinity-routing)).

```yaml
models:
  - name: "kokoro"
    model: "kokoro-en-v0_19"
    usecase: "tts"
    loader: "sherpa_onnx"
    num_cpus: 2
    num_replicas: 2
```

A `vllm` or `llama_server` model with `num_replicas` above `1` or an `autoscaling_config` must set its context length: `vllm_engine_kwargs.max_model_len`, or `llama_server_config.n_ctx`. The config is rejected without it. Left unset, each replica would fit its own context to the node it lands on, so the same request could succeed on one replica and overflow the context on another. To pick a value, deploy the model with one replica and read the context from its `deployed` log line.

```yaml
models:
  - name: "qwen"
    model: "lmstudio-community/Qwen3-0.6B-GGUF:*Q4_K_M.gguf"
    usecase: "generate"
    loader: "llama_server"
    num_cpus: 2
    num_replicas: 2
    llama_server_config:
      n_ctx: 8192
```

Changing `num_replicas` on a deployed model changes the running deployment: its replicas keep serving while Ray Serve starts or stops the others, and `--replace-strategy` plays no part. A scale-down runs before the request's new deployments come up and a scale-up after them ([What a deploy changes](#what-a-deploy-changes)). The deploy waits for the new count, so a scale-up the cluster has no room for holds the gateway's deploy queue like any pending deploy; `mship deploy --cancel` puts the committed count back. Any other change to the model replaces its deployment. Setting the context length on a model deployed without one is such a change, so set it from the first deploy when you plan to scale.

## Autoscaling

Set `autoscaling_config` instead of a fixed `num_replicas` to let Ray Serve grow/shrink replica count with load. The two are mutually exclusive — setting both is a config error.

```yaml
models:
  - name: "bursty-llm"
    model: "Qwen/Qwen3-0.6B"
    usecase: "generate"
    loader: "vllm"
    num_gpus: 0.3
    vllm_engine_kwargs:
      max_model_len: 8192        # required with autoscaling_config
    autoscaling_config:
      min_replicas: 1            # floor; 0 enables scale-to-zero (cold-start on first request)
      max_replicas: 4            # ceiling
      target_ongoing_requests: 8 # autoscaler setpoint: in-flight requests per replica (lower = scales out sooner)
      initial_replicas: 1        # seed count on first deploy, before load signal (default: min_replicas)
      upscale_delay_s: 10        # debounce before scaling out
      downscale_delay_s: 300     # debounce before scaling in (longer avoids thrashing GPU warm-up)
```

| Field | Type | Description |
|---|---|---|
| `min_replicas` | int | Lower bound. Default `1`. `0` enables scale-to-zero |
| `max_replicas` | int | Upper bound. Default `1`. Must be `≥ min_replicas` |
| `initial_replicas` | int | Seed count before the autoscaler has a load signal. Default: `min_replicas` |
| `target_ongoing_requests` | float | Desired in-flight requests per replica — the autoscaler's setpoint. Lower scales out sooner. Default: Ray Serve's own |
| `upscale_delay_s` | float | Seconds of sustained over-load before adding replicas. Default: Ray Serve's own |
| `downscale_delay_s` | float | Seconds of sustained under-load before removing replicas. Default: Ray Serve's own. Raise it to avoid thrashing on models with slow GPU warm-up |

Changing `autoscaling_config` on a deployed model is applied to the running deployment like a `num_replicas` change ([Scaling a Deployment](#scaling-a-deployment)): its replicas keep serving, and the current count is kept within the new bounds.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace access token. Read from each node's own environment, never forwarded by the driver | — |
| `MSHIP_CACHE_DIR` | Model cache directory (HuggingFace, sherpa_onnx, etc.); may be shared storage | `/.cache` |
| `MSHIP_NODE_CACHE_DIR` | Node-local compile/JIT cache directory (vLLM, Triton, FlashInfer); must not be shared storage | `$MSHIP_HOME/node-cache` |
| `MSHIP_STATE_STORE` | State-store connection URI for the gateways' deploy versions + `/v1/responses` conversations (see [State store](#state-store-mship_state_store)) | `memory://` |
| `MSHIP_DEPLOY_SWITCH_TIMEOUT_S` | Seconds a deploy waits, once its new models are running, for every gateway replica to switch to them before it rolls back. Read by the deploy coordinator, set on the process that creates it (`start`) | `60` |
| `MSHIP_DEPLOY_CANCEL_GRACE_S` | Seconds a cancelled deploy's worker gets to start its rollback before it is killed and a new worker rolls back instead. Read by the deploy coordinator, set on the process that creates it (`start`) | `30` |
| `MSHIP_REDIS_PASSWORD` | Password for a `redis://`/`rediss://` state store. Read from each node's own environment, never forwarded by the driver | — |
| `MSHIP_GATEWAY_NAME` | Name for the API gateway app | `modelship` |
| `MSHIP_GATEWAY_MIN_REPLICAS` / `MSHIP_GATEWAY_MAX_REPLICAS` | Range each API gateway autoscales in, set on `start`; a gateway `deploy` creates uses the head's | `1` / `4` |
| `MSHIP_GATEWAY_TARGET_ONGOING_REQUESTS` | Ongoing requests per gateway replica that autoscaling aims for | `64` |
| `MSHIP_GATEWAY_MAX_ONGOING_REQUESTS` | Most requests one gateway replica handles at once | `1024` |
| `MSHIP_OPENAI_API_PORT` | Port for the OpenAI-compatible API | `8000` |
| `MSHIP_MAX_REQUEST_BODY_BYTES` | Maximum allowed request body size in bytes | `52428800` (50 MB) |
| `MSHIP_LOG_TARGET` | Log target: `console` or syslog URI | `console` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry OTLP endpoint for log export. Requires `uv sync --extra otel` | — |
| `CUDA_DEVICE_ORDER` | GPU enumeration order; set to `PCI_BUS_ID` for deterministic ordering in multi-GPU systems | `PCI_BUS_ID` |
| `MSHIP_RAY_DASHBOARD_HOST` | Ray dashboard bind host (`--ray-dashboard-host`), `start` only. Dashboard always starts; this sets *where* it binds — `0.0.0.0` exposes it beyond the container (the ShadowRay/CVE-2023-48022 exposure vector), so only do this on a trusted network | `127.0.0.1` |
| `MSHIP_RAY_AUTH` | Ray cluster authentication (`--enable-ray-auth`), `start` only. `true` makes `start` require a bearer token for the dashboard and all cluster-internal RPC: `MSHIP_RAY_AUTH_TOKEN`, or one Ray generates at `~/.ray/auth_token`. The OpenAI API and Prometheus metrics are never gated either way | `false` |
| `MSHIP_GCS_PORT` | Ray GCS server port, `start` only. Pinned so `mship join --gcs-address` has a stable target; not `6379` since that collides with the recommended same-host Redis state store under `--network=host` | `6380` |
| `MSHIP_RAY_DASHBOARD_PORT` | Ray dashboard port, `start` only. Needed only when running multiple modelship heads on one host under `--network=host` | `8265` |
| `MSHIP_GCS_ADDRESS` | The head's GCS address as `host[:port]` (port `6380` by default) that `mship join` joins (`--gcs-address`) | — |
| `MSHIP_RAY_AUTH_TOKEN` | The Ray auth token, never a flag. `join`/`deploy` send it to a cluster started with `--enable-ray-auth`; `start --enable-ray-auth` uses it instead of generating one, and refuses it without the flag | — |
| `MSHIP_NODE_NUM_CPUS` | Override: CPUs this node reserves | auto-detect |
| `MSHIP_NODE_NUM_GPUS` | Override: GPUs this node reserves. Refused at startup if it exceeds what the container can actually see | auto-detect |
| `MSHIP_NODE_LABELS` | Ray labels for this node as `KEY=VALUE[,KEY=VALUE...]` (`--node-labels`), matched by a model's `node_selector`. The `modelship/` prefix is reserved | — |
| `RAY_OBJECT_STORE_SHM_SIZE` | Shared memory for Ray object store | `8g` |
| `VLLM_USE_V1` | Use vLLM v1 API | `1` |
| `ONNX_PROVIDER` | ONNX Runtime execution provider | `CUDAExecutionProvider` |
| `NVIDIA_CUDA_VERSION` | CUDA toolkit version | `12.8.1` |

### State store (`MSHIP_STATE_STORE`)

One pluggable store holds each gateway's **deploy versions** (the model set its last successful deploy committed, and the one before it; a deploy with no models redeploys what's missing from it to self-heal after cluster loss) and **`/v1/responses` conversations** (see [Stateful responses](#stateful-responses)). A single connection URI picks the backend and carries its connection:

| URI | Backend | Durability |
|---|---|---|
| `memory://` (default) | dict shared cluster-wide by a detached Ray actor on the head node | survives a deploy re-run, a deploy or gateway coordinator restart, a gateway-replica restart — **not** cluster death |
| `redis://host:6379/0` (`rediss://` = TLS) | one JSON value per key in Redis | survives head, deploy or gateway coordinator death **and** cluster loss; password comes from `MSHIP_REDIS_PASSWORD`, not the URI |

Redis keys live under `modelship/state/`. Add `?namespace=<name>` (letters, digits, `.`, `_`, `-`) to put them under `modelship/state/<name>/`, so clusters sharing one Redis db keep separate state. The Helm chart sets it per release.

`memory://` is cluster-scoped, not process-local — every gateway replica and model actor shares one detached Ray actor on the head node, so it's correct at any replica count and outlives any worker node. Sized for small-traffic single-node deployments: every operation is a Ray RPC through that one actor, and large values spill to the object store.

`mship start` sets the store for the whole cluster; `mship deploy` takes none and uses the head's.

A password belongs in `MSHIP_REDIS_PASSWORD` on every node, not in the URI: the head forwards the URI to gateway and model replicas in `runtime_env`, which is plain-text cluster metadata, so the URI travels password-free and each node adds its own before connecting (percent-encoded, so `/`, `#`, `%` and friends in a password are safe). A password inside `--state-store`/`MSHIP_STATE_STORE` is rejected at startup.

The Helm chart always sets `redis://…` in Kubernetes (with the password from its Secret as `MSHIP_REDIS_PASSWORD` on every pod); the same Redis also backs Ray GCS fault tolerance (chart's **Head-node HA** section) and keeps each gateway's committed version across a head restart, so routing carries on without a redeploy.

> A `file://` backend existed before v0.7.0 and was removed: a poor fit for per-turn conversation snapshots (one JSON file each, no native TTL, last-writer-wins across replicas). Migrate `--state-store file://…`/`MSHIP_STATE_DIR` to `redis://`, or drop to `memory://` if you don't need to survive cluster loss.

### Stateful responses

`/v1/responses` keeps conversations server-side, so a follow-up turn sends only the new input instead of replaying the whole history:

```bash
curl localhost:8000/modelship/v1/responses -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "input": "my name is Alex"}'
# response id comes back in `id`

curl localhost:8000/modelship/v1/responses -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "input": "what is my name?", "previous_response_id": "resp_…"}'
```

| Route | Purpose |
|---|---|
| `POST /v1/responses` | `store` (default **true**) persists the response; `previous_response_id` continues from one; `background: true` queues instead of blocking |
| `GET /v1/responses/{id}` | Fetch a stored response — for a background response, poll until `status` is terminal |
| `DELETE /v1/responses/{id}` | Drop a stored response; on an in-flight background response this implies cancel |
| `POST /v1/responses/{id}/cancel` | Cancel an in-flight background response |
| `GET /v1/responses/{id}/input_items` | The input a stored response was produced from |

Send `"store": false` to opt out of storage — no id to continue from. An unknown, expired, or already-deleted `previous_response_id` is `404`; an unreachable state store is `503` (never a silent stateless fallback).

`"background": true` returns `status: "queued"` immediately instead of blocking; poll `GET` until `status` reaches a terminal value (`completed`/`incomplete`/`failed`/`cancelled`). Requires `store`. Combined with `"stream": true`, the initial call instead streams live, and a disconnected client can resume with `GET /v1/responses/{id}?stream=true&starting_after=<last sequence_number seen>` — the replay buffer is short-lived (`MSHIP_RESPONSES_STREAM_BUFFER_TTL_S`, default 600s). Use `redis://` for production background use, since a background response must outlive the request that created it.

Conversations are scoped to the caller's identity, so one caller can never read or continue another's. **Without a trusted identity header, every caller shares the single `unscoped` identity** — one conversation pool. Set `MSHIP_TRUSTED_IDENTITY_HEADER` (see [Trusted identity header](#trusted-identity-header)) before serving more than one user.

| Variable | Description | Default |
|---|---|---|
| `MSHIP_RESPONSES_TTL_S` | How long a stored response can be fetched or continued from. A conversation's history lives until this long after its last turn. Values under `86400` (24 hours) are raised to it; `0` disables expiry | `2592000` (30 days) |
| `MSHIP_STATE_SWEEP_INTERVAL_S` | How often the `memory://` store reclaims expired keys. `0` disables sweeping | `300` |

Sizing: each turn stores only its own input and output, so an *n*-turn conversation costs O(n) storage. A client that sends its whole history in `input` on every call, with no `previous_response_id`, stores it again each time; send `"store": false` if it doesn't need the stored copy. On the default `memory://` that all sits in one Ray actor's RAM for up to the TTL, so for sustained multi-user traffic lower `MSHIP_RESPONSES_TTL_S` or move to `redis://`.
