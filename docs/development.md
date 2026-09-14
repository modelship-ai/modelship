# Development

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (optional — only required for GPU development)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

## Quick start (Dev Container)

The recommended way to develop Modelship is with VS Code Dev Containers. The configuration in `.devcontainer/` builds the dev image, mounts the repo, forwards ports, and installs all required extensions automatically.

1. Set required environment variables on your host:

   ```bash
   export HF_TOKEN=your_token_here
   ```

2. Open the repo in VS Code and run **Dev Containers: Reopen in Container** from the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).

3. Once inside the container, sync dependencies:

   ```bash
   uv sync --extra dev
   ```

4. Start the server. It starts its own Ray head, auto-detecting CPUs/GPUs unless `MSHIP_NODE_NUM_CPUS` / `MSHIP_NODE_NUM_GPUS` are set:

   ```bash
   uv run mship_deploy.py
   ```

> **Why not `mship deploy`?** That console script ships only in the separate `bootstrap/` package (the `mship` installer), never synced into this repo's venv. `mship_deploy.py` and `python -m modelship.launcher deploy` are equivalent here.
>
> **Why the extra steps?** The `dev` target is built for source work: it syncs dependencies from `uv.lock` into `/.venv` and never installs the project, so you sync and start it yourself. The published images instead run `mship bootstrap` at build time and start at `mship deploy` — see [Docker install](install-docker.md).

The Dev Container automatically:
- Builds the dev image from `Dockerfile` (target: `dev`)
- Bind-mounts the repo to `/modelship` for live editing
- Forwards ports `8000` (API), `8265` (Ray Dashboard), and `8079` (Metrics)
- Installs extensions: Ruff, Python, Pyright, and Claude Code
- Configures the Python interpreter and linting to use the container's venv at `/.venv`

### Environment variables

The following environment variables are set in the dev image with sensible defaults. Override them in your host shell or in `.devcontainer/devcontainer.json` under `remoteEnv` as needed:

| Variable | Default | Description |
|---|---|---|
| `MSHIP_NODE_NUM_CPUS` | *(unset)* | **Optional override:** CPUs this node reserves (`--node-num-cpus` flag). Node-scoped, not head-only — the same knob a future worker node sizes itself from. If unset, auto-detects. |
| `MSHIP_NODE_NUM_GPUS` | *(unset)* | **Optional override:** GPUs this node reserves (`--node-num-gpus` flag). If unset, auto-detects. |
| `MSHIP_NODE_MEMORY` | *(unset)* | **Optional override:** this node's total memory budget, e.g. `8Gi` (`--node-memory` flag). Split into Ray's `object_store_memory` and schedulable `memory` resource via Ray's own object-store proportion. If unset, auto-detects from actually-free host RAM (not Ray's own uncapped-cgroup estimate, which can miss non-Docker consumers) — still double/triple-counts when co-locating multiple modelship containers on one host without per-container cgroup memory limits, see the sharp-edge note in `AGENTS.md`. |
| `MSHIP_RAY_DASHBOARD` | `127.0.0.1` | Ray dashboard bind host, own-head only. The dashboard always starts; this sets *where* it binds — `0.0.0.0` exposes it beyond the container (ShadowRay/CVE-2023-48022 exposure vector; only do this on a trusted/private network). |
| `MSHIP_RAY_AUTH` | `none` | Ray cluster authentication, own-head only (`--ray-auth` flag). `token` requires a bearer token for the dashboard and cluster-internal RPC. Never gates the OpenAI API or Prometheus metrics. |
| `MSHIP_RAY_PORT` | `6380` | Ray GCS server port, own-head only (`--ray-port` flag). Pinned by default (not `6379`, which collides with the recommended same-host Redis state store under `--network=host`) so a joiner's `--address` has a stable target across head restarts. |
| `MSHIP_RAY_DASHBOARD_PORT` | `8265` | **Optional override:** Ray dashboard port, own-head only (`--dashboard-port` flag). Only needed to run multiple modelship heads on one host under `--network=host`, where Ray's fixed default would otherwise collide between them. |
| `MSHIP_ADDRESS` | *(unset)* | **Optional:** join an existing Ray cluster as an additional compute node, given the head's GCS address as `host:port` (`--address` flag). See [Multi-node without Kubernetes](multi-node-docker.md). Mutually exclusive with `MSHIP_USE_EXISTING_RAY_CLUSTER`. |
| `MSHIP_RAY_AUTH_TOKEN` | *(unset)* | **Optional:** cluster auth token for joining a head running `--ray-auth=token` (`--token` flag). Only meaningful with `MSHIP_ADDRESS`. |
| `MSHIP_CACHE_DIR` | `/.cache` | Model cache directory (`--cache-dir` flag). May be shared storage. |
| `MSHIP_NODE_CACHE_DIR` | `/opt/mship/node-cache` | Node-local vLLM/Triton/FlashInfer compile caches (`--node-cache-dir` flag). Must not be shared storage. |
| `MSHIP_STATE_STORE` | `memory://` | State-store URI for the effective config, deploy coordinator + `/v1/responses` conversations: `memory://` or `redis://[:pw@]host:port/db`. See [model-configuration.md](model-configuration.md#state-store-mship_state_store). The chart always sets `redis://` for k8s. |
| `MSHIP_USE_EXISTING_RAY_CLUSTER` | `false` | Set to `true` to connect to a Ray cluster you manage (must run on a cluster node) instead of starting one; implies deploy-and-exit |
| `MSHIP_GATEWAY_REPLICAS` | `1` | Number of API gateway replicas. Raise for routing/ingress HA and to spread request-proxying load under high concurrency; replicas keep routing tables in sync via the deploy coordinator's watch loop. |
| `MSHIP_GATEWAY_MAX_ONGOING` | `1024` | Per-replica Ray Serve concurrency cap for the gateway. The gateway holds a slot for the whole lifetime of each streamed response, so a low cap throttles before the engine does. |
| `MSHIP_LLAMA_SERVER_BIN` | `/opt/mship/bin/llama-server.sh` in the Dev Container (unset otherwise) | Unified `llama` executable used by the `llama_server` loader and its preflight; see [llama-server binary](#llama-server-binary-llama_server-loader). |

## Manual setup (without Dev Containers)

If you prefer not to use Dev Containers, you can build and run the dev image directly.

### Building the dev image

**CUDA (GPU):**
```bash
docker build -t modelship_dev_cuda --target dev .
```

**CPU:**
```bash
docker build -t modelship_dev_cpu --target dev --build-arg MSHIP_VARIANT=cpu .
```

**Thin (control/coordinator, no torch/vllm):**
```bash
docker build -t modelship_dev_thin --target dev --build-arg MSHIP_VARIANT=thin .
```

### Running with live source mounting

The dev image does not bake in source files. Mount the repo root so changes take effect without rebuilding:

**CUDA:**
```bash
docker run -it --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token_here \
  --mount type=bind,src=./,dst=/modelship \
  -v ./models-cache:/.cache \
  -p 8000:8000 modelship_dev_cuda
```

**CPU:**
```bash
docker run -it --rm --shm-size=8g \
  -e HF_TOKEN=your_token_here \
  --mount type=bind,src=./,dst=/modelship \
  -v ./models-cache:/.cache \
  -p 8000:8000 modelship_dev_cpu
```

The dev image drops into a shell. Start the server — it starts its own Ray head, auto-detecting resources:

```bash
uv run mship_deploy.py
```

## llama-server binary (`llama_server` loader)

The `llama_server` loader and its preflight both invoke the unified `llama` binary (`llama serve` / `llama fit-params`) found via `MSHIP_LLAMA_SERVER_BIN` — the env var name predates the binary's unification and stuck. `mship bootstrap` provisions it (downloaded, sha256-verified, cached under `$MSHIP_HOME/builds/<variant>/llama.cpp/`) and `mship deploy` points the env var at its wrapper script — see `bootstrap/mship_bootstrap/llama_cpp.py` and its `_ASSETS` table. These are modelship's own builds of one llama.cpp tag for every platform it ships ([`modelship-ai/llama-cpp-builds`](https://github.com/modelship-ai/llama-cpp-builds)), so every node in a cluster runs the identical binary. The `cuda` variant adds `libggml-cuda.so` beside it. The `thin` variant ships no binary at all — it never loads a model.

The published images run that same bootstrap at build time, so they arrive preconfigured. The `dev` target does **not** run it, but the Dev Container fetches it for you: `.devcontainer/provision-llama-server.sh` runs from `postCreateCommand`, importing `bootstrap/mship_bootstrap/llama_cpp.py` straight from source (it's stdlib-only) to reuse its fetch/verify logic against the `cuda` variant, and `MSHIP_LLAMA_SERVER_BIN` is preset (via `remoteEnv`) to the stable symlink it leaves at `$MSHIP_HOME/bin/llama-server.sh`. Building and running the dev image manually still skips this — set `MSHIP_LLAMA_SERVER_BIN` yourself, run that script, or run `mship bootstrap` inside it, to exercise the `llama_server` loader there.

For a platform outside that table, or to use a different llama.cpp build than the pinned one, download a build from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and extract it. The raw `llama` binary does **not** run standalone — it dynamically links the `libggml*`/`libllama*` libraries that ship as sibling files in the same archive, so point `MSHIP_LLAMA_SERVER_BIN` at a small wrapper that puts the extracted directory on the loader path:

```bash
#!/bin/sh
# llama-server.sh — use DYLD_LIBRARY_PATH instead on macOS
export LD_LIBRARY_PATH="/path/to/extracted${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /path/to/extracted/llama "$@"
```

```bash
export MSHIP_LLAMA_SERVER_BIN=/path/to/llama-server.sh
```

## Production Builds

Three images are published — a thin control/coordinator image (no torch/vllm) plus two accelerator
node images:

| Variant | Tag suffix | Platforms |
|---|---|---|
| Thin (control/coordinator) | *(none)* — default | amd64, arm64 |
| CUDA (GPU node) | `-cuda` | amd64 |
| CPU (CPU node) | `-cpu` | amd64, arm64 |

All three share the `prod` target; `MSHIP_VARIANT` is what differs.

Floating tags (`:latest`, `:latest-cuda`, `:latest-cpu`) are single-node only — Ray refuses to form
a cluster across mismatched versions, so any multi-node deployment must pin every node to the same
`:X.Y.Z` (or `-cuda`/`-cpu`) tag.

The production images install the release wheels rather than the source tree, so build those first —
`release.yml` does the same, from a `wheels` job the image builds depend on:

```bash
uv build -o wheels .
uv build -o wheels bootstrap
```

Then, per variant (`thin` shown; swap in `cpu` or `cuda`):

```bash
docker build -t modelship:dev-thin \
  --target prod \
  --build-arg MSHIP_VARIANT=thin \
  --build-arg MSHIP_VERSION="$(uv version --short)" \
  --build-context wheels=./wheels \
  .
```

`MSHIP_VERSION` must match the wheels in `./wheels` — the build resolves
`mship==<version>` against that directory, so a stale `wheels/` from an earlier version fails to
resolve rather than silently installing the wrong one. Rebuild the wheels after a version bump.

## Ports

| Port | Service |
|---|---|
| `8000` | API |
| `8265` | Ray dashboard |
| `8079` | Metrics (`MSHIP_METRICS=false` to disable) |
