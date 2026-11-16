# Troubleshooting

Common issues hit during first-run and deployment.

## `HF_TOKEN` not set / 401 on gated models

Some HuggingFace models (Llama 3, Gemma, Mistral variants) require accepting a license and authenticating. Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), accept the model license on its HF page, then pass the token in:

```bash
docker run ... -e HF_TOKEN=hf_xxx ghcr.io/modelship-ai/modelship:latest-cpu start
```

Ungated models (e.g. `lmstudio-community/Qwen2.5-7B-Instruct-GGUF`) don't need a token.

## Permission denied on `/.cache`

The container starts as root and drops to the owner of the directory mounted at `/.cache`. A root-owned one — including a missing `-v` source Docker created for you — is chowned to UID/GID 1000 on first start instead. Set `MSHIP_UID`/`MSHIP_GID` to pick the user explicitly (`0` runs as root).

Under `--user` or a Kubernetes `runAsUser` the container can't change ownership, so the mount must already be writable by that UID.

If you previously used the old `/root/.cache/huggingface` mount path, switch to `/.cache` and move any cached weights across — the container no longer looks at the old location.

## `shm-size` too small / deployment is slow

Ray's object store (up to 256 MiB per node) and vLLM/NCCL need shared memory. Pass `--shm-size=8g` (or larger for big models) — Docker's 64MB default falls well short. If it's too small, Ray doesn't error, it silently falls back to slower disk-backed Plasma storage.

## arm64 vs amd64 image selection

`ghcr.io/modelship-ai/modelship:latest` (thin), `:latest-cpu`, `:latest-cuda` and `:latest-vulkan` are multi-arch (amd64 + arm64). Docker picks the right one automatically for your host. If you need to force an arch (e.g. cross-building), use `--platform linux/arm64` or `linux/amd64`.

`:latest-cuda`'s arm64 image uses NVIDIA's `sbsa` CUDA packages: it is for arm64 servers with a CUDA 13 driver, such as DGX Spark and GH200. Jetson Orin's JetPack 6 ships CUDA 12, below the `cuda` variant's floor. Apple Silicon should always use `:latest-cpu` (no CUDA path applies).

## Port 8000 already in use

Another service is bound to `8000`. Either free it up or remap:

```bash
docker run ... -p 8001:8000 ...   # exposed on host:8001
```

## Model download is slow / stalls

Weights are cached to `/.cache/huggingface` inside the container. Mount a persistent host directory (`-v ./models-cache:/.cache`) so subsequent runs reuse them. For large models, set a longer `docker run` timeout or pre-pull with `huggingface-cli download`.

## `Could not find nvcc` on a native CUDA install

flashinfer JIT-compiles its kernels when vLLM loads a model, so `mship bootstrap --cuda` requires the CUDA toolkit and `ninja` beyond the driver and refuses to run without them; install them per [Native install](install-native.md#platform-prerequisites). `nvcc` need not be on `PATH` — the apt packages create the `/usr/local/cuda` symlink that torch resolves against, and the check looks there too. `mship info` reports the same check on an already-bootstrapped host.

## `CUDA out of memory` with vLLM

vLLM reserves VRAM based on `num_gpus` — a whole number of GPUs, or a fraction of one when sharing a card — and fits the context to what's left. If a single model uses more than its budget, lower `num_gpus` for other deployments, or set `vllm_engine_kwargs.max_model_len` to cap KV cache size; an explicit value replaces the auto-fit.

## Workers killed for low memory on a DGX Spark

Ray kills workers once the node's memory use passes `RAY_memory_usage_threshold` (95% by default), and on a DGX Spark the GPU's memory is the node's memory. Preflight keeps Ray's headroom free when it sizes a deploy, but explicit settings bypass it: fractional `num_gpus` that sum past about 0.85, or a large `n_ctx` or `max_model_len`. Lower them until the models fit.

Two things on a Spark do not limit model memory:

- `docker run --memory` caps the container's memory, but CUDA allocations are charged to no container, so a model can use more than the limit. llama.cpp's managed memory (`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`) is the exception: it counts against the limit.
- In an LXC container whose `/proc/meminfo` comes from LXCFS, `MemAvailable` leaves out GPU allocations, so llama.cpp, vLLM and preflight all count memory a loaded model already holds as free. vLLM then fails with `Error in memory profiling`. Run on the host or in Docker instead.

## `could not be sized for this node`

A `llama_server` model on `num_gpus: 0` is sized against the RAM its node has free. When preflight cannot fit it into that RAM, at its smallest context or at the `n_ctx` you set, the model is not loaded and its deploy fails. The message gives the RAM that was available and `fit-params`' own reason; `failed to fit CLI arguments to free memory` means the model needs more than that. Free memory on the node, use a smaller model or quantization, or set a smaller `llama_server_config.n_ctx` on the models already loaded there: a model without one takes its full context when it loads first. `--no-preflight` loads the model without the check. A replica that fails this on a restart or a scale-up is retried until it fits, and the model answers 503 while none of its replicas is up.

## Deploy stuck pending, never schedules

Every deploy requests an `mship_<loader>_<gpu|cpu>` Ray resource; nodes only advertise the loaders their variant runs, on the devices it runs them on. A deploy no node can run (e.g. `loader: vllm` with `num_gpus: 1` on `-cpu` nodes only, or with `num_gpus: 0` on `-cuda` nodes only) pends instead of failing. Check `ray status`/dashboard for the missing `mship_*` resource; `MSHIP_NODE_CAPABILITIES` (JSON) replaces a node's set. A pending deploy has no timeout and holds its gateway's deploy queue, so later deploys to that gateway wait behind it; `mship deploy --cancel ID` cancels it and rolls back what it submitted.

## Can't reach the server from another host

The API binds to `0.0.0.0:8000` by default, but if you're on a remote machine, make sure the port is reachable through your firewall and you're using the host's IP, not `localhost`.

## Getting more diagnostic detail

- Set `MSHIP_LOG_LEVEL=DEBUG` for verbose logs.
- Set `MSHIP_LOG_LEVEL=TRACE` to log full request/response payloads (and enable llama.cpp `verbose` mode).
- The Ray dashboard is **always on**, publish port `8265` to reach it. It binds to `127.0.0.1` inside the container by default — pass `--ray-dashboard-host 0.0.0.0` (or a specific interface) to `mship start` to expose it beyond the container. This is the exposure vector behind ShadowRay/CVE-2023-48022, so only do this on a trusted/private network, with `--enable-ray-auth`; `start` warns without it. Prometheus metrics on `8079` are exported regardless.
- Ray cluster authentication is **off by default**. Pass `--enable-ray-auth` to `mship start` to require a bearer token for the dashboard and cluster-internal RPC — the dashboard UI will then ask for one on first load; retrieve it with `docker exec <container> cat /home/modelship/.ray/auth_token` and paste it in once. The OpenAI API on `8000` and Prometheus metrics on `8079` are never gated by this either way.
