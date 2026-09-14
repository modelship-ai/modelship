# Docker install

The images are built by running the [native install](install-native.md) against the
release wheel — `mship bootstrap` provisions the environment at image-build time, so
a container starts straight into `mship deploy` and installs nothing at runtime.

## Image variants

Three images are published from a single `Dockerfile`, all under
`ghcr.io/modelship-ai/modelship`:

| Variant | Tag suffix | Platforms | Purpose |
|---|---|---|---|
| Thin | *(none)* — bare tag | amd64, arm64 | Control/coordinator image, no torch/vllm. Runs the driver/head role only — cannot serve models by itself |
| CUDA | `-cuda` | amd64 | GPU node image (vLLM, Diffusers, llama.cpp GPU offload) |
| CPU | `-cpu` | amd64, arm64 | CPU node image (vLLM CPU backend, llama.cpp, stable_diffusion_cpp) |

Floating tags (`:latest`, `:latest-cuda`, `:latest-cpu`) are single-node
only — multi-node deployments must pin every node to the same `X.Y.Z`
(`-cuda`/`-cpu`) tag, or Ray refuses to form the cluster across mismatched
versions.

## Running a single node

The image takes the same subcommands as the `mship` CLI, so the first argument is
`deploy`:

CPU, no GPU required:

```bash
docker run --rm --shm-size=8g \
  -v ./models.yaml:/modelship/config/models.yaml \
  -v modelship-cache:/.cache \
  -p 8000:8000 \
  ghcr.io/modelship-ai/modelship:latest-cpu deploy
```

GPU, with the NVIDIA Container Toolkit installed:

```bash
docker run --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token_here \
  -v ./models.yaml:/modelship/config/models.yaml \
  -v modelship-cache:/.cache \
  -p 8000:8000 \
  ghcr.io/modelship-ai/modelship:latest-cuda deploy
```

`deploy` reads `/modelship/config/models.yaml` by default; pass `--config` for
another path, and any other [CLI flag](model-configuration.md) after it. A
single model needs no config file at all — see
[Quick start](index.md#quick-start).

!!! note "Named volume vs. bind mount for the cache"
    `-v modelship-cache:/.cache` lets Docker create the volume and inherit the
    image's ownership of `/.cache`, so nothing has to be chowned first. To keep
    the weights in a directory you can browse, bind-mount instead
    (`-v ./models-cache:/.cache`) and create it up front — a bind-mounted host
    directory keeps its own ownership, and Docker creates it as `root` if it
    does not exist yet.

!!! tip
    Always set `--shm-size=8g` (or higher) — Ray falls back to slower
    disk-backed storage instead of `/dev/shm` if the container's shared
    memory is too small for the object store.

## Inspecting an image

```bash
docker run --rm ghcr.io/modelship-ai/modelship:latest-cpu info
```

reports the bootstrapper's state: which variant the image was built for, where its
environment lives, and whether it is current. Adding the variant flag
(`info --cpu`) reports the engine's own view instead — accelerator, Python, cache
directory, Ray version, and the `llama-server` binary in use.

## What is in the image

```
/opt/mship/envs/<variant>/.venv   the engine environment (CPython 3.12.10)
/opt/mship/builds/<variant>/      llama-server binaries
/opt/mship/env                    the variant the image was built for
/modelship/config/examples/       reference models.yaml files
/opt/mship/node-cache/            compiled kernels (MSHIP_NODE_CACHE_DIR)
/.cache                           model weights (MSHIP_CACHE_DIR)
```

Mount your own config over `/modelship/config/models.yaml` and a volume at
`/.cache` so weights survive container restarts. Compiled kernels live outside
that volume, so a new container compiles them again once; add
`-v modelship-node-cache:/opt/mship/node-cache` to keep them as well, but never
share that volume with another node. The engine is installed as an
ordinary wheel into that venv — there is no source tree in the image.

## Scaling beyond one node

To join multiple hosts into one Ray cluster with plain `docker run` (no
Kubernetes), see [Multi-node without Kubernetes](multi-node-docker.md). For
Kubernetes, see the [Helm install](install-helm.md).
