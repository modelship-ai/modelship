# Installation

Pick the install method for the node you are setting up. All three run the same
engine at the same pinned versions — the images are built by running the native
install against the release wheel, so a native node and a container node can join
the same cluster.

| Method | Use it for | Guide |
|---|---|---|
| **Docker** | Servers, GPU boxes, anything already running containers | [Docker install](install-docker.md) |
| **Native** | Apple Silicon, homelab hosts, systemd units, bare metal | [Native install](install-native.md) |
| **Helm / Kubernetes** | Multi-node clusters on Kubernetes, via KubeRay | [Helm install](install-helm.md) |

## Requirements

- **Docker**, or any **Python 3.10+** for a native install — `mship` provisions the
  CPython 3.12.10 the engine needs itself
- **NVIDIA GPU** (optional) — 16 GB+ VRAM recommended for a full stack (LLM +
  TTS + STT + embeddings) via vLLM; 8 GB is sufficient for lighter setups.
  Not required when using the vLLM or llama.cpp backends on CPU
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)**
  — required only when running GPU models in Docker
- **HuggingFace token** for gated models
- **glibc** for a native install — Alpine and other musl distros are refused

## Node roles

Whichever method you use, a node is provisioned for exactly one role. There is no
default and no auto-detection — the role is always something you stated, as an
image tag, a bootstrap flag, or a chart value.

| Role | Serves models | For |
|---|---|---|
| `cuda` | yes | NVIDIA GPU node (vLLM, Diffusers, llama.cpp GPU offload) |
| `cpu` | yes | CPU node (vLLM CPU, llama.cpp, whisper.cpp, sherpa-onnx, stable-diffusion.cpp) |
| `metal` | yes | Apple Silicon (Metal offload) — native only |
| `thin` | no | Coordinator/head only; joins capacity from other nodes |

## Next

- [Quick start](index.md#quick-start) — one command per hardware target, then a first request
- [Model Configuration](model-configuration.md) — the full `models.yaml` reference
- [Multi-node without Kubernetes](multi-node-docker.md) — joining hosts with plain `docker run`
- [Development Setup](development.md) — building from source and the Dev Container
