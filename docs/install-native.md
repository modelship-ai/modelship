# Native install

One install command on every platform, then one bootstrap to pick the node's role:

```bash
uv tool install mship       # or: pipx install mship / pip install mship
```

| Bootstrap command | Node role |
|---|---|
| `mship bootstrap --cuda` | NVIDIA GPU node (vLLM, Diffusers, llama.cpp GPU offload) |
| `mship bootstrap --cpu` | CPU node (vLLM CPU, llama.cpp, whisper.cpp, sherpa-onnx, stable-diffusion.cpp) |
| `mship bootstrap --metal` | Apple Silicon (Metal offload) |
| `mship bootstrap --thin` | Coordinator/head only — serves nothing itself |

Then, with no variant flag:

```bash
mship deploy --config models.yaml
```

Bootstrapping with no variant is an error that lists these options; there is no
default and no auto-detection, so a node's role is always something you stated.
`MSHIP_VARIANT` is the environment-variable equivalent, for systemd units and CI,
and overrides the recorded role for a single command.

## What `mship bootstrap` does

The `mship` package is a small installer that runs on any Python 3.10+. Bootstrapping:

1. Refuses unsupported platforms (Windows, musl/Alpine) before downloading anything.
2. Checks the variant's build prerequisites — `--cuda` needs `nvcc` and `ninja`.
3. Provisions `~/.modelship/envs/<variant>/` on **CPython 3.12.10**, installing from a
   hash-pinned dependency list shipped inside the package.
4. Fetches the pinned `llama-server` build for the platform.
5. Records the variant in `~/.modelship/env`.

Every node therefore lands on an identical interpreter and dependency set, which is
what lets a native node join a cluster of Docker nodes — Ray refuses to form a
cluster across mismatched Python versions. The [Docker images](install-docker.md)
run these same steps at image-build time.

A copy of the exact pinned list that built each environment is left at
`~/.modelship/envs/<variant>/pins.txt`. `mship info` reports what is provisioned.

`deploy` itself installs nothing. It stops if the environment is missing or was
built by a different `mship` version:

```
error: the cuda environment was built for mship 0.7.12, but this is 0.7.13.

Run: mship bootstrap --cuda
```

So upgrading is `uv tool upgrade mship` followed by `mship bootstrap`.

To bootstrap more than one variant on a host, run `bootstrap` for each — the last
one is the recorded default, and `mship deploy --thin` selects another explicitly.

Bootstrapping never checks for the accelerator itself, so `--cuda` provisions on a
host with no GPU — a golden image, or a node whose driver is not up yet. The variant
flag decides what to install; `deploy` is what refuses to run without the hardware.

## Platform prerequisites

These are not installed for you.

```bash
# macOS — compiles stable-diffusion-cpp-python on first install
xcode-select --install

# Linux, all variants
sudo apt-get install -y build-essential cmake

# Linux --cpu, additionally (lscpu feeds vLLM's CPU NUMA detection; its absence
# surfaces as an opaque `Engine core initialization failed`)
sudo apt-get install -y libnuma1 openssl util-linux

# Linux --cuda, additionally: flashinfer JIT-compiles kernels at model-load time
sudo apt-get install -y ninja-build gnupg
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
  | sudo gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
  | sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-get update && sudo apt-get install -y \
  cuda-nvcc-13-0 cuda-cuobjdump-13-0 libcurand-dev-13-0
```

`nvcc` does not need to be on `PATH` — those packages create the `/usr/local/cuda`
symlink flashinfer looks for. The first vLLM deploy is slow while kernels compile;
they are cached per GPU architecture under `~/.modelship/cache/flashinfer`.

`mship bootstrap --cuda` refuses to run without `nvcc` and `ninja`, rather than
letting the first vLLM deploy fail with `Could not find nvcc` after several GB have
downloaded. The `--cuda` variant ships vLLM and Diffusers, so both are required even
though `llama_server` GGUF offload needs neither. Being software, they are checkable
wherever you provision — unlike the GPU, which bootstrap does not look for at all.
`mship info` reports the same check afterwards.

**Loader coverage on `--cuda`.** `vllm`, `diffusers`, and `llama_server` all get full
GPU. `llama_server` gets a CUDA ggml backend beside the same binary every other
platform runs. ggml skips a backend it cannot load without saying so, so if a GGUF
deploy seems slow, check `"$MSHIP_LLAMA_SERVER_BIN" --list-devices` — it prints
`(none)` instead of a `CUDA0:` line. `stable_diffusion_cpp`, `whispercpp`, and
`sherpa_onnx` are CPU-only here, same as in the image.

## Files on disk

```
~/.modelship/
  env                       the variant bootstrap recorded (MSHIP_VARIANT=…)
  cache/                    models and other downloads (MSHIP_CACHE_DIR)
  node-cache/               compiled kernels (MSHIP_NODE_CACHE_DIR)
  envs/<variant>/           one environment per variant
  builds/<variant>/         llama-server binaries
  bin/uv                    only if uv was not already installed
```

`~/.modelship/env` is written read-only by `bootstrap` and holds a single
`MSHIP_VARIANT=` line. Being an ordinary env file, it drops straight into a systemd
unit:

```ini
[Service]
EnvironmentFile=/home/youruser/.modelship/env
ExecStart=/home/youruser/.local/bin/mship deploy --config /etc/modelship/models.yaml
```

Change it with `mship bootstrap`, not by hand.

`MSHIP_CACHE_DIR` may point at shared storage — model weights are identical on every
node. `MSHIP_HOME` (default `~/.modelship`) must stay node-local: environments and
binaries are platform- and variant-specific, and the kernels in `node-cache/` are
compiled for this node's GPUs. `MSHIP_NODE_CACHE_DIR` can move them, but never onto
shared storage. To reset a variant, delete
`~/.modelship/envs/<variant>/` and bootstrap it again; your models are untouched.

## Scaling beyond one node

To join multiple hosts into one Ray cluster, see
[Multi-node without Kubernetes](multi-node-docker.md) — the `--address`/`--token`
flags work the same for a native node as for a container.
