# CLI reference

`mship` has five commands. Install and provision with `bootstrap`; run a cluster with `start` and `join`; change its models with `deploy`; inspect with `info`.

| Command | What it does | Stays running |
|---|---|---|
| [`mship bootstrap`](#mship-bootstrap) | Installs the engine environment for a node role. Run once per host, and again after upgrading `mship` | no |
| [`mship start`](#mship-start) | Starts a cluster on this machine: head node, API gateway and the models given | yes |
| [`mship join`](#mship-join) | Adds this machine to a running cluster as a worker node | yes |
| [`mship deploy`](#mship-deploy) | Adds, replaces or removes models on a running cluster, or cancels a deploy request | no |
| [`mship info`](#mship-info) | Reports what is provisioned on this host | no |

## Typical flow

```bash
uv tool install mship                  # once
mship bootstrap --cuda                 # once per node role; --cpu, --metal or --thin
mship start --config models.yaml       # head node, gateway and models

# on another machine, same two steps, then:
mship join --gcs-address HEAD:6380     # worker node

# later, from the head's machine (or another one with --ray-dashboard-url):
mship deploy --config more-models.yaml --wait
```

Docker images run the same commands: `docker run ghcr.io/modelship-ai/modelship:latest-cuda start --config …` reaches the same CLI, with the node role fixed by the image tag.

## Node roles

`bootstrap` and the commands after it take one role flag. There is no default and no auto-detection.

| Flag | Role |
|---|---|
| `--cuda` | NVIDIA GPU node |
| `--cpu` | CPU node |
| `--metal` | Apple Silicon |
| `--thin` | Coordinator/head only, serves nothing itself |

`bootstrap` records the role in `~/.modelship/env`, so later commands need no flag. Pass a role flag (or set `MSHIP_VARIANT`) to a later command only to pick a different bootstrapped role on a host that has several. See [Native install](install-native.md).

## `mship bootstrap`

```bash
mship bootstrap --cuda|--cpu|--metal|--thin
```

Provisions a pinned Python 3.12.10 environment for the role, the matching `llama-server` binary where the role serves models, and records the role. It takes no other arguments. After `uv tool upgrade mship`, run it again: the other commands refuse to run against an environment built by a different `mship` version. Details and platform prerequisites: [Native install](install-native.md).

## `mship start`

```bash
mship start [options] [--config models.yaml | --model REF [--<block>.<key> VALUE ...]]
```

Creates this machine's Ray head, brings up the gateway and sends the models given. The gateway then serves exactly those models. It stays running and tears the cluster down on exit. Without `--config` or `--model`, it serves the models its gateway last committed.

```bash
mship start --config models.yaml
mship start --model Qwen/Qwen3-8B-GGUF:'*Q4_K_M.gguf' --loader llama_server --usecase generate
```

Refuses to start when a Ray node already runs on the machine.

## `mship join`

```bash
mship join --gcs-address HOST[:PORT] [options]
```

Starts this machine's Ray node as a worker of the cluster whose head listens at `HOST` (its `--gcs-port`, `6380` by default). Node only: no gateway, no model changes. It stays running and leaves the cluster on exit. See [Multi-node without Kubernetes](multi-node-docker.md).

## `mship deploy`

```bash
mship deploy [--ray-dashboard-url URL] [options] [--config models.yaml | --model REF ...]
mship deploy [--ray-dashboard-url URL] --cancel ID [--wait]
```

Sends a deploy request to the cluster running on this machine, or with `--ray-dashboard-url` to the cluster whose Ray dashboard answers there, and exits.

- Adds models by default; `--reconcile` makes the cluster match the config exactly.
- `--wait` waits for the request to succeed or fail and exits with the outcome.
- `--cancel ID` cancels a queued or running request and rolls back what it submitted.

Exit codes are listed under [CLI Options](model-configuration.md#cli-options).

## `mship info`

```bash
mship info [--cuda|--cpu|--metal|--thin]
```

With no role flag, no `MSHIP_VARIANT` and no recorded role, reports the bootstrapper's state: its version, `MSHIP_HOME`, the recorded role, and each provisioned environment with its size (flagging stale ones, and a missing CUDA toolkit). With a role, it runs inside that environment and reports the engine's view: accelerator, Python, cache directories, Ray version and the `llama-server` binary in use. In a container, `docker run <image> info` gives the engine report.

## Flags

Every flag, its environment variable, default and the commands it applies to are in the [CLI Options table](model-configuration.md#cli-options). Per-model flags such as `--llama-server-config.n-ctx 8192` are generated from the `models.yaml` schema: see [Single-model deploys](model-configuration.md#single-model-deploys-no-config-file). Environment variables work as fallbacks; the CLI wins.

`mship <command> --help` prints the usage and flags of one command.
