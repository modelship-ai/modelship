# CLI reference

`mship` has six commands. Install and provision with `bootstrap`; run a cluster with `start` and `join`; change its models with `deploy`; shut it down with `stop`; inspect with `info`.

| Command | What it does | Stays running |
|---|---|---|
| [`mship bootstrap`](#mship-bootstrap) | Installs the engine environment for a node role. Run once per host, and again after upgrading `mship` | no |
| [`mship start`](#mship-start) | Starts a cluster on this machine: head node, API gateway and the models given | yes |
| [`mship join`](#mship-join) | Adds this machine to a running cluster as a worker node | yes |
| [`mship deploy`](#mship-deploy) | Adds, replaces or removes models on a running cluster, or cancels a deploy request | no |
| [`mship stop`](#mship-stop) | Stops the nodes running on this machine | no |
| [`mship info`](#mship-info) | Reports what is provisioned on this host | no |

## Typical flow

```bash
uv tool install mship                  # once
mship bootstrap --cuda                 # once per node role; --cpu, --vulkan, --metal or --thin
mship start --config models.yaml       # head node, gateway and models

# on another machine, same two steps, then:
mship join --gcs-address HEAD:6380     # worker node

# later, from the head's machine (or another one with --ray-dashboard-url):
mship deploy --config more-models.yaml --wait

# done: on each machine
mship stop
```

Docker images run the same commands: `docker run ghcr.io/modelship-ai/modelship:latest-cuda start --config …` reaches the same CLI, with the node role fixed by the image tag.

## Node roles

`bootstrap` and the commands after it take one role flag. There is no default and no auto-detection.

| Flag | Role |
|---|---|
| `--cuda` | NVIDIA GPU node |
| `--cpu` | CPU node |
| `--vulkan` | Vulkan GPU node (Linux) |
| `--metal` | Apple Silicon |
| `--thin` | Coordinator/head only, serves nothing itself |

`bootstrap` records the role in `~/.modelship/env`, so later commands need no flag. Pass a role flag (or set `MSHIP_VARIANT`) to a later command only to pick a different bootstrapped role on a host that has several. See [Native install](install-native.md).

## `mship bootstrap`

```bash
mship bootstrap --cuda|--cpu|--vulkan|--metal|--thin
```

Provisions a pinned Python 3.12.10 environment for the role, the matching `llama-server` binary where the role serves models, and records the role. It takes no other arguments. After `uv tool upgrade mship`, run it again: the other commands refuse to run against an environment built by a different `mship` version. Details and platform prerequisites: [Native install](install-native.md).

## `mship start`

```bash
mship start [options] [--config models.yaml | --model REF [--model.<key> VALUE ...] ... [--pipeline NAME [--pipeline.<key> VALUE ...] ...]]
```

Creates this machine's Ray head, brings up the gateway and sends the models given. The gateway then serves exactly those models. It stays running and tears the cluster down on exit. Without `--config` or `--model`, it serves the models its gateway last committed.

```bash
mship start --config models.yaml
mship start --model Qwen/Qwen3-8B-GGUF:'*Q4_K_M.gguf' --model.loader llama_server --model.usecase generate
```

Refuses to start when a Ray node already runs on the machine, or when one of its ports is held by another program or given to two of them: `--gcs-port`, `--openai-api-port`, `--metrics-port` and, when set, `--ray-dashboard-port`.

## `mship join`

```bash
mship join --gcs-address HOST[:PORT] [options]
```

Starts this machine's Ray node as a worker of the cluster whose head listens at `HOST` (its `--gcs-port`, `6380` by default). Node only: no gateway, no model changes. It stays running and leaves the cluster on exit. See [Multi-node without Kubernetes](multi-node-docker.md).

Refuses to start when `--metrics-port` is set and another program holds that port.

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

## `mship stop`

```bash
mship stop [--timeout SECONDS]
```

Stops every node that `mship start` or `mship join` runs on this machine, worker nodes first. Each one shuts down as it does on Ctrl+C in its terminal: a head deletes its models (with a Redis-backed GCS it keeps them for the next head) and stops Ray, a worker leaves the cluster. A node still running after `--timeout` seconds (default 90) is killed, with every process it started, model backends included. Ray nodes that `mship` did not start are left alone.

Exits `0` once nothing of the nodes is left, or when none was running, and `1` when a process survived, listing it. In Docker, stop the container with `docker stop` instead: see [Stopping a node](install-docker.md).

## `mship info`

```bash
mship info [--cuda|--cpu|--vulkan|--metal|--thin]
```

Plain `mship info` reports the bootstrapper's state: its version, `MSHIP_HOME`, the recorded role, and each provisioned environment with its size (flagging stale ones, and a missing CUDA toolkit). The recorded role does not select the engine report: that needs an explicit role flag or `MSHIP_VARIANT`, and then it runs inside that environment and reports the engine's view: variant, Python, cache directories, Ray version, the `llama-server` binary in use and, on `cuda`, `vulkan` and `metal`, how many GPUs the node sees (the count `start` and `join` advertise) and each one (on `cuda` also the CUDA version its driver supports), or why it sees none. It then prints the Ray labels `start` and `join` give this node (`modelship/variant`, `modelship/os`, `modelship/hostname` and any `--node-labels`), and, for each cluster a node on this machine belongs to, its live nodes with their CPUs, GPUs and labels:

```
node labels: modelship/variant=vulkan, modelship/os=linux, modelship/hostname=node-b
cluster 10.0.0.1:6380: 2 node(s)
  node-a (head): 8 CPU(s), 2 GPU(s)  gpu=fast, modelship/hostname=node-a, modelship/os=linux, modelship/variant=cuda
  node-b: 16 CPU(s), 1 GPU(s)  modelship/hostname=node-b, modelship/os=linux, modelship/variant=vulkan
```

These are the labels a model's `node_selector` matches. A cluster started with `--enable-ray-auth` needs `MSHIP_RAY_AUTH_TOKEN` in the environment (on Kubernetes, KubeRay's `RAY_AUTH_TOKEN` in the pod is enough); without it, `info` says so. With no node running here, it prints `cluster: no node running on this machine`. In a container, `docker run <image> info` gives the bootstrapper report; pass the image's role (`info --cpu`) for the engine's.

## Flags

Every flag, its environment variable, default and the commands it applies to are in the [CLI Options table](model-configuration.md#cli-options). Per-model flags such as `--model.llama-server-config.n-ctx 8192` are generated from the `models.yaml` schema: see [Deploying from the command line](model-configuration.md#deploying-from-the-command-line-no-config-file). Environment variables work as fallbacks; the CLI wins.

`mship start --help`, `mship join --help`, `mship deploy --help` and `mship stop --help` print the usage and flags of that command. `bootstrap` and `info` take only the role flag.
