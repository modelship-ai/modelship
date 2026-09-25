# Multi-node without Kubernetes

Modelship has two well-worn rungs: one container running its own Ray head, or full
Kubernetes via the [Helm chart](https://github.com/modelship-ai/modelship/blob/main/helm/modelship/README.md). This page covers the
rung in between — a handful of plain `docker run` VMs, no cluster orchestrator,
joined into one cluster: `mship start` on one of them, `mship join` on the rest.

If you're choosing a topology from scratch: single container is right until you
need more GPUs than one box has; this page is right for a few-VM fleet you manage
by hand; Kubernetes is right once you want autoscaling, self-healing pod
scheduling, or you already run k8s for everything else.

## Non-negotiables

- **Private network only.** Ray's cluster-internal RPC (GCS, raylet, object
  manager) is not designed to be internet-facing — `RAY_AUTH_MODE=token`
  authenticates it, but the token travels as plaintext and never expires, so it
  is not encryption and not a substitute for network isolation. Put every node on
  a private VPC/VLAN; for cross-provider fleets, a VPN (WireGuard is a common,
  simple choice) that makes the VMs behave like they're on one LAN. Exposing Ray
  ports to the public internet is exactly the [ShadowRay /
  CVE-2023-48022](https://www.oligo.security/blog/shadowray-attack-ai-workloads-actively-exploited-in-the-wild)
  exposure class — mass-exploited GPU clusters, arbitrary code execution via the
  jobs API.
- **`--network=host` on every node.** Ray advertises the node's own IP and opens a
  wide dynamic port range; bridge networking is a well-documented source of pain
  with it. Host networking on an already-private network sidesteps that entirely.
- **Pin every node to the identical release version.** Floating tags (`:latest`,
  `:latest-cuda`, `:latest-cpu`) are single-node only — Ray refuses to form a
  cluster across mismatched versions. Multi-node means every `docker run`, on
  every VM, uses the same `:X.Y.Z` release — the thin/`-cuda`/`-cpu` suffix can
  differ per node's role, but the version number can't.

## Quick start: a two-VM cluster

VM A becomes the head (control plane + gateway; no models scheduled there) and
uses the **thin** (bare-tag) image — no torch/vllm needed for that role. VM B
joins it as a GPU worker on the `-cuda` tag. Every node in a multi-node cluster
must share the same version, even across variants — pin all of them to the
identical `X.Y.Z` release.

**VM A — head, with cluster auth enabled:**

```bash
docker run -d --network=host --shm-size=8g \
  -v ./models.yaml:/modelship/config/models.yaml \
  -v ./models-cache:/.cache \
  -e MSHIP_STATE_STORE=redis://your-redis-host:6379/0 \
  -e HF_TOKEN=your_token_here \
  ghcr.io/modelship-ai/modelship:0.6.5 start \
  --ray-auth=token --ray-port=6380
```

`--ray-port` defaults to `6380` already (deliberately not Ray's own `6379`
default, which collides with the Redis state store above under host
networking) — passed explicitly here only for clarity.

Retrieve the token a joiner needs (Ray generates and owns it; modelship never
writes or logs it):

```bash
docker exec <head-container> cat ~/.ray/auth_token
```

`mship deploy` against this cluster needs it too: `--ray-auth=token` on VM A,
where the token file is, `--token=<token>` on any other node.

**VM B — joins VM A as a GPU worker:**

```bash
docker run -d --network=host --shm-size=8g --gpus all \
  -v ./models-cache:/.cache \
  -e HF_TOKEN=your_token_here \
  ghcr.io/modelship-ai/modelship:0.6.5-cuda join \
  --cluster=<vm-a-private-ip>:6380 --token=<token-from-above>
```

`HF_TOKEN` goes on every node: the head checks model sources with it, and each
replica reads it from its own node's environment — the driver never forwards it.

A joiner only adds capacity: replicas waiting for room schedule onto it on
their own. To change the model set, run `mship deploy` on any node of the
cluster; a model the last deploy gave up on is retried by `mship deploy --reconcile`.

**`--token` only means anything if the head runs `--ray-auth=token`.** Joining
with a token against a head that has auth disabled doesn't fail — the joiner's
own node starts demanding bearer tokens on *inbound* RPC while the head never
sends them, so the join looks like it succeeded and then cluster↔worker traffic
fails confusingly. There's no reliable way to detect this from the joining side;
treat "auth enabled on the head" and "token passed on the join" as a matched
pair you set deliberately, not independent toggles.

## Ports and firewall

| Port | What | Configurable via |
|---|---|---|
| `8000` | Gateway HTTP API (`ProxyLocation.EveryNode` — every node with ≥1 replica runs a proxy) | `--openai-api-port` |
| `8079` | Prometheus metrics | `--metrics-port`; random on a joiner unless set, and listed in the head's service-discovery file either way |
| `8265` | Ray dashboard (head only) | `--ray-dashboard-port` (bind host separately via `--ray-dashboard-host`, default `127.0.0.1` — keep it there unless you have a specific reason to expose it) |
| GCS (head control plane) | what `mship join --cluster` points at | `--ray-port` (default `6380`) |
| `10002–19999` + node/object manager | Ray's dynamic worker range | not configurable; open the range between fleet nodes |

Open cluster ports **only between fleet nodes** on the private network. From
outside that network, only `8000` (the gateway) should be reachable at all,
ideally behind TLS termination.

## Per-node weight cache

Each node downloads its own copy of whatever models get scheduled onto it —
built-in loaders resolve and validate model references on the driver (auth
failures, missing repos, bad selectors all surface immediately at startup), but
the actual weight download happens **on the node hosting the replica**, not the
driver. A thin head therefore never downloads model weights at all, and a
worker only pulls what it's actually asked to run.

This means, per fleet:
- Every node that can host a given model needs its own disk for that model's
  weights, and its own egress from HuggingFace (or wherever the weights live).
- A **shared NFS/EFS mount** for the cache directory (`MSHIP_CACHE_DIR`) is an
  optional optimization — every node dedupes onto one copy — not a requirement.
  If you don't have shared storage, per-node caches work correctly on their own.
- Local-path `model:` references (as opposed to a HuggingFace repo id) are
  resolved on whichever node actually hosts the replica — the file must exist at
  that path on every node that could host it, since there's no cross-node
  copying of a local reference.

## Capability-aware scheduling

Every node advertises `mship_<loader>` Ray custom resources for whatever it can
actually run (probed via `importlib.util.find_spec()`, plus a real-binary check
for `llama_server`), and every deploy requests its loader's resource regardless
of `num_gpus`. A thin head advertises none — it never accidentally schedules a
model onto itself. A worker missing the right extras (e.g. a `-cpu` node given a
`loader: vllm` config that needs `cuda`) just leaves that deploy pending; Ray's
own scheduling message names the missing resource, nothing hard-fails at driver
time. Override the probe wholesale with `MSHIP_NODE_CAPABILITIES` (JSON) if
auto-detection is wrong or you want to disable a loader on one node.

## `MSHIP_STATE_STORE=redis://` is the multi-node recommendation

Without it, the effective config (this gateway's desired model set) lives in a
cluster-scoped Ray actor — it survives a redeploy or coordinator restart, but not
the loss of the head/cluster itself. A `redis://` store survives cluster loss too, so `mship start --reconcile`
with no `--config` on a fresh cluster restores the real model set instead of
coming back empty. See [State store
(`MSHIP_STATE_STORE`)](model-configuration.md#state-store-mship_state_store) for
the full connection-URI reference — the head is otherwise a single point of
failure for this state, same as it is for Ray's GCS itself.

If that Redis needs a password, pass `-e MSHIP_REDIS_PASSWORD=…` on every node
and keep it out of the URI: gateway replicas can run on any node, and the URI
they get is password-free — each node adds its own before connecting.

## Co-location: running more than one node per physical box

Co-location is a supported topology, not a footgun to avoid — the two patterns
below are deliberate ways to pack more onto hardware you already have. What
*is* still your responsibility: fencing which physical resources each container
gets, so two containers on one box don't both believe they own the same
hardware. See [AGENTS.md's co-location
note](https://github.com/modelship-ai/modelship/blob/main/AGENTS.md#gotchas) for the full fencing discipline
(`--gpus device=N`, `--node-memory` + `--shm-size`, `--cpuset-cpus` +
`--node-num-cpus`) — this page only covers the two topologies it unlocks.
Reserving more GPUs than a container can actually see is refused at startup
(not silently broken later); each node also logs every GPU it sees, by index,
name, and UUID, at startup — check that log to confirm two co-located
containers were actually handed distinct physical cards.

### Head-farm: several cluster heads on one machine

One box can host the control plane for 3-4 independent modelship clusters, each
a separate `mship start` container with its own distinct ports. Each needs its own
process namespace (no `--pid=host`): `mship start` refuses to run while it can see
another Ray node.

```bash
docker run -d --network=host --shm-size=8g \
  -v ./cluster-a/models.yaml:/modelship/config/models.yaml \
  -v ./cluster-a/cache:/.cache \
  ghcr.io/modelship-ai/modelship:0.6.5 start \
  --ray-port=6380 --openai-api-port=8000 --ray-dashboard-port=8265 --metrics-port=8079

docker run -d --network=host --shm-size=8g \
  -v ./cluster-b/models.yaml:/modelship/config/models.yaml \
  -v ./cluster-b/cache:/.cache \
  ghcr.io/modelship-ai/modelship:0.6.5 start \
  --ray-port=6381 --openai-api-port=8001 --ray-dashboard-port=8266 --metrics-port=8089
```

Each head needs a distinct `--ray-port`, `--openai-api-port`, `--ray-dashboard-port`,
and `--metrics-port` — see the port table above for what each one
gates. Workers on other machines join whichever cluster they're meant to serve,
by pointing `mship join --cluster` at that head's GCS port.

### One GPU, shared across two clusters

A single physical GPU can back worker containers in two *different* clusters at
once — this is plain GPU sharing (the same idea as two vLLM processes each
capped at `gpu_memory_utilization=0.4`), not a bookkeeping error, because each
cluster's resource ledger is independent and has no visibility into the other's:

```bash
# Joins cluster A
docker run -d --network=host --gpus device=0 \
  ghcr.io/modelship-ai/modelship:0.6.5-cuda join \
  --cluster=<cluster-a-head>:6380 --node-num-gpus=1

# Joins cluster B — same physical GPU, different cluster
docker run -d --network=host --gpus device=0 \
  ghcr.io/modelship-ai/modelship:0.6.5-cuda join \
  --cluster=<cluster-b-head>:6380 --node-num-gpus=1
```

Ray does not arbitrate VRAM across independent clusters — that budget is
yours to manage, the same way you'd size two co-resident models on one GPU
within a single cluster (fractional `num_gpus`, which sizes vLLM's
`gpu_memory_utilization` for you, or `llama_server`'s `n_gpu_layers`) so both
sides' footprints actually fit together on the card.

## See also

- [helm/modelship/README.md](https://github.com/modelship-ai/modelship/blob/main/helm/modelship/README.md) — the Kubernetes rung:
  same image variants and version-pinning rule, but autoscaling, self-healing
  pod scheduling, and Ray cluster auth set with one value (`rayAuth`) instead
  of this page's manual token setup.
- [development.md](development.md) — the full CLI/env var reference table,
  image variants, and dev-container setup.
- [model-configuration.md](model-configuration.md) — `models.yaml` reference,
  including the [state store](model-configuration.md#state-store-mship_state_store)
  section referenced above.
