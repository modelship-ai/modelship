# modelship Helm chart

Deploy [modelship](https://github.com/modelship-ai/modelship) — an OpenAI-compatible,
multi-model inference server — on Kubernetes via [KubeRay](https://github.com/ray-project/kuberay).

The chart brings up a **RayCluster** whose head runs `mship start` and whose
worker groups run `mship join` — the same commands as a Docker or native install —
and a **RayJob** that runs `mship deploy` **on** the cluster (KubeRay's
supported way to run a driver against a RayCluster) and deploy the models
declared in your `models.yaml`. Re-running (`helm upgrade`) re-applies the config
additively, or reconciles it when `deploy.reconcile=true`. The RayJob succeeds only
when the deploy does: a failed deploy is rolled back, and a model waiting for
capacity keeps the RayJob running until the nodes arrive (cancel it with
`mship deploy --cancel ID`, on the head or [from your
machine](../../docs/install-helm.md#deploy-or-cancel-from-your-machine)).

The head pod turns Ready only once `mship start` is done: its `startupProbe` waits
for the gateway's `/<gateway>/health`, which start brings up last, after the deploy
coordinator. So the RayJob, which waits for the head, never reaches one still
starting, after a head restart included. Readiness and liveness are KubeRay's own.

Each successful deploy commits this gateway's model set, keeping the one before it,
to a **state store** (see [Head-node HA](#head-node-ha-redis)). Routing is
recomputed from Ray Serve's own state and that committed set, so it comes back by
itself after a head restart.

## Prerequisites

- A Kubernetes cluster (a local [kind](https://kind.sigs.k8s.io/) cluster works
  for the CPU image; GPU models need real GPU nodes with the NVIDIA device plugin).
- **The KubeRay operator + CRDs, 1.6 or newer** (tested on 1.7.1). This is a
  cluster-scoped, install-once dependency. Either install it yourself:

  ```bash
  helm repo add kuberay https://ray-project.github.io/kuberay-helm/
  helm install kuberay-operator kuberay/kuberay-operator --version 1.7.1
  ```

  …or, on a single-tenant cluster, let this chart bootstrap it:
  `--set kuberay-operator.enabled=true`.

  To upgrade an existing operator, apply the new CRDs first; `helm upgrade`
  never updates them:

  ```bash
  helm pull kuberay/kuberay-operator --version 1.7.1 --untar
  kubectl apply --server-side --force-conflicts -f kuberay-operator/crds/
  helm upgrade kuberay-operator kuberay/kuberay-operator --version 1.7.1
  ```
- For GPU models: a node pool with `nvidia.com/gpu` resources.
- Helm 3 or 4.

## Install

```bash
# From the repo (path install; fetch the vendored operator subchart first):
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm dependency build ./helm/modelship
helm install mship ./helm/modelship -f my-values.yaml

# From GHCR (OCI). Chart version is kept in lockstep with the app/image version
# (see https://github.com/modelship-ai/modelship/releases for the latest), so
# --version <X.Y.Z> always pairs with the matching image:
helm install mship oci://ghcr.io/modelship-ai/charts/modelship --version <X.Y.Z> -f my-values.yaml
```

Because images and model weights take time to pull, raise Helm's timeout:
`--timeout 20m --wait`. Note that `--wait` does **not** track the RayJob to
completion — watch `kubectl get rayjob` and the gateway `/readyz` for readiness.

## Configure your models

Set `models.config` to your `models.yaml` contents (see `config/examples/` in the
repo). The deploy RayJob carries its own copy, so each `helm upgrade` applies
exactly the config it was given.

```yaml
models:
  config: |
    models:
      - name: qwen
        loader: vllm
        model: Qwen/Qwen2.5-7B-Instruct
        num_gpus: 1
```

Gated/private weights need a Hugging Face token:

```yaml
secrets:
  huggingfaceToken: "hf_..."   # mounted as HF_TOKEN
```

(Or reference an existing Secret with key `HF_TOKEN` via
`secrets.existingSecret`.)

## Topology

- **Head** — coordination-only; runs `mship start`, which brings up GCS, Serve
  and the gateway. Always runs the `thin` image (no torch/vllm, and it advertises
  no CPUs or GPUs) regardless of the cluster-wide `image.variant`, so no model can
  schedule there — override with `head.image.variant` if you genuinely want
  capacity on the head. The RayJob submitter pod uses the same (thin) image.
  KubeRay's `ray.io/overwrite-container-cmd` annotation keeps these commands
  instead of generating `ray start`.
- **Serve HTTP proxies** — one runs on **every** Ray node (`proxy_location=EveryNode`),
  not just the head, and the gateway Service load-balances across all of them so
  ingress survives losing any single pod. Each proxy can route to any gateway
  replica wherever it's scheduled. The gateway autoscales between
  `gateway.autoscaling.minReplicas` and `maxReplicas`; set `minReplicas` to 2 or more
  (with ≥1 worker) for routing/ingress HA. Each replica copies its routing table from
  the gateway coordinator.
- **Worker groups** — where models actually run. **Empty by default**, so a
  no-values install brings up only the head and schedules nothing; declare the
  groups that match your hardware under `workerGroups` (a commented cuda+cpu
  example ships in `values.yaml`). This is a **list — Helm replaces it wholesale**
  (no per-item merge), so always declare the full set you want; omitting the key
  keeps the empty default. Each worker runs `mship join`, which detects the
  loaders its image can run. Its CPU count and memory budget come from the group's
  limits, else its requests; set `MSHIP_NODE_*` in the group's `env` to override.
  With neither, the worker counts the host's free memory as its own.
  KubeRay never deletes the pods of a group removed from the list, renamed ones
  included ([kuberay#1739](https://github.com/ray-project/kuberay/issues/1739)): scale
  the group to 0 (`replicas`, `minReplicas`, `maxReplicas`) in one upgrade, then
  remove it in the next.
- **Cache** — a shared PVC for model weights at `/.cache`. Single-node clusters
  can use `ReadWriteOnce`; **multi-node requires `ReadWriteMany`** so every worker
  shares one copy.
- **Node cache** — an emptyDir per pod at `/opt/mship/node-cache` for vLLM, Triton
  and FlashInfer compile caches, kept off the shared PVC. A rescheduled pod
  compiles them again; cap the size with `nodeCache.sizeLimit`.
- **/dev/shm** — an in-memory emptyDir (default 8Gi); vLLM/NCCL need it.

## Reaching the gateway

The gateway Service load-balances across the `serve` port of every Ray pod. Serve
runs a proxy only on nodes hosting at least one replica. Check `/readyz`
for app-level readiness — it returns 503 until all models are loaded (use it for
an external LB/Ingress health check). Port-forward for local access, or set
`service.type=LoadBalancer`:

```bash
kubectl port-forward svc/<release>-gateway 8000:8000
curl http://localhost:8000/modelship/v1/models
```

## Redis (required)

`redis.address` is required. The chart wires an address but does **not** deploy
Redis — bring your own (a small single instance with a PVC is plenty). Rendering
fails with a clear message if it's unset, because there is no durable fallback to
degrade to.

```yaml
redis:
  address: my-redis-master:6379
  password: "s3cret"          # or existingSecret + passwordKey
```

One Redis backs three things at once:

1. **Ray GCS fault tolerance** (`gcsFaultToleranceOptions`) — the head pod runs the
   GCS (Ray's control store). In-memory, a head restart (OOM, drain, eviction) loses
   cluster state: KubeRay recycles the workers and every model has to be redeployed —
   minutes of outage for a routine reschedule. Backed by Redis, a restarted head
   recovers GCS; workers and model actors **survive**, and Serve's controller
   redeploys anything that died. The restart becomes a sub-minute blip.
2. **The modelship state store** (`MSHIP_STATE_STORE=redis://…`) — each gateway's
   committed version lives in Redis, so the gateway coordinator, which rebuilds routing
   from Serve's state on recovery, still knows which deployment each model should run.
3. **`/v1/responses` conversations** — stored responses survive head restarts and
   full cluster loss, so `previous_response_id` keeps working across them.

**What recovers automatically:**

| event | outcome |
|-------|---------|
| head pod restart | actors survive, routing self-heals — no redeploy |
| full cluster loss, Redis kept | Serve restores from Redis, routing is recomputed; conversations intact |
| full cluster loss, Redis also gone | `helm upgrade` |

`redis.externalStorageNamespace` (default: the release name) namespaces Ray's keys and
modelship's (`modelship/state/<namespace>/`), so a recreated cluster recovers and
releases can share a Redis db. Same-named releases in two k8s namespaces need
distinct values. The password comes from the Secret and never lands in the pod manifest;
Ray itself passes it on the command line of the head's `gcs_server` and `raylet`.

> Before v0.7.0 `redis.enabled=false` fell back to a `file://` state store on the
> cache PVC. That backend is gone — see the main
> [state-store docs](../../docs/model-configuration.md#state-store-mship_state_store).
> Outside k8s the default is `memory://`, which is cluster-scoped but dies with the
> cluster.

## Uninstall

```bash
helm uninstall modelship
```

Uninstall first runs a Job (a pre-delete hook) that:
1. deletes the deploy RayJob and the RayCluster;
2. waits up to 3 minutes for the RayCluster to go, while KubeRay deletes Ray's keys
   from Redis;
3. deletes modelship's keys (`modelship/state/<namespace>/*`: deploy versions,
   `/v1/responses` conversations).

Nothing of the release stays in Redis; back it up first to keep conversations. If the
Job fails, Helm stops the uninstall there, and `kubectl logs job/modelship-uninstall`
says why. `helm uninstall --no-hooks` skips it: the RayJob and modelship's keys stay,
and with a chart-created Secret KubeRay's cleanup can't start, so the RayCluster takes
5 minutes to go and leaves Ray's keys.

The RayCluster carries KubeRay's Redis cleanup finalizer from creation. An operator
run with `ENABLE_GCS_FT_REDIS_CLEANUP=false` never removes it: the Job waits its 3
minutes, and the cluster stays up after uninstall until you run

```bash
kubectl patch raycluster <release> --type=merge -p '{"metadata":{"finalizers":null}}'
```

Ray's keys then stay in Redis, and a reinstall under the same
`redis.externalStorageNamespace` starts from the old cluster's state. Delete
`RAY<namespace>@*` too for a clean start.

## Ray cluster authentication

Always on: the head's dashboard listens on the pod network, and its job API runs
arbitrary code (see the ShadowRay/CVE-2023-48022 background in the main
[multi-node docs](../../docs/multi-node-docker.md)). The chart sets the
RayCluster's `authOptions` and passes `--enable-ray-auth` to `mship start`.
KubeRay gives the same token to the head, every worker group **and** the RayJob
submitter (the pod that runs `ray job submit` to deploy your `models.yaml`), and
uses it for its own job-status calls.

KubeRay generates the token once, in a Secret named after the RayCluster, and
deletes it with the cluster. Read it with:

```bash
kubectl get secret <release> -o jsonpath='{.data.auth_token}' | base64 -d
```

`mship deploy --ray-dashboard-url` sends deploys and cancels from your machine with
it, over a port-forward to `<release>-head-svc:8265`; see [Deploy or
cancel from your machine](../../docs/install-helm.md#deploy-or-cancel-from-your-machine).

To manage the token yourself, point `rayAuth.existingSecret` at a Secret holding
it under key `auth_token`. Any string works: Ray's own check is a shared-secret
equality comparison, not an issued credential (e.g. `openssl rand -hex 32`).

This never gates the OpenAI API (`gateway.port`) or Prometheus metrics
(`metrics.port`) — only Ray's own dashboard and cluster-internal RPC.

## Common values

| Key | Default | Purpose |
|-----|---------|---------|
| `image.repository` / `image.tag` | `ghcr.io/modelship-ai/modelship` / `<app version>` | Stamped to the release version |
| `image.variant` | `cuda` | `cuda`\|`cpu`\|`thin`. Worker default — appends `-cuda`/`-cpu` to the tag (`thin` is bare). Set `cpu` on CPU-only clusters, or per worker group for a mixed cluster. Does **not** affect the head (see below) |
| `head.image.variant` | `thin` | The head/RayJob submitter always default to `thin` regardless of `image.variant` above — override only if you genuinely want model capacity on the head |
| `rayVersion` | `2.54.1` | Must match the Ray in the image |
| `models.config` | `models: []` | Your model set |
| `gateway.autoscaling.minReplicas` / `maxReplicas` | `1` / `4` | Range the API gateway autoscales in; a `minReplicas` of 2 or more (with ≥1 worker) gives routing/ingress HA |
| `gateway.autoscaling.targetOngoingRequests` | `64` | Ongoing requests per gateway replica that autoscaling aims for |
| `gateway.maxOngoingRequests` | `1024` | Most requests one gateway replica handles at once |
| `secrets.huggingfaceToken` | `""` | HF token |
| `cache.size` / `cache.accessModes` | `100Gi` / `[ReadWriteOnce]` | Shared weight cache |
| `nodeCache.sizeLimit` | `""` (uncapped) | Per-pod compile-cache emptyDir; a pod over the cap is evicted |
| `workerGroups` | `[]` | Worker pool layout (a list — set the full set; copy the example in `values.yaml`) |
| `head.env` / `workerGroups[].env` | `[]` | Extra env for the head's `mship start` / a group's `mship join` (e.g. `MSHIP_NODE_NUM_GPUS`) |
| `deploy.reconcile` | `false` | Remove dropped models on upgrade |
| `deploy.replaceStrategy` | `blue_green` | How changed models are replaced |
| `redis.address` | `""` | **Required.** `host:port` of your Redis — backs GCS-FT + the state store (see [Redis](#redis-required)) |
| `redis.password` / `redis.existingSecret` | `""` | Redis password inline, or reference an existing Secret (`passwordKey`) |
| `rayAuth.existingSecret` | `""` | An existing Secret holding the Ray auth token under key `auth_token`; empty, KubeRay generates it (see [Ray cluster authentication](#ray-cluster-authentication)) |
| `service.type` | `ClusterIP` | Set `LoadBalancer` to expose externally |
| `podMonitor.enabled` | `false` | Prometheus Operator scraping; needs `metrics.enabled` |
| `prometheusRule.enabled` | `false` | Ship the modelship alert rules as a PrometheusRule |
| `grafanaDashboard.enabled` | `false` | Ship the Grafana dashboard as a sidecar-imported ConfigMap |
| `kuberay-operator.enabled` | `false` | Bootstrap the operator as a subchart |

See [values.yaml](values.yaml) for the full set with inline documentation.
