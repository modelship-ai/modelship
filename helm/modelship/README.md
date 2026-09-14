# modelship Helm chart

Deploy [modelship](https://github.com/modelship-ai/modelship) — an OpenAI-compatible,
multi-model inference server — on Kubernetes via [KubeRay](https://github.com/ray-project/kuberay).

The chart brings up a **RayCluster** (one CPU-only head + worker groups) and a
**RayJob** that runs `mship deploy` **on** the cluster (KubeRay's
supported way to run a driver against a RayCluster) and deploy the models
declared in your `models.yaml`. Re-running (`helm upgrade`) re-applies the config
additively, or reconciles it when `deploy.reconcile=true`.

Each deploy persists this gateway's **effective config** (its desired model set)
and routing registry to a **state store** (see [Head-node HA](#head-node-ha-redis)),
so the gateway self-heals its routing after a head restart and `helm upgrade`
reconciles the live cluster back to the recorded set.

## Prerequisites

- A Kubernetes cluster (a local [kind](https://kind.sigs.k8s.io/) cluster works
  for the CPU image; GPU models need real GPU nodes with the NVIDIA device plugin).
- **The KubeRay operator + CRDs.** This is a cluster-scoped, install-once
  dependency. Either install it yourself:

  ```bash
  helm repo add kuberay https://ray-project.github.io/kuberay-helm/
  helm install kuberay-operator kuberay/kuberay-operator
  ```

  …or, on a single-tenant cluster, let this chart bootstrap it:
  `--set kuberay-operator.enabled=true`.
- For GPU models: a node pool with `nvidia.com/gpu` resources.

## Install

```bash
# From the repo (path install):
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
repo), or point at a ConfigMap you manage with `models.existingConfigMap`.

```yaml
models:
  config: |
    models:
      - name: qwen
        loader: vllm
        model: Qwen/Qwen2.5-7B-Instruct
        num_gpus: 1
```

Gated/private weights need a Hugging Face token; gateway auth needs API keys:

```yaml
secrets:
  huggingfaceToken: "hf_..."   # mounted as HF_TOKEN
  apiKeys: "sk-local-1,sk-local-2"
```

(Or reference an existing Secret with keys `HF_TOKEN` / `MSHIP_API_KEYS` via
`secrets.existingSecret`.)

## Topology

- **Head** — coordination-only (`num-gpus: 0`, `num-cpus: 0`); runs GCS and the
  gateway. Always runs the `thin` image (no torch/vllm) regardless of the
  cluster-wide `image.variant`, so no model can schedule there even if it only
  asks for CPU — override with `head.image.variant` if you genuinely want
  capacity on the head. The RayJob submitter pod uses the same (thin) image.
- **Serve HTTP proxies** — one runs on **every** Ray node (`proxy_location=EveryNode`),
  not just the head, and the gateway Service load-balances across all of them so
  ingress survives losing any single pod. Each proxy can route to any gateway
  replica wherever it's scheduled. Set `gateway.replicas > 1` (with ≥1 worker) for
  routing/ingress HA; replicas keep their routing tables in sync via the deploy
  coordinator.
- **Worker groups** — where models actually run. **Empty by default**, so a
  no-values install brings up only the head and schedules nothing; declare the
  groups that match your hardware under `workerGroups` (a commented cuda+cpu
  example ships in `values.yaml`). This is a **list — Helm replaces it wholesale**
  (no per-item merge), so always declare the full set you want; omitting the key
  keeps the empty default.
- **Cache** — a shared PVC for model weights at `/.cache`. Single-node clusters
  can use `ReadWriteOnce`; **multi-node requires `ReadWriteMany`** so every worker
  shares one copy.
- **Node cache** — an emptyDir per pod at `/opt/mship/node-cache` for vLLM, Triton
  and FlashInfer compile caches, kept off the shared PVC. A rescheduled pod
  compiles them again; cap the size with `nodeCache.sizeLimit`.
- **/dev/shm** — an in-memory emptyDir (default 8Gi); vLLM/NCCL need it.

## Reaching the gateway

The gateway Service load-balances across the Serve proxy on every Ray node, gated
per-pod by proxy health (a pod only joins once its proxy is up). Check `/readyz`
for app-level readiness — it returns 503 until all models are loaded (use it for
an external LB/Ingress health check). Port-forward for local access, or set
`service.type=LoadBalancer`:

```bash
kubectl port-forward svc/<release>-modelship-gateway 8000:8000
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
2. **The modelship state store** (`MSHIP_STATE_STORE=redis://…`) — the deploy
   coordinator's routing registry and effective config live in Redis, so the gateway
   self-heals its routing on recovery instead of coming back empty.
3. **`/v1/responses` conversations** — stored responses survive head restarts and
   full cluster loss, so `previous_response_id` keeps working across them.

**What recovers automatically:**

| event | outcome |
|-------|---------|
| head pod restart | actors survive, routing self-heals — no redeploy |
| full cluster loss, Redis kept | Serve + coordinator restore from Redis; conversations intact |
| full cluster loss, Redis also gone | `helm upgrade` |

`externalStorageNamespace` is pinned to the release name so a recreated cluster
recovers; the password is injected via the Secret and expanded into the URI at
runtime, never landing in the pod manifest or argv.

> Before v0.7.0 `redis.enabled=false` fell back to a `file://` state store on the
> cache PVC. That backend is gone — see the main
> [state-store docs](../../docs/model-configuration.md#state-store-mship_state_store).
> Outside k8s the default is `memory://`, which is cluster-scoped but dies with the
> cluster.

## Ray cluster authentication (optional)

Off by default — the same posture as a single-node deploy, and consistent with
Ray's own insecure-by-default stance (see the ShadowRay/CVE-2023-48022
background in the main [multi-node docs](../../docs/multi-node-docker.md)).
Enable it and every Ray pod — the head, every worker group, **and** the RayJob
submitter (the pod that runs `ray job submit` to deploy your `models.yaml`) —
gets `RAY_AUTH_MODE=token` plus the same `RAY_AUTH_TOKEN`. All three must agree:
a mismatch breaks cluster-internal RPC or job submission, not just one of them.

```yaml
rayAuth:
  enabled: true
  token: "s0me-long-random-string"   # or existingSecret + tokenKey
```

Unlike modelship's own-head Docker path — where Ray generates and owns the
token at `~/.ray/auth_token` because there's no "before the head exists" moment
— the chart has no such moment either way, so **you** supply the token. Any
string works: Ray's own check is a shared-secret equality comparison, not an
issued credential. Generate one with `ray get-auth-token --generate` (run
anywhere with Ray installed) or `openssl rand -hex 32`.

This never gates the OpenAI API (`gateway.port`) or Prometheus metrics
(`metrics.port`) — only Ray's own dashboard and cluster-internal RPC.

## Common values

| Key | Default | Purpose |
|-----|---------|---------|
| `image.repository` / `image.tag` | `ghcr.io/modelship-ai/modelship` / `<app version>` | Stamped to the release version |
| `image.variant` | `cuda` | `cuda`\|`cpu`\|`thin`. Worker default — appends `-cuda`/`-cpu` to the tag (`thin` is bare). Set `cpu` on CPU-only clusters, or per worker group for a mixed cluster. Does **not** affect the head (see below) |
| `head.image.variant` | `thin` | The head/RayJob submitter always default to `thin` regardless of `image.variant` above — override only if you genuinely want model capacity on the head |
| `rayVersion` | `2.54.1` | Must match the Ray in the image |
| `models.config` / `models.existingConfigMap` | `models: []` | Your model set |
| `gateway.replicas` | `1` | API gateway replicas; raise (with ≥1 worker) for routing/ingress HA |
| `secrets.huggingfaceToken` / `secrets.apiKeys` | `""` | HF token / gateway API keys |
| `cache.size` / `cache.accessModes` | `100Gi` / `[ReadWriteOnce]` | Shared weight cache |
| `nodeCache.sizeLimit` | `""` (uncapped) | Per-pod compile-cache emptyDir; a pod over the cap is evicted |
| `workerGroups` | `[]` | Worker pool layout (a list — set the full set; copy the example in `values.yaml`) |
| `deploy.reconcile` | `false` | Remove dropped models on upgrade |
| `deploy.replaceStrategy` | `blue_green` | How changed models are replaced |
| `redis.address` | `""` | **Required.** `host:port` of your Redis — backs GCS-FT + the state store (see [Redis](#redis-required)) |
| `redis.password` / `redis.existingSecret` | `""` | Redis password inline, or reference an existing Secret (`passwordKey`) |
| `rayAuth.enabled` | `false` | Ray cluster authentication (`RAY_AUTH_MODE=token`) across head, workers, and the RayJob submitter (see [Ray cluster authentication](#ray-cluster-authentication-optional)) |
| `rayAuth.token` / `rayAuth.existingSecret` | `""` | Auth token inline, or reference an existing Secret (`tokenKey`). Required when `rayAuth.enabled` |
| `service.type` | `ClusterIP` | Set `LoadBalancer` to expose externally |
| `podMonitor.enabled` | `false` | Prometheus Operator scraping |
| `prometheusRule.enabled` | `false` | Ship the modelship alert rules as a PrometheusRule |
| `grafanaDashboard.enabled` | `false` | Ship the Grafana dashboard as a sidecar-imported ConfigMap |
| `kuberay-operator.enabled` | `false` | Bootstrap the operator as a subchart |

See [values.yaml](values.yaml) for the full set with inline documentation.
