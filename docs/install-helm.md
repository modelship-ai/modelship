# Helm / Kubernetes install

The chart brings up a **RayCluster** (one CPU-only head plus worker groups) and a
**RayJob** that runs `mship deploy` **on** the cluster — KubeRay's supported way to
run a driver against a RayCluster — deploying the models declared in your
`models.yaml`.

The [chart README](https://github.com/modelship-ai/modelship/tree/main/helm/modelship)
is the full values reference; this page covers getting one installed.

## Prerequisites

- A Kubernetes cluster. A local [kind](https://kind.sigs.k8s.io/) cluster works for
  the CPU image; GPU models need real GPU nodes with the NVIDIA device plugin.
- **The KubeRay operator + CRDs, 1.6 or newer** (tested on 1.7.1) — cluster-scoped
  and install-once. Either install it yourself:

  ```bash
  helm repo add kuberay https://ray-project.github.io/kuberay-helm/
  helm install kuberay-operator kuberay/kuberay-operator --version 1.7.1
  ```

  …or, on a single-tenant cluster, let the chart bootstrap it with
  `--set kuberay-operator.enabled=true`. Upgrading an existing operator needs its
  new CRDs applied first; see the chart README.
- A **Redis** instance. The chart backs the head's GCS with it for fault tolerance
  and uses it as the state store, so the gateway self-heals after a head restart.
- For GPU models: a node pool advertising `nvidia.com/gpu`.

## Install

The chart version is kept in lockstep with the app and image version, so
`--version X.Y.Z` always pairs with the matching image:

```bash
helm install modelship \
  oci://ghcr.io/modelship-ai/charts/modelship \
  --version <X.Y.Z> \
  -f my-values.yaml
```

Or from a checkout, after fetching the vendored operator subchart with
`helm dependency build ./helm/modelship` (it needs the `kuberay` repo added, as
above): `helm install modelship ./helm/modelship -f my-values.yaml`.

Images and model weights take time to pull, so raise Helm's timeout:
`--timeout 20m --wait`. Note `--wait` does **not** track the RayJob to completion —
watch `kubectl get rayjob` and the gateway's `/readyz` for readiness.

## Configure your models

Set `models.config` to your `models.yaml` contents; the deploy RayJob carries it:

```yaml
models:
  config: |
    models:
      - name: qwen
        loader: vllm
        model: Qwen/Qwen2.5-7B-Instruct

redis:
  address: redis-master.default.svc.cluster.local:6379
```

`helm upgrade` re-applies the config additively, or reconciles the cluster to match
it exactly with `deploy.reconcile=true`.

## Image variant

The chart selects the image variant for you from the worker group's resources; see
`image.variant` in the chart README to pin it. The same three images the
[Docker install](install-docker.md) describes are used here.

## Multi-node

This is the supported path for multi-node clusters. Every pod runs the same
`X.Y.Z` image tag — Ray refuses to form a cluster across mismatched versions, so
never point a chart at a floating `:latest` tag.
