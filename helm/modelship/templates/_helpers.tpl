{{/*
Chart name, optionally overridden by nameOverride.
*/}}
{{- define "modelship.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully qualified app name. Honors fullnameOverride; otherwise release-name based.
*/}}
{{- define "modelship.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "modelship.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/name: {{ include "modelship.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: modelship
{{- end -}}

{{/*
Selector labels.
*/}}
{{- define "modelship.selectorLabels" -}}
app.kubernetes.io/name: {{ include "modelship.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Effective image tag, resolving the gpu/cpu/thin variant onto the base tag. `variant:
cpu` appends "-cpu"; an explicit `tag`/`variant` (e.g. a per-worker-group override)
wins over the cluster-wide image values. `isHead: true` defaults to `thin` (via
head.image.variant, itself defaulting to "thin") instead of the cluster-wide
`image.variant` — the head never runs models, so it doesn't need torch/vllm. Call
with a dict: (dict "root" $ "isHead" true) or
(dict "root" $ "tag" $img.tag "variant" $img.variant).
*/}}
{{- define "modelship.imageTag" -}}
{{- $tag := .tag | default .root.Values.image.tag -}}
{{- $variant := .variant -}}
{{- if not $variant -}}
{{- if .isHead -}}
{{- $variant = .root.Values.head.image.variant | default "thin" -}}
{{- else -}}
{{- $variant = .root.Values.image.variant | default "cuda" -}}
{{- end -}}
{{- end -}}
{{- if eq $variant "cpu" -}}{{ printf "%s-cpu" $tag }}{{- else if eq $variant "cuda" -}}{{ printf "%s-cuda" $tag }}{{- else -}}{{ $tag }}{{- end -}}
{{- end -}}

{{/*
The container image reference shared by the Ray head and the RayJob submitter —
both are coordination-only (no models scheduled there), so both default to `thin`.
*/}}
{{- define "modelship.image" -}}
{{- printf "%s:%s" .Values.image.repository (include "modelship.imageTag" (dict "root" . "isHead" true)) -}}
{{- end -}}

{{/*
Name of the ConfigMap holding models.yaml (existing or chart-templated).
*/}}
{{- define "modelship.configMapName" -}}
{{- if .Values.models.existingConfigMap -}}
{{- .Values.models.existingConfigMap -}}
{{- else -}}
{{- printf "%s-models" (include "modelship.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
Name of the Secret holding the HF token / API keys (existing or templated).
*/}}
{{- define "modelship.secretName" -}}
{{- if .Values.secrets.existingSecret -}}
{{- .Values.secrets.existingSecret -}}
{{- else -}}
{{- printf "%s-secrets" (include "modelship.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
Name of the cache PVC (existing or chart-templated).
*/}}
{{- define "modelship.cacheClaimName" -}}
{{- if .Values.cache.existingClaim -}}
{{- .Values.cache.existingClaim -}}
{{- else -}}
{{- printf "%s-cache" (include "modelship.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
KubeRay names the cluster's head Service "<raycluster-name>-head-svc"; the
RayCluster object itself is named by modelship.fullname.
*/}}
{{- define "modelship.headServiceName" -}}
{{- printf "%s-head-svc" (include "modelship.fullname" .) -}}
{{- end -}}

{{/*
envFrom for the HF token / API keys Secret. optional:true so pods start fine
when no Secret was created (e.g. all-ungated models, no auth).
*/}}
{{- define "modelship.envFrom" -}}
- secretRef:
    name: {{ include "modelship.secretName" . }}
    optional: true
{{- end -}}

{{/*
Name of the Secret holding the Redis password (existing or the chart's own).
*/}}
{{- define "modelship.redisSecretName" -}}
{{- .Values.redis.existingSecret | default (include "modelship.secretName" .) -}}
{{- end -}}

{{/*
Name of the Secret holding the Ray auth token (existing or the chart's own).
*/}}
{{- define "modelship.rayAuthSecretName" -}}
{{- .Values.rayAuth.existingSecret | default (include "modelship.secretName" .) -}}
{{- end -}}

{{/*
Env for Ray cluster authentication (RAY_AUTH_MODE=token + RAY_AUTH_TOKEN). Included
identically on the head, every worker group, AND the RayJob submitter pod — all
three must present the same token or cluster-internal RPC / `ray job submit`
fails (see the auth half-match finding in docs/multi-node-docker.md). Empty (no
env at all) when disabled, matching Ray's own unauthenticated-by-default posture.
*/}}
{{- define "modelship.rayAuthEnv" -}}
{{- if .Values.rayAuth.enabled }}
{{- if not (or .Values.rayAuth.token .Values.rayAuth.existingSecret) }}
{{ fail "rayAuth.enabled is true but neither rayAuth.token nor rayAuth.existingSecret is set — Ray token auth needs an actual token (see the chart README)." }}
{{- end }}
- name: RAY_AUTH_MODE
  value: "token"
- name: RAY_AUTH_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ include "modelship.rayAuthSecretName" . }}
      key: {{ .Values.rayAuth.tokenKey }}
{{- end }}
{{- end -}}

{{/*
Capability resources for an image variant — mirrors modelship/deploy/capabilities.py's
ALL_CAPABILITY_LOADERS (kept in lockstep by a CI check comparing this against that table).
The chart can't probe like the Python side does (KubeRay starts the raylet, so no
modelship code runs first); it doesn't need to, since the variant already determines
what's installed. Call with a variant string, e.g. (include "modelship.capabilityResources" "cuda").
*/}}
{{- define "modelship.capabilityResources" -}}
{{- if eq . "cuda" -}}
{"mship_vllm": 1, "mship_diffusers": 1, "mship_llama_server": 1, "mship_stable_diffusion_cpp": 1, "mship_whispercpp": 1, "mship_sherpa_onnx": 1}
{{- else if eq . "cpu" -}}
{"mship_vllm": 1, "mship_llama_server": 1, "mship_stable_diffusion_cpp": 1, "mship_whispercpp": 1, "mship_sherpa_onnx": 1}
{{- else -}}
{}
{{- end -}}
{{- end -}}

{{/*
Full rayStartParams for a Ray node: chart-managed defaults plus the caller's
overrides (overrides win). metrics-export-port is pinned to metrics.port on every
node so Ray's Prometheus endpoint matches the `metrics` containerPort + PodMonitor
(unset, ray start binds a random port and scrapes fail). The head is additionally
pinned to num-gpus/num-cpus 0 (it's coordination-only) and binds the dashboard on
all interfaces; workers instead render `resources` from their resolved variant, so
a model can't schedule onto a node missing its loader's backend.
Call with (dict "root" $ "isHead" <bool> "params" <rayStartParams> "variant" <variant, workers only>).
*/}}
{{- define "modelship.rayStartParams" -}}
{{- $defaults := dict "metrics-export-port" (.root.Values.metrics.port | toString) -}}
{{- if .isHead -}}
{{- $_ := set $defaults "num-gpus" "0" -}}
{{- $_ := set $defaults "num-cpus" "0" -}}
{{- $_ := set $defaults "dashboard-host" "0.0.0.0" -}}
{{- else -}}
{{- $capabilities := include "modelship.capabilityResources" .variant | fromJson | toJson -}}
{{- $escaped := $capabilities | replace "\"" "\\\"" -}}
{{- $_ := set $defaults "resources" (printf "\"%s\"" $escaped) -}}
{{- end -}}
{{- merge (deepCopy (.params | default dict)) $defaults | toYaml -}}
{{- end -}}

{{/*
Explicit env for every Ray pod (head + workers): the state-store URI the
coordinator, effective-config and /v1/responses read via get_state_store(). It MUST
be on every pod so the coordinator — scheduled on any node — agrees with the driver.

Always redis://<addr>/<db>, password-free: the driver forwards this URI in runtime_env,
so the password rides as a ${MSHIP_REDIS_PASSWORD} placeholder each pod fills from its
own env (the Secret). The same Redis also backs GCS fault tolerance. The chart wires an
address but does not deploy Redis, so redis.address is required.
*/}}
{{- define "modelship.env" -}}
{{- $addr := required "redis.address is required: modelship on k8s stores its effective config, routing registry and /v1/responses conversations in Redis. Point redis.address at a Redis instance (see the chart README)." .Values.redis.address }}
{{- if or .Values.redis.password .Values.redis.existingSecret }}
- name: MSHIP_REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "modelship.redisSecretName" . }}
      key: {{ .Values.redis.passwordKey }}
{{- end }}
- name: MSHIP_STATE_STORE
  value: "redis://{{ $addr }}/{{ .Values.redis.db }}"
{{- end -}}

{{/*
Volumes shared by every Ray pod (head + workers): an in-memory /dev/shm for
vLLM/NCCL, a per-pod node-local compile cache, and the model-weight cache PVC.
*/}}
{{- define "modelship.volumes" -}}
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: {{ .Values.shm.sizeLimit }}
- name: node-cache
{{- if .Values.nodeCache.sizeLimit }}
  emptyDir:
    sizeLimit: {{ .Values.nodeCache.sizeLimit }}
{{- else }}
  emptyDir: {}
{{- end }}
{{- if .Values.cache.enabled }}
- name: cache
  persistentVolumeClaim:
    claimName: {{ include "modelship.cacheClaimName" . }}
{{- end }}
{{- end -}}

{{/*
Matching volumeMounts for the volumes above. node-cache's mountPath must match
the image's MSHIP_NODE_CACHE_DIR.
*/}}
{{- define "modelship.volumeMounts" -}}
- name: dshm
  mountPath: /dev/shm
- name: node-cache
  mountPath: /opt/mship/node-cache
{{- if .Values.cache.enabled }}
- name: cache
  mountPath: {{ .Values.cache.mountPath }}
{{- end }}
{{- end -}}
