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
Name of the Secret holding the Ray auth token under key `auth_token` (existing or
the chart's own).
*/}}
{{- define "modelship.rayAuthSecretName" -}}
{{- .Values.rayAuth.existingSecret | default (printf "%s-ray-auth" (include "modelship.fullname" .)) -}}
{{- end -}}

{{/*
MSHIP_NODE_NUM_CPUS/MSHIP_NODE_MEMORY for a worker, from its container's limits (else
requests) via the downward API. Skips a var the group's own env sets. An explicit
`divisor` survives KubeRay's write-back unchanged.
Call with (dict "resources" <resources> "env" <group env> "container" <container name>).
*/}}
{{- define "modelship.nodeResourceEnv" -}}
{{- $resources := .resources | default dict -}}
{{- $limits := $resources.limits | default dict -}}
{{- $requests := $resources.requests | default dict -}}
{{- $names := list -}}
{{- range .env }}{{- $names = append $names .name -}}{{- end -}}
{{- range $var, $res := dict "MSHIP_NODE_NUM_CPUS" "cpu" "MSHIP_NODE_MEMORY" "memory" }}
{{- if and (or (get $limits $res) (get $requests $res)) (not (has $var $names)) }}
- name: {{ $var }}
  valueFrom:
    resourceFieldRef:
      containerName: {{ $.container }}
      resource: {{ ternary "limits" "requests" (hasKey $limits $res) }}.{{ $res }}
      divisor: "1"
{{- end }}
{{- end }}
{{- end -}}

{{/*
Explicit env for every Ray pod (head + workers): the state-store URI the
coordinator, effective-config and /v1/responses read via get_state_store(). It MUST
be on every pod so the coordinator — scheduled on any node — agrees with the driver.

Always redis://<addr>/<db>, password-free: the driver forwards this URI in runtime_env,
and each pod adds MSHIP_REDIS_PASSWORD from its own env (the Secret). The same Redis also
backs GCS fault tolerance. The chart wires an address but does not deploy Redis, so
redis.address is required.
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
