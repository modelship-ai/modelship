#!/usr/bin/env bash
# A/B benchmark: a modelship loader vs the vanilla server it wraps, same
# image/config. --loader picks vllm or llama_server; --device picks cpu/gpu.
#
# Usage: bench/run.sh [--loader vllm|llama_server] [--device gpu|cpu] [--image TAG]
#                      [--config PATH] [--num-prompts N] [--concurrency N]
#                      [--input-len N] [--output-len N] [--num-warmups N] [--repeats N]
#                      [--preflight on|off] [--gpu-device ID[,ID...]]
#                      [--api-port N] [--metrics-port N]
set -euo pipefail

LOADER="vllm"
DEVICE="gpu"
IMAGE=""
CONFIG=""
TOKENIZER=""
NUM_PROMPTS=100
CONCURRENCY=8
INPUT_LEN=128
OUTPUT_LEN=512
# Requests sent and discarded before timing, to avoid a cold-start tail
# (CUDA graph capture, JIT) skewing the timed sweep.
NUM_WARMUPS=20
# Timed sweeps per stack; we report the median so one noisy run can't dominate.
REPEATS=3
READY_TIMEOUT=900
# On: both arms run preflight. Off: both fall back to loader/pydantic defaults.
PREFLIGHT="on"
# One device id, or a comma-separated list. Pins both arms to the same physical
# GPU, which a host with unlike GPUs needs.
GPU_DEVICE="0"
# Host-side ports the harness polls on. The load client shares the server arm's
# bridge network instead, so these only have to be free.
API_PORT=18000
METRICS_PORT=18079

while [[ $# -gt 0 ]]; do
    case "$1" in
        --loader) LOADER="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --num-warmups) NUM_WARMUPS="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --preflight) PREFLIGHT="$2"; shift 2 ;;
        --gpu-device) GPU_DEVICE="$2"; shift 2 ;;
        --api-port) API_PORT="$2"; shift 2 ;;
        --metrics-port) METRICS_PORT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

case "$LOADER" in
    vllm|llama_server) ;;
    *) echo "--loader must be vllm or llama_server, got: $LOADER" >&2; exit 2 ;;
esac
case "$DEVICE" in
    gpu|cpu) ;;
    *) echo "--device must be gpu or cpu, got: $DEVICE" >&2; exit 2 ;;
esac
case "$PREFLIGHT" in
    on|off) ;;
    *) echo "--preflight must be on or off, got: $PREFLIGHT" >&2; exit 2 ;;
esac
MSHIP_PREFLIGHT_ENV="true"
[[ "$PREFLIGHT" == "off" ]] && MSHIP_PREFLIGHT_ENV="false"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO_ROOT/bench"
# shellcheck source=bench/lib.sh
source "$BENCH_DIR/lib.sh"

# cuda/cpu are separate image variants; published cpu images use a "-cpu" tag
# suffix. Both arms need the prod target — the only one carrying llama-server.
if [[ -z "$IMAGE" ]]; then
    IMAGE="modelship:bench-cuda"
    [[ "$DEVICE" == "cpu" ]] && IMAGE="modelship:bench-cpu"
fi

CONFIG_PREFIX="vllm"
BASELINE_ENTRYPOINT="rawvllm_entrypoint.py"
BASELINE_LABEL="raw vllm"
if [[ "$LOADER" == "llama_server" ]]; then
    CONFIG_PREFIX="llama"
    BASELINE_ENTRYPOINT="rawllama_entrypoint.py"
    BASELINE_LABEL="vanilla llama-server"
fi
[[ -n "$CONFIG" ]] || CONFIG="$BENCH_DIR/configs/${CONFIG_PREFIX}-${DEVICE}.yaml"
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 2; }

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="$BENCH_DIR/results/$TS"
mkdir -p "$RESULTS_DIR"

CACHE_DIR="${MSHIP_CACHE_DIR:-$REPO_ROOT/models-cache}"
mkdir -p "$CACHE_DIR"

SERVED_NAME="$(yaml_scalar '^[[:space:]]*-[[:space:]]*name:' "$CONFIG")"
MODEL_ID="$(yaml_scalar '^[[:space:]]*model:' "$CONFIG")"
[[ -n "$MODEL_ID" && -n "$SERVED_NAME" ]] || { echo "failed to parse $CONFIG" >&2; exit 2; }

if [[ -z "$TOKENIZER" ]]; then
    TOKENIZER="$(yaml_scalar '^[[:space:]]*#[[:space:]]*bench-tokenizer:' "$CONFIG")"
fi
if [[ -z "$TOKENIZER" ]]; then
    TOKENIZER="$MODEL_ID"
fi

NUM_CPUS="$(yaml_scalar '^[[:space:]]*num_cpus:' "$CONFIG")"
BASELINE_ENV_ARGS=()
if [[ -n "${NUM_CPUS:-}" ]]; then
    BASELINE_ENV_ARGS+=(-e "OMP_NUM_THREADS=$NUM_CPUS")
fi

MODELSHIP_CONTAINER=bench-modelship
BASELINE_CONTAINER=bench-baseline
# Not --network host: an arm binds Ray's GCS, dashboard and proxy on fixed
# ports. Also carries the load client, on the same bridge as the server.
BENCH_NET=bench-net
# The gateway mounts under a slug of its name; same regex as
# serve_utils.gateway_route_prefix.
GATEWAY_NAME="${MSHIP_GATEWAY_NAME:-modelship}"
GATEWAY_PREFIX="$(python3 -c \
    'import re, sys; print("/" + re.sub(r"[^a-z0-9_-]+", "-", sys.argv[1].lower()).strip("-"))' \
    "$GATEWAY_NAME")"
[[ "$GATEWAY_PREFIX" != "/" ]] || { echo "MSHIP_GATEWAY_NAME=$GATEWAY_NAME has no URL-safe characters" >&2; exit 2; }
trap cleanup EXIT

# Defensive: remove any pre-existing bench containers from a prior aborted run.
docker rm -f "$MODELSHIP_CONTAINER" "$BASELINE_CONTAINER" >/dev/null 2>&1 || true
docker network inspect "$BENCH_NET" >/dev/null 2>&1 || docker network create "$BENCH_NET" >/dev/null

DOCKER_GPU_ARGS=()
# docker reads --gpus as CSV, so a multi-device id list needs the embedded quotes.
[[ "$DEVICE" == "gpu" ]] && DOCKER_GPU_ARGS=(--gpus "\"device=$GPU_DEVICE\"")

# The image supplies the dependency set; this mounts the working tree over the
# modelship copy the image was built from, so the bench measures current source.
SITE_MODELSHIP="$(docker run --rm --entrypoint python "$IMAGE" \
    -c 'import modelship, os; print(os.path.dirname(modelship.__file__))')"
[[ -n "$SITE_MODELSHIP" ]] || { echo "could not locate the modelship package in $IMAGE" >&2; exit 2; }
SOURCE_MOUNT=(-v "$REPO_ROOT/modelship:$SITE_MODELSHIP:ro")
SOURCE_REV="$(git -C "$REPO_ROOT" describe --always --dirty 2>/dev/null || echo unknown)"

# `mship deploy` resolves this itself; the baseline runs python directly. Read
# out of the image so a llama.cpp tag bump needs no edit.
LLAMA_SERVER_BIN=""
if [[ "$LOADER" == "llama_server" ]]; then
    LLAMA_SERVER_BIN="$(docker run --rm --entrypoint bash "$IMAGE" \
        -c 'find /opt/mship/builds -name llama-server.sh 2>/dev/null | head -1')"
    [[ -n "$LLAMA_SERVER_BIN" ]] || { echo "no llama-server.sh in $IMAGE" >&2; exit 2; }
    BASELINE_ENV_ARGS+=(-e "MSHIP_LLAMA_SERVER_BIN=$LLAMA_SERVER_BIN")
fi

start_modelship() {
    docker run -d "${DOCKER_GPU_ARGS[@]}" --ipc=host \
        --network "$BENCH_NET" -p "$API_PORT:8000" -p "$METRICS_PORT:8079" \
        -e MSHIP_METRICS=true \
        -e MSHIP_PREFLIGHT="$MSHIP_PREFLIGHT_ENV" \
        -e MSHIP_GATEWAY_NAME="$GATEWAY_NAME" \
        -e MSHIP_GATEWAY_REPLICAS="${MSHIP_GATEWAY_REPLICAS:-1}" \
        -e MSHIP_GATEWAY_MAX_ONGOING="${MSHIP_GATEWAY_MAX_ONGOING:-1024}" \
        -v "$CONFIG:/modelship/config/models.yaml:ro" \
        -v "$CACHE_DIR:/.cache:rw" \
        "${SOURCE_MOUNT[@]}" \
        --name "$MODELSHIP_CONTAINER" "$IMAGE" \
        deploy --config /modelship/config/models.yaml >/dev/null
}

start_baseline() {
    # Same image and entrypoint as the modelship arm (same uid), but python
    # instead of `mship`; modelship imports from the engine venv on PATH.
    docker run -d "${DOCKER_GPU_ARGS[@]}" --ipc=host \
        --network "$BENCH_NET" -p "$API_PORT:8000" \
        -e MSHIP_PREFLIGHT="$MSHIP_PREFLIGHT_ENV" \
        "${BASELINE_ENV_ARGS[@]}" \
        -v "$CONFIG:/modelship/config/models.yaml:ro" \
        -v "$BENCH_DIR/$BASELINE_ENTRYPOINT:/modelship/bench/$BASELINE_ENTRYPOINT:ro" \
        -v "$CACHE_DIR:/.cache:rw" \
        "${SOURCE_MOUNT[@]}" \
        --entrypoint /modelship/scripts/entrypoint.sh \
        --name "$BASELINE_CONTAINER" "$IMAGE" \
        python "/modelship/bench/$BASELINE_ENTRYPOINT" >/dev/null
}

echo "=== bench $TS — loader=$LOADER device=$DEVICE image=$IMAGE config=$(basename "$CONFIG") prompts=$NUM_PROMPTS conc=$CONCURRENCY in=$INPUT_LEN out=$OUTPUT_LEN warmups=$NUM_WARMUPS repeats=$REPEATS preflight=$PREFLIGHT source=$SOURCE_REV ==="

# Phase A — modelship
echo "[A] starting modelship..."
start_modelship
wait_ready "$MODELSHIP_CONTAINER"
echo "[A] running $REPEATS sweep(s)..."
warm_model_cache
mkdir -p "$RESULTS_DIR/modelship"
start_mem_sampler modelship "$MODELSHIP_CONTAINER"
start_component_sampler "$RESULTS_DIR/modelship/components.txt"
run_stack modelship
stop_mem_sampler
stop_component_sampler
scrape_prom "$RESULTS_DIR/modelship/prom.txt"
docker logs "$MODELSHIP_CONTAINER" > "$RESULTS_DIR/${MODELSHIP_CONTAINER}.log" 2>&1 || true
docker rm -f "$MODELSHIP_CONTAINER" >/dev/null
vram_gate

# Phase B — baseline (vanilla vllm or vanilla llama-server, same image/config)
pin_baseline_engine_args
echo "[B] starting baseline ($BASELINE_LABEL)..."
start_baseline
wait_ready "$BASELINE_CONTAINER"
echo "[B] running $REPEATS sweep(s)..."
warm_model_cache
mkdir -p "$RESULTS_DIR/baseline"
start_mem_sampler baseline "$BASELINE_CONTAINER"
run_stack baseline
stop_mem_sampler
docker logs "$BASELINE_CONTAINER" > "$RESULTS_DIR/${BASELINE_CONTAINER}.log" 2>&1 || true
docker rm -f "$BASELINE_CONTAINER" >/dev/null

# Fail before summarizing if the two arms weren't launched with identical
# engine args — nothing downstream is meaningful otherwise.
assert_launch_parity

# Summary
SUMMARY="$RESULTS_DIR/summary.md"
{
    echo "# bench $TS — $LOADER / $DEVICE"
    echo
    echo "image: \`$IMAGE\`  config: \`$(basename "$CONFIG")\`  prompts: $NUM_PROMPTS  concurrency: $CONCURRENCY  input/output: $INPUT_LEN/$OUTPUT_LEN  warmups: $NUM_WARMUPS  repeats: $REPEATS  preflight: $PREFLIGHT  source: \`$SOURCE_REV\`"
    echo
    echo "Values are the median across \`repeats\` sweeps."
    echo
    write_summary modelship baseline "$BASELINE_LABEL"
} | tee "$SUMMARY"

echo
echo "results: $RESULTS_DIR"

# Runs last so the summary above is always written first, then fails if
# either arm dropped requests.
assert_result_parity
