# Shared helpers for the bench/run-*.sh A/B scripts. Sourced, not executed —
# assumes the caller has already set REPO_ROOT, BENCH_DIR, RESULTS_DIR,
# CACHE_DIR, and the sampler/cleanup PID vars it declares below.

# Extracts the first scalar matching a key regex from a bench config yaml.
# Handles quoted/unquoted values and trims whitespace.
yaml_scalar() {
    local pattern="$1" file="$2"
    grep -m1 -E "$pattern" "$file" \
        | sed -E "s/^[^:]*:[[:space:]]*//; s/[[:space:]]*\$//; s/^(['\"])(.*)\1\$/\2/" \
        || true
}

cleanup() {
    for pid in "${MEM_SAMPLER_PID:-}" "${COMPONENT_SAMPLER_PID:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    for c in "$MODELSHIP_CONTAINER" "$BASELINE_CONTAINER"; do
        if [[ -n "$c" ]] && docker inspect "$c" >/dev/null 2>&1; then
            docker logs "$c" >"$RESULTS_DIR/${c}.log" 2>&1 || true
            docker rm -f "$c" >/dev/null 2>&1 || true
        fi
    done
    if [[ -n "${BENCH_NET:-}" ]]; then
        docker network rm "$BENCH_NET" >/dev/null 2>&1 || true
    fi
}

# modelship answers under the gateway's route prefix; the vanilla server it
# wraps answers at the root.
arm_base_url() {
    local host="$1" port="$2" container="$3"
    if [[ "$container" == "$MODELSHIP_CONTAINER" ]]; then
        printf 'http://%s:%s%s' "$host" "$port" "$GATEWAY_PREFIX"
    else
        printf 'http://%s:%s' "$host" "$port"
    fi
}

wait_ready() {
    local name="$1"
    local deadline=$(( $(date +%s) + READY_TIMEOUT ))
    while (( $(date +%s) < deadline )); do
        # /v1/models reachable AND lists the served model id
        local response
        if response=$(curl -fsS "$(arm_base_url localhost "$API_PORT" "$name")/v1/models" 2>/dev/null); then
            if python3 -c "import sys, json; data = json.loads(sys.argv[1]); print('match' if any(m.get('id') == sys.argv[2] for m in data.get('data', [])) else '')" "$response" "$SERVED_NAME" | grep -q "match"; then
                return 0
            fi
        fi
        if ! docker ps --filter "name=^${name}$" --format '{{.Names}}' | grep -q "$name"; then
            echo "container $name died" >&2
            docker logs --tail 80 "$name" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "timeout waiting for $name to be ready (served=$SERVED_NAME)" >&2
    docker logs --tail 80 "$name" >&2 || true
    return 1
}

# Used VRAM summed over the GPUs the run is pinned to ($GPU_DEVICE, the same
# id list handed to docker --gpus). Empty when nvidia-smi is absent.
gpu_used_mib() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        -i "${GPU_DEVICE:-0}" 2>/dev/null \
        | awk '{t+=$1} END {if (NR) printf "%d", t}'
}

start_mem_sampler() {
    local stack="$1"
    local container="$2"
    local out="$RESULTS_DIR/$stack/mem.tsv"
    : > "$out"
    (
        while :; do
            local ts vram cmem cgstat amib fmib smib
            ts=$(date +%s)
            # || true: avoid aborting this subshell under pipefail+set -e.
            # No nvidia-smi (CPU host) just leaves vram at 0.
            vram=$(gpu_used_mib) || true
            # docker stats MemUsage is "1.234GiB / 64GiB" — take the first field
            # and normalize any unit (binary or decimal, any case) to MiB.
            cmem=$(docker stats --no-stream --format '{{.MemUsage}}' "$container" 2>/dev/null \
                | awk -F'/' '{print $1}' \
                | awk '{
                    s=$0; gsub(/[[:space:]]/,"",s);   # e.g. "1.234GiB"
                    num=s; unit=s;
                    sub(/[A-Za-z]+$/,"",num);         # numeric part
                    sub(/^[0-9.]+/,"",unit);          # unit part
                    U=toupper(unit);
                    base=(U ~ /I/)?1024:1000;         # *iB binary, *B decimal
                    p=substr(U,1,1);
                    if      (p=="T") mib=num*base*base*base*base/1048576;
                    else if (p=="G") mib=num*base*base*base/1048576;
                    else if (p=="M") mib=num*base*base/1048576;
                    else if (p=="K") mib=num*base/1048576;
                    else if (U=="B") mib=num/1048576;
                    else             mib=num;         # unknown/unitless: assume MiB
                    printf "%.1f", mib
                  }') || true
            # cgroup memory.stat (bytes→MiB), for both stacks: anon = process RSS,
            # shmem = tmpfs/plasma (charged to the cgroup, not any process), file =
            # reclaimable page cache. cgroup v2 path; v1/missing reads as zeros.
            cgstat=$(docker exec "$container" cat /sys/fs/cgroup/memory.stat 2>/dev/null) || true
            read -r amib fmib smib < <(printf '%s\n' "$cgstat" | awk '
                $1=="anon"{a=$2} $1=="file"{f=$2} $1=="shmem"{s=$2}
                END {printf "%.1f %.1f %.1f", a/1048576, f/1048576, s/1048576}') || true
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${ts:-0}" "${vram:-0}" "${cmem:-0}" "${amib:-0}" "${fmib:-0}" "${smib:-0}" >> "$out"
            sleep 1
        done
    ) &
    MEM_SAMPLER_PID=$!
}

stop_mem_sampler() {
    if [[ -n "${MEM_SAMPLER_PID:-}" ]] && kill -0 "$MEM_SAMPLER_PID" 2>/dev/null; then
        kill "$MEM_SAMPLER_PID" 2>/dev/null || true
        wait "$MEM_SAMPLER_PID" 2>/dev/null || true
    fi
    MEM_SAMPLER_PID=""
}

# Samples the Ray reporter's per-component memory (port 8079) during the sweep,
# keeping the highest-total-USS scrape so the breakdown reflects peak load, not
# idle. modelship-only; loader-agnostic (Ray Serve's own metrics).
start_component_sampler() {
    local out="$1"
    : > "$out"
    (
        local best=-1 comp score
        while :; do
            # || true: a failed scrape under pipefail+set -e must not kill the loop.
            comp=$(curl -fsS "http://localhost:$METRICS_PORT/metrics" 2>/dev/null \
                | awk '/^ray_component_(uss_mb|rss_mb|mem_shared_bytes)[{ ]/') || true
            if [[ -n "$comp" ]]; then
                # Score = total private (USS); $NF is robust to spaces in
                # Component label values.
                score=$(printf '%s\n' "$comp" \
                    | awk '/^ray_component_uss_mb[{ ]/ {s+=$NF} END {printf "%.0f", s+0}')
                if [[ -n "$score" ]] && (( score > best )); then
                    best=$score
                    printf '%s\n' "$comp" > "$out"
                fi
            fi
            sleep 2
        done
    ) &
    COMPONENT_SAMPLER_PID=$!
}

stop_component_sampler() {
    if [[ -n "${COMPONENT_SAMPLER_PID:-}" ]] && kill -0 "$COMPONENT_SAMPLER_PID" 2>/dev/null; then
        kill "$COMPONENT_SAMPLER_PID" 2>/dev/null || true
        wait "$COMPONENT_SAMPLER_PID" 2>/dev/null || true
    fi
    COMPONENT_SAMPLER_PID=""
}

# No-op (returns immediately) on a host with no nvidia-smi — there is no VRAM
# to wait on between phases on a CPU-only run.
vram_gate() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    # The floor is per selected GPU, since the reading is summed over them.
    local ngpus threshold
    ngpus=$(awk -F, '{print NF}' <<< "${GPU_DEVICE:-0}")
    threshold=$(( 500 * ngpus ))
    local deadline=$(( $(date +%s) + 60 ))
    while (( $(date +%s) < deadline )); do
        local used
        # "" on error/non-numeric output; || true for pipefail+set -e.
        used=$(gpu_used_mib) || true
        if [[ -n "$used" ]] && (( used < threshold )); then return 0; fi
        sleep 1
    done
    echo "warn: VRAM not freed within 60s" >&2
}

# Reads model weights into the host page cache before each phase's sweeps, so
# neither phase is unfairly cold — drop_host_caches can't drop without root, so
# without this the second phase would inherit the first's warm cache.
# Globs *.gguf; -L is required since huggingface_hub exposes the weights only
# via a snapshots/ symlink to a hash-named blob, which plain find would skip.
warm_model_cache() {
    echo "  pre-warming model cache..."
    local found=0
    while IFS= read -r -d '' f; do
        found=1
        cat "$f" > /dev/null 2>&1 || true
    done < <(find -L "$CACHE_DIR" -type f -name '*.gguf' -print0 2>/dev/null)
    if (( found )); then
        echo "  model cache warmed."
    else
        echo "  no .gguf found under cache dir yet — skipping warm."
    fi
}

drop_host_caches() {
    echo "  dropping host page caches..."
    if { sync && echo 3 > /proc/sys/vm/drop_caches; } >/dev/null 2>&1; then
        echo "  caches dropped successfully."
    elif sudo -n sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1; then
        echo "  caches dropped successfully via sudo."
    else
        echo "  warn: failed to drop host caches (no write access and no passwordless sudo, or unsupported in this environment). Page-cache states may differ." >&2
    fi
}

# Run REPEATS timed sweeps against an already-ready stack, saving each to its
# own result_<n>.json. The summary takes the median across them.
run_stack() {
    local stack="$1"
    local out_dir="$RESULTS_DIR/$stack"
    mkdir -p "$out_dir"
    for i in $(seq 1 "$REPEATS"); do
        drop_host_caches
        echo "  sweep $i/$REPEATS ($stack)..."
        run_sweep "$stack" "result_${i}.json"
    done
}

run_sweep() {
    local stack="$1"
    local fname="$2"
    local out_dir="$RESULTS_DIR/$stack"

    local extra_client_args=()
    # Pin the client to separate cores from the server on --device cpu.
    if [[ "$DEVICE" == "cpu" ]]; then
        local num_cores
        num_cores=$(nproc)
        if (( num_cores > 2 )); then
            local c_start=$(( num_cores - 2 ))
            local c_end=$(( num_cores - 1 ))
            extra_client_args+=(--cpuset-cpus "${c_start}-${c_end}")
            echo "  pinning client container to cpuset ${c_start}-${c_end} (of ${num_cores} cores)"
        else
            extra_client_args+=(--cpuset-cpus "0")
            echo "  pinning client container to cpuset 0"
        fi
    fi

    # --temperature 0: vllm bench serve no longer forces greedy decoding, so
    # without this the two arms sample independently and results are
    # nondeterministic. Also makes a rare llama-server grammar-rejection
    # (triggered by --ignore-eos forcing past EOS) deterministic and symmetric
    # across both arms, instead of landing randomly on one.
    # Addressed by container name on the server arm's own bridge, so the load
    # never crosses a published port.
    local server_container="$MODELSHIP_CONTAINER"
    [[ "$stack" == "baseline" ]] && server_container="$BASELINE_CONTAINER"

    docker run --rm --network "$BENCH_NET" --user "$(id -u):$(id -g)" \
        "${extra_client_args[@]}" \
        -e HF_HOME=/.cache/huggingface \
        -v "$CACHE_DIR:/.cache:rw" \
        -v "$out_dir:/out:rw" --entrypoint vllm "$IMAGE" \
        bench serve \
            --backend openai-chat \
            --base-url "$(arm_base_url "$server_container" 8000 "$server_container")" \
            --endpoint /v1/chat/completions \
            --model "$SERVED_NAME" \
            --tokenizer "$TOKENIZER" \
            --dataset-name random \
            --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$CONCURRENCY" \
            --num-warmups "$NUM_WARMUPS" \
            --ignore-eos \
            --temperature 0 \
            --percentile-metrics ttft,tpot,itl,e2el \
            --metric-percentiles 50,95,99 \
            --save-result \
            --save-detailed \
            --result-dir /out \
            --result-filename "$fname"
}

scrape_prom() {
    local out="$1"
    # Router/request histograms are cumulative counters, so one end-of-sweep
    # scrape suffices; per-component memory (a gauge) is sampled separately
    # under load, in start_component_sampler. || true: an empty scrape must
    # not abort the run under pipefail.
    curl -fsS "http://localhost:$METRICS_PORT/metrics" 2>/dev/null \
        | awk '/^ray_modelship_(request|generation)_duration_seconds_(sum|count)/ \
              || /^ray_serve_request_router_fulfillment_time_ms_(sum|count)/' \
        > "$out" || true
}

assert_launch_parity() {
    echo "=== verifying launch-args parity ==="
    python3 - "$RESULTS_DIR" "$LOADER" <<'PY'
import sys, re, ast, json, shlex, os
from pathlib import Path

root = Path(sys.argv[1])
loader = sys.argv[2]

modelship_log = root / "bench-modelship.log"
baseline_log = root / "bench-baseline.log"

if not modelship_log.exists() or not baseline_log.exists():
    sys.exit(f"Logs missing. modelship log exists: {modelship_log.exists()}, baseline log exists: {baseline_log.exists()}")

m_content = modelship_log.read_text()
b_content = baseline_log.read_text()

def normalize_path(p):
    if not p:
        return ""
    if p.startswith("/") or "/" in p:
        return os.path.basename(p)
    return p

if loader == "llama_server":
    m_match = re.search(r"llama-server launch args for '.*':\s*(\[.*\])", m_content)
    if not m_match:
        sys.exit("Could not find 'llama-server launch args for' in modelship log")
    m_args = ast.literal_eval(m_match.group(1))

    b_match = re.search(r"rawllama exec:\s*(.*)", b_content)
    if not b_match:
        sys.exit("Could not find 'rawllama exec:' in baseline log")
    b_args = shlex.split(b_match.group(1))

    # Legitimately different between the arms.
    IGNORED = {"--host", "--port", "--api-key", "--alias"}
    # Compared by basename: the same weights resolve under different roots.
    PATH_VALUED = {"-m", "--mmproj", "--chat-template-file", "--chat-template"}

    def is_flag(token):
        """`-1` is a value (llama.cpp's auto-fit), `-ngl` is a flag."""
        return token.startswith("-") and re.match(r"^-+\d", token) is None

    def normalize_llama_args(args):
        """Each flag is paired with its own value, so swapping two values
        between flags is a difference rather than the same token multiset."""
        res = list(args[1:])
        normalized = []
        i = 0
        while i < len(res):
            arg = res[i]
            has_value = i + 1 < len(res) and not is_flag(res[i + 1])
            if arg in IGNORED:
                i += 2 if has_value else 1
            elif arg in PATH_VALUED:
                normalized.append((arg, normalize_path(res[i + 1]) if has_value else ""))
                i += 2 if has_value else 1
            elif has_value:
                normalized.append((arg, res[i + 1]))
                i += 2
            else:
                normalized.append((arg, ""))
                i += 1
        return sorted(normalized)

    m_norm = normalize_llama_args(m_args)
    b_norm = normalize_llama_args(b_args)

    with open(root / "launch-parity.txt", "w") as f:
        f.write(f"Modelship normalized: {m_norm}\n")
        f.write(f"Baseline normalized:  {b_norm}\n")

    if m_norm != b_norm:
        print("LAUNCH PARITY FAILED for llama_server!", file=sys.stderr)
        for only_in, other, label in ((m_norm, b_norm, "modelship"), (b_norm, m_norm, "baseline")):
            for item in sorted(set(only_in) - set(other)):
                print(f"  only in {label}: {item[0]} {item[1]}".rstrip(), file=sys.stderr)
        sys.exit(1)
    else:
        print("LAUNCH PARITY PASSED for llama_server.")

elif loader == "vllm":
    # Both derived at the engine boundary, so modelship logs them beside the
    # kwargs dict rather than inside it.
    m_match = re.search(
        r"initialising vllm engine with args:\s*(\{.*\})\s*\(model=.*?"
        r"gpu_memory_utilization=([\d.]+),\s*max_model_len=(-?\d+)\)",
        m_content,
    )
    if not m_match:
        sys.exit("Could not find 'initialising vllm engine with args:' in modelship log")
    m_dict = ast.literal_eval(m_match.group(1))
    m_dict["gpu_memory_utilization"] = float(m_match.group(2))
    m_dict["max_model_len"] = int(m_match.group(3))

    b_match = re.search(r"rawvllm exec:\s*(.*)", b_content)
    if not b_match:
        sys.exit("Could not find 'rawvllm exec:' in baseline log")
    b_args = shlex.split(b_match.group(1))

    def coerce(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            return val

    def parse_vllm_flags(args):
        """A flag followed by another flag (or nothing) is a store_true; vLLM
        spells the negation `--no-<name>`."""
        parsed = {}
        i = 0
        while i < len(args):
            arg = args[i]
            if not arg.startswith("--"):
                i += 1
                continue
            name = arg[2:].replace("-", "_")
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                parsed[name] = coerce(args[i + 1])
                i += 2
                continue
            if name.startswith("no_"):
                parsed[name[3:]] = False
            else:
                parsed[name] = True
            i += 1
        return parsed

    b_dict = parse_vllm_flags(b_args)

    # Every engine arg VllmInfer forwards that the raw arm can also set. A key
    # missing on one side compares as None against the other's value.
    fields = [
        'gpu_memory_utilization',
        'max_model_len',
        'tensor_parallel_size',
        'pipeline_parallel_size',
        'dtype',
        'tokenizer',
        'quantization',
        'kv_cache_dtype',
        'enforce_eager',
        'trust_remote_code',
        'enable_prefix_caching',
        'max_num_batched_tokens',
        'max_num_seqs',
        'enable_auto_tool_choice',
        'tool_call_parser',
        'enable_log_requests',
        'disable_log_stats',
        'chat_template_content_format',
        'limit_mm_per_prompt',
        'mm_processor_kwargs',
    ]
    # modelship dumps these as dicts; the raw arm's flag carries JSON text.
    JSON_VALUED = {'limit_mm_per_prompt', 'mm_processor_kwargs'}

    def canon_json(val):
        if isinstance(val, str):
            val = json.loads(val)
        if not val:
            return None
        return json.dumps(val, sort_keys=True)

    m_norm = {}
    b_norm = {}

    for fld in fields:
        mv = m_dict.get(fld)
        bv = b_dict.get(fld)
        # vLLM's own default for an unset kv cache dtype.
        if fld == 'kv_cache_dtype':
            mv = mv or 'auto'
            bv = bv or 'auto'
        if fld in JSON_VALUED:
            mv = canon_json(mv)
            bv = canon_json(bv)
        if mv in [None, False]:
            mv = None
        if bv in [None, False]:
            bv = None
        if isinstance(mv, float) and isinstance(bv, float):
            if abs(mv - bv) < 1e-5:
                bv = mv
        m_norm[fld] = mv
        b_norm[fld] = bv

    with open(root / "launch-parity.txt", "w") as f:
        f.write(f"Modelship normalized: {m_norm}\n")
        f.write(f"Baseline normalized:  {b_norm}\n")

    if m_norm != b_norm:
        print("LAUNCH PARITY FAILED for vllm!", file=sys.stderr)
        for fld in fields:
            if m_norm[fld] != b_norm[fld]:
                print(f"  {fld}: modelship={m_norm[fld]!r} baseline={b_norm[fld]!r}", file=sys.stderr)
        sys.exit(1)
    else:
        print("LAUNCH PARITY PASSED for vllm.")
PY
}

# Preflight reads free RAM/VRAM, which moves between the two phases. Phase B
# replays phase A's resolved values; its flag translation still runs
# independently, so assert_launch_parity still checks it.
pin_baseline_engine_args() {
    local log="$RESULTS_DIR/${MODELSHIP_CONTAINER}.log"
    [[ -f "$log" ]] || { echo "  warn: no modelship log to pin baseline args from" >&2; return 0; }

    if [[ "$LOADER" == "vllm" ]]; then
        local gmu mml
        gmu=$(sed -n 's/.*gpu_memory_utilization=\([0-9.]*\), max_model_len=.*/\1/p' "$log" | tail -1)
        mml=$(sed -n 's/.*gpu_memory_utilization=[0-9.]*, max_model_len=\(-\?[0-9]*\)).*/\1/p' "$log" | tail -1)
        [[ -n "$gmu" ]] && BASELINE_ENV_ARGS+=(-e "BENCH_PIN_GPU_MEMORY_UTILIZATION=$gmu")
        [[ -n "$mml" ]] && BASELINE_ENV_ARGS+=(-e "BENCH_PIN_MAX_MODEL_LEN=$mml")
        echo "  baseline pinned to phase A: gpu_memory_utilization=${gmu:-unset} max_model_len=${mml:-unset}"
    else
        local args_line ctx ngl ts
        args_line=$(grep -o "llama-server launch args for .*" "$log" | tail -1)
        [[ -n "$args_line" ]] || { echo "  warn: no llama-server launch args in the modelship log" >&2; return 0; }
        ctx=$(grep -o "'-c', '[0-9-]*'" <<< "$args_line" | grep -oE -- '-?[0-9]+' | tail -1)
        ngl=$(grep -o "'-ngl', '[0-9-]*'" <<< "$args_line" | grep -oE -- '-?[0-9]+' | tail -1)
        ts=$(grep -o "'-ts', '[0-9.,]*'" <<< "$args_line" | grep -oE -- '[0-9.,]+' | tail -1)
        [[ -n "$ctx" ]] && BASELINE_ENV_ARGS+=(-e "BENCH_PIN_N_CTX_TOTAL=$ctx")
        [[ -n "$ngl" ]] && BASELINE_ENV_ARGS+=(-e "BENCH_PIN_N_GPU_LAYERS=$ngl")
        # Pinned even when empty: phase A launching without a split is itself
        # the value to replay, and fit-params re-derives one from free VRAM.
        BASELINE_ENV_ARGS+=(-e "BENCH_PIN_TENSOR_SPLIT=$ts")
        echo "  baseline pinned to phase A: -c ${ctx:-unset} -ngl ${ngl:-unset} -ts ${ts:-none}"
    fi
}

# Fails the run if one arm silently drops more requests than the other — the
# summary's medians are computed only over successful requests, so an arm
# that drops its slowest ones would look better purely by survivorship.
assert_result_parity() {
    echo "=== verifying result-population parity (header + in-band SSE errors) ==="
    python3 - "$RESULTS_DIR" "$OUTPUT_LEN" <<'PY'
import json, sys
from pathlib import Path

# Two failure modes, only one caught by the client's own accounting:
#   1. Header failure — connection errors before the response; counted in `failed`.
#   2. In-band failure — HTTP 200 already sent, then an OpenAI-style SSE error
#      chunk truncates the stream. The client's SSE parser only reads
#      choices/usage, so it silently counts this as `completed`.
# Since the sweep runs --ignore-eos, every healthy request emits exactly
# --random-output-len tokens, so a shorter completed request is a hidden
# in-band failure — the only signal that catches it.
#
# Severity is relative: both arms drive the same llama-server binary under the
# same workload, so a shared engine-level error (e.g. a grammar rejection) hits
# both and isn't a wrapping cost. modelship dropping/truncating MORE than
# baseline is a real cost (hard fail); baseline dropping more is a finding in
# modelship's favor (its front door absorbs errors baseline exposes), not a
# failure; equal counts (including zero) pass.
root = Path(sys.argv[1])
expected = int(sys.argv[2])

def scan(d):
    """Return (header_failed, hidden_inband, worst_partial_tok) for one result."""
    completed = d.get("completed", 0)
    failed = d.get("failed", 0)
    output_lens = d.get("output_lens")
    if output_lens:  # exact per-request path (needs --save-detailed)
        # Header failures already appended output_len 0, so subtract them to
        # isolate the hidden (200-with-error) failures miscounted as completed.
        short = sum(1 for ol in output_lens if ol < expected)
        hidden = max(0, short - failed)
        worst = min((ol for ol in output_lens if 0 < ol < expected), default=0)
        return failed, hidden, worst
    if completed and expected:  # aggregate fallback if arrays were stripped
        got = d.get("total_output_tokens", 0)
        full = completed * expected
        hidden = max(0, round((full - got) / expected)) if got < full else 0
        return failed, hidden, 0
    return failed, 0, 0

def summarize(stack):
    """Print per-sweep detail for one arm and return its (header, hidden) totals."""
    header_tot = hidden_tot = 0
    for p in sorted((root / stack).glob("result_*.json")):
        d = json.loads(p.read_text())
        completed = d.get("completed", 0)
        total = d.get("num_prompts", completed + d.get("failed", 0))
        failed, hidden, worst = scan(d)
        header_tot += failed
        hidden_tot += hidden
        msgs = []
        if failed:
            msgs.append(f"{failed} header FAILED / {completed} completed of {total}"
                        + ("  (cpp-httplib keep-alive resets)" if stack == "baseline" else ""))
        if hidden:
            partial = f", shortest partial {worst} tok" if worst else ""
            msgs.append(f"{hidden} HIDDEN in-band failure(s) — HTTP 200 but truncated "
                        f"below output_len={expected}{partial}, miscounted as completed")
        for m in msgs:
            print(f"  {stack}/{p.name}: {m}")
    return header_tot, hidden_tot

m_header, m_hidden = summarize("modelship")
b_header, b_hidden = summarize("baseline")
m_drops = m_header + m_hidden
b_drops = b_header + b_hidden
print(f"  totals: modelship {m_drops} dropped/truncated ({m_header} header + {m_hidden} in-band); "
      f"baseline {b_drops} ({b_header} header + {b_hidden} in-band)")

if m_drops > b_drops:
    sys.stdout.flush()  # keep the per-sweep detail (stdout) ahead of the verdict (stderr)
    print(
        f"\nRESULT PARITY FAILED: the modelship arm dropped or truncated MORE requests "
        f"than the raw baseline ({m_drops} vs {b_drops}). That excess is a cost of the "
        f"wrapper under test, and its latency/throughput medians compare unequal "
        f"populations (survivorship bias), so they are NOT trustworthy. Investigate "
        f"before trusting this run.",
        file=sys.stderr,
    )
    sys.exit(1)
if b_drops > m_drops:
    print()
    print("FINDING — baseline robustness gap (NOT a failure; run still passes):")
    print(
        f"The raw llama-server baseline dropped/truncated more requests than modelship "
        f"({b_drops} vs {m_drops}) under this load — largely the bench client hitting "
        f"cpp-httplib's keep-alive resets directly, which modelship's uvicorn front door "
        f"absorbs. This is a point in modelship's favour. Caveat: the baseline "
        f"latency/throughput medians in summary.md are computed over its surviving "
        f"requests only — read them as an upper bound on the baseline's advantage, not a "
        f"like-for-like population (see the completed/failed rows)."
        + (f" (modelship itself truncated {m_drops} request(s) — an in-band error it "
           f"shares with the baseline's engine, not a drop the baseline avoided.)"
           if m_drops else "")
    )
    sys.exit(0)
if m_drops:  # equal and non-zero
    print(
        f"\nRESULT PARITY PASSED: both arms dropped/truncated the same number of requests "
        f"({m_drops}) — shared workload/engine behaviour (e.g. llama-server rejecting "
        f"--ignore-eos-forced malformed tool calls), not a cost of the wrapper. The "
        f"populations match, so the medians are comparable."
    )
    sys.exit(0)
print("RESULT PARITY PASSED: both arms completed every request in full (no header or in-band errors).")
PY
}

# Renders summary.md: latency/throughput (median), memory (peak), modelship's
# per-component breakdown + cgroup cross-check, and router-fulfillment timing.
write_summary() {
    local modelship_stack="$1" baseline_stack="$2" baseline_label="$3"
    echo "| metric | modelship | $baseline_label | overhead |"
    echo "| --- | ---: | ---: | ---: |"
    python3 - "$RESULTS_DIR" "$modelship_stack" "$baseline_stack" <<'PY'
import json, sys, statistics
from pathlib import Path
root = Path(sys.argv[1])
def load(stack):
    # One result_<n>.json per repeat; return them all so we can take medians.
    runs = [json.loads(p.read_text()) for p in sorted((root / stack).glob("result_*.json"))]
    if not runs:
        sys.exit(f"no result_*.json found for {stack}")
    return runs
def med(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return statistics.median(vals) if vals else None
m = load(sys.argv[2]); r = load(sys.argv[3])
keys = [
    # Population first — latency/throughput below are over successful requests
    # only; assert_result_parity already guarantees failed==0 on a passing run.
    ("completed",          "completed", 0),
    ("failed",             "failed", 0),
    ("request_throughput", "req/s", 3),
    ("output_throughput",  "output tok/s", 2),
    ("mean_ttft_ms",       "TTFT mean (ms)", 1),
    ("p50_ttft_ms",        "TTFT p50 (ms)", 1),
    ("p95_ttft_ms",        "TTFT p95 (ms)", 1),
    ("mean_tpot_ms",       "TPOT mean (ms)", 1),
    ("mean_itl_ms",        "ITL mean (ms)", 2),
    ("p95_itl_ms",         "ITL p95 (ms)", 2),
    ("mean_e2el_ms",       "E2E mean (ms)", 1),
    ("p50_e2el_ms",        "E2E p50 (ms)", 1),
    ("p95_e2el_ms",        "E2E p95 (ms)", 1),
]
for key, label, prec in keys:
    mv = med(m, key); rv = med(r, key)
    if mv is None or rv is None:
        continue
    if rv == 0:
        ratio = "—"
    else:
        ratio = f"{(mv - rv) / rv * 100:+.1f}%"
    print(f"| {label} | {mv:.{prec}f} | {rv:.{prec}f} | {ratio} |")

# Token-count parity: proves both arms tokenized equivalent prompts. Launch
# args matching doesn't guarantee the prompt bodies did, and this is the only
# independent check — modelship's subprocess logs are TRACE-suppressed here.
def per_prompt_in(runs):
    vals = [rr["total_input_tokens"] / rr["completed"] for rr in runs if rr.get("completed")]
    return statistics.median(vals) if vals else None
mi = per_prompt_in(m); ri = per_prompt_in(r)
if mi is not None and ri is not None:
    delta = mi - ri
    flag = "⚠️ prompts differ" if abs(delta) > 1.0 else "✓"
    print()
    print(f"_input tokens/prompt (client tokenizer): modelship **{mi:.1f}** vs "
          f"baseline **{ri:.1f}** (Δ {delta:+.1f}) {flag}_")
PY

    echo
    echo "## memory (peak across all sweeps)"
    python3 - "$RESULTS_DIR" "$modelship_stack" "$baseline_stack" "$baseline_label" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
# mem.tsv columns: ts, vram, container_rss, anon, file(cache), shmem — all MiB
# except ts. Older runs only have the first 3; missing columns peak at 0.
COLS = ["vram", "rss", "anon", "file", "shmem"]
def peak(stack):
    f = root / stack / "mem.tsv"
    if not f.exists():
        return None
    peaks = {c: 0.0 for c in COLS}
    for line in f.read_text().splitlines():
        parts = line.split("\t")
        for i, c in enumerate(COLS, start=1):
            if i < len(parts):
                try:
                    peaks[c] = max(peaks[c], float(parts[i]))
                except ValueError:
                    pass
    return peaks
m = peak(sys.argv[2]); r = peak(sys.argv[3])
label = sys.argv[4]
print(f"| metric | modelship | {label} | overhead |")
print("| --- | ---: | ---: | ---: |")
def row(name, key, unit="MiB"):
    if m is None or r is None:
        return
    mv, rv = m[key], r[key]
    delta = mv - rv
    pct = f"{(delta / rv * 100):+.1f}%" if rv else "—"
    print(f"| {name} | {mv:.0f} {unit} | {rv:.0f} {unit} | {delta:+.0f} {unit} ({pct}) |")
row("peak VRAM (GPU0)", "vram")
row("peak container RSS", "rss")
row("  ├─ anon (process RSS)", "anon")
row("  ├─ shmem (tmpfs/plasma)", "shmem")
row("  └─ file (page cache)", "file")
print()
print("_**anon** is the real RAM overhead. **file** (page cache) is reclaimable "
      "and non-deterministic — it depends on which cgroup first faulted the weights "
      "and can swing GB between runs, so the container-RSS delta over- or "
      "under-states the true cost. Each row is an independent peak (different "
      "instants), so the sub-rows need not sum to peak RSS._")
PY
    echo
    echo "## modelship per-component memory (Ray reporter, peak under load)"
    echo
    # Splits modelship's RSS by Ray process: the model-serving actor
    # (ray::*Deployment*) vs the control plane (gcs_server/raylet/agent/
    # ProxyActor/ServeController — fixed Ray cost). USS is private memory;
    # shared is shared libraries (torch/CUDA), not plasma. Peak-under-load
    # scrape from start_component_sampler, not idle post-sweep.
    if [[ -s "$RESULTS_DIR/$modelship_stack/components.txt" ]]; then
        python3 - "$RESULTS_DIR" "$modelship_stack" <<'PY'
import sys, re
from pathlib import Path
root = Path(sys.argv[1])
pat = re.compile(r'^(ray_component_(?:uss_mb|rss_mb|mem_shared_bytes))\{([^}]*)\}\s+([0-9eE+.\-]+)')
key = {"ray_component_uss_mb": "uss", "ray_component_rss_mb": "rss", "ray_component_mem_shared_bytes": "shared"}
comp: dict[str, dict[str, float]] = {}
for line in (root / sys.argv[2] / "components.txt").read_text().splitlines():
    m = pat.match(line)
    if not m:
        continue
    metric, labels, val = m.group(1), m.group(2), float(m.group(3))
    name = dict(re.findall(r'(\w+)="([^"]*)"', labels)).get("Component", "?")
    d = comp.setdefault(name, {})
    # Ray emits rss/uss in MB (bytes/1e6); shared is raw bytes — normalize to MB.
    v = val / 1e6 if metric == "ray_component_mem_shared_bytes" else val
    d[key[metric]] = d.get(key[metric], 0.0) + v
# Private = USS when the agent could read it, else RSS - shared as a floor.
def private(d):
    return d["uss"] if "uss" in d else max(d.get("rss", 0.0) - d.get("shared", 0.0), 0.0)
rows = sorted(comp.items(), key=lambda kv: private(kv[1]), reverse=True)
print("| component | private (MB) | rss (MB) | shared (MB) |")
print("| --- | ---: | ---: | ---: |")
tot_priv = tot_rss = tot_shared = 0.0
for name, d in rows:
    p, r, s = private(d), d.get("rss", 0.0), d.get("shared", 0.0)
    tot_priv += p; tot_rss += r; tot_shared += s
    print(f"| `{name}` | {p:.0f} | {r:.0f} | {s:.0f} |")
print(f"| **total** | **{tot_priv:.0f}** | **{tot_rss:.0f}** | **{tot_shared:.0f}** |")
if not any("uss" in d for _, d in rows):
    print()
    print("_USS unavailable (reporter couldn't read smaps); private column is "
          "rss − shared, an upper bound._")

# Cross-checks the Ray reporter against cgroup memory.stat (kernel ground
# truth, peak). anon ≈ Σ private, shmem ≈ Σ shared. Sampled independently,
# so expect rough, not exact, agreement.
def cgroup_peak(col):  # mem.tsv: ts,vram,rss,anon,file,shmem (MiB)
    f = root / sys.argv[2] / "mem.tsv"
    if not f.exists():
        return None
    peak = 0.0
    for line in f.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) > col:
            try:
                peak = max(peak, float(parts[col]))
            except ValueError:
                pass
    return peak
anon = cgroup_peak(3)  # MiB ≈ MB for this sanity check
if anon:
    print()
    gap = (tot_priv - anon) / anon * 100
    flag = "  ⚠️ diverges" if abs(gap) > 25 else ""
    print("_Reporter cross-check vs cgroup `memory.stat` (kernel ground truth, peak):_")
    print(f"- private: reporter Σ USS **{tot_priv:.0f} MB** vs cgroup anon "
          f"**{anon:.0f} MB** ({gap:+.0f}%){flag}")
    if gap < -25:
        print("  - reporter undercounts — it sees only Ray worker PIDs, so memory in "
              "non-Ray child processes (notably the loader's own inference subprocess/engine) "
              "is missing. Trust the cgroup figure; treat the per-component split as "
              "relative attribution, not an absolute total.")
    # mem_shared_bytes is shared *libraries* (torch/CUDA), not plasma, so it
    # has no cgroup counterpart. Actual tmpfs/plasma is cgroup shmem (memory table).
PY
    else
        echo "_no component metrics scraped (reporter agent down or 8079 unreachable)_"
    fi
    echo
    echo "## modelship internal (Prometheus)"
    echo
    # modelship_request/generation_duration are recorded when the streaming
    # generator is *created*, not after it drains — so for streaming they only
    # capture setup/TTFT, not full request time. Not used for "gateway overhead".
    if [[ -s "$RESULTS_DIR/$modelship_stack/prom.txt" ]]; then
        python3 - "$RESULTS_DIR/$modelship_stack/prom.txt" <<'PY'
import sys, re
sums = {}; counts = {}
pat = re.compile(
    r'(ray_serve_request_router_fulfillment_time_ms)'
    r'_(sum|count)\S*\s+([0-9eE+\-.]+)'
)
for line in open(sys.argv[1]):
    m = pat.match(line)
    if not m: continue
    name, kind, val = m.group(1), m.group(2), float(m.group(3))
    (sums if kind == "sum" else counts).setdefault(name, 0.0)
    if kind == "sum": sums[name] += val
    else: counts[name] += val
n = "ray_serve_request_router_fulfillment_time_ms"
cnt = counts.get(n, 0.0)
if cnt:
    print(f"- mean router fulfillment (routing + queue wait): **{sums.get(n, 0.0) / cnt:.1f} ms** "
          f"over {cnt:.0f} routed requests")
else:
    print("- no router metrics scraped")
print()
print("_E2E / engine durations omitted: their histograms are recorded before "
      "streaming completes and are not meaningful for streaming responses._")
PY
    else
        echo "_no metrics scraped_"
    fi
}
