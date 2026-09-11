# bench — modelship vs vanilla-loader A/B harness

Two-phase benchmark that runs `vllm bench serve` against a **modelship**
loader, then against the **vanilla server it wraps** (raw `vllm serve`, or a
bare `llama-server` subprocess) from the *same Docker image* with the same
engine config, and diffs throughput, latency, and memory.

This measures modelship's own wrapping overhead — Ray Serve, the gateway, the
loader's proxy layer — not one inference stack against another. `--loader`
picks which wrapped stack to measure (`vllm` or `llama_server`); `--device`
picks GPU vs CPU for that stack. The four combinations:

| `--loader` | `--device` | config | baseline |
| --- | --- | --- | --- |
| `vllm` | `gpu` (default) | `configs/vllm-gpu.yaml` | raw `vllm serve` on GPU |
| `vllm` | `cpu` | `configs/vllm-cpu.yaml` | raw `vllm serve` on the CPU backend |
| `llama_server` | `gpu` | `configs/llama-gpu.yaml` | vanilla `llama-server`, fully GPU-offloaded |
| `llama_server` | `cpu` | `configs/llama-cpu.yaml` | vanilla `llama-server`, CPU-only |

## Prerequisites

- `docker` and `curl` on the host.
- For `--device gpu`: the NVIDIA container runtime and `nvidia-smi`.
- A built modelship image, **prod target** — the artifact users run, and the only
  one carrying the pinned `llama-server` build. Default tag is
  `modelship:bench-cuda` (CUDA) / `modelship:bench-cpu` (CPU); override with
  `--image`.

```bash
uv build -o wheels .
uv build -o wheels bootstrap
docker build -t modelship:bench-cuda \
  --target prod \
  --build-arg MSHIP_VARIANT=cuda \
  --build-arg MSHIP_VERSION="$(uv version --short)" \
  --build-context wheels=./wheels \
  .
```

The image supplies only the dependency set. Both arms mount `modelship/` from the
working tree over the copy the image was built from, so a source edit needs no
rebuild and the sweep always measures current source. Every run header and
`summary.md` records the tree it ran (`source: v0.7.14-62-gada5fe6-dirty`).

## Run

```bash
bench/run.sh                                        # vllm on GPU (default): 100 prompts, conc 8, in/out 128/512, 20 warmups, 3 repeats
bench/run.sh --loader vllm --device cpu
bench/run.sh --loader llama_server --device gpu
bench/run.sh --loader llama_server --device cpu
bench/run.sh --loader vllm --concurrency 32 --num-prompts 500
```

`--num-warmups N` (default 20) sends warmup requests that are discarded before
timing so cold-start (CUDA graph capture / compilation / first-request JIT)
doesn't skew the result. `--repeats N` (default 3) runs the sweep N times per
stack; the summary reports the **median** so a single noisy run can't dominate.
`--config PATH` overrides the config file picked by `--loader`/`--device`.

- `--preflight on|off` (default `on`) — run modelship's hardware-aware preflight
  in **both** arms, so the sweep measures the engine settings modelship ships.
  `off` falls both arms back to loader/pydantic defaults.
- `--gpu-device ID[,ID...]` (default `0`) — the physical GPU(s) both arms are
  pinned to. Required for a meaningful result on a host with unlike GPUs; pass a
  comma-separated list for a multi-slot (`tp*pp > 1`) config.
- `--api-port N` / `--metrics-port N` (default 18000/18079) — host-side ports the
  harness polls. The load client shares the server arm's bridge network, so
  these never carry benchmark traffic and only have to be free.

Tunable env vars (forwarded to the modelship phase):

- `MSHIP_GATEWAY_REPLICAS` (default 1) — gateway replica count.
- `MSHIP_GATEWAY_MAX_ONGOING` (default 1024) — gateway per-replica concurrency cap.
- `MSHIP_CACHE_DIR` — model cache to reuse across phases (default `./models-cache`).

The model and the per-model `max_ongoing_requests` cap come from the config
file (`configs/vllm-gpu.yaml`, `configs/vllm-cpu.yaml`, `configs/llama-gpu.yaml`,
or `configs/llama-cpu.yaml`).

## Output

Each run writes a timestamped dir under `bench/results/` (gitignored) containing
`result_<n>.json` (one per repeat, per phase), `mem.tsv`, `prom.txt`,
`components.txt`, container logs, and a `summary.md` whose tables show the
**median across repeats**. The two phase subdirectories are always named
`modelship` and `baseline` regardless of `--loader`/`--device`.

`summary.md` also breaks the modelship container's memory down per Ray process
(from `components.txt`, scraped from the reporter agent on port 8079): a
**per-component memory** table ranks `ray::*` model-serving actors and the
control-plane processes (`gcs_server`, `raylet`, `agent`, `ProxyActor`,
`ServeController`) by private memory (USS), with shared *libraries* (torch/CUDA,
mapped into every worker — not plasma) reported separately so they aren't charged
to any one actor. This attributes the host-RAM overhead — model-serving actor vs
fixed Ray control plane. The snapshot is the **peak-private scrape sampled during
the sweep** (not the idle post-sweep state); modelship-only, since the baseline
stack has no Ray. Note this table **undercounts** the true total: the Ray
reporter sees only Ray worker PIDs, so a loader's own inference subprocess
(vLLM's `EngineCore`, or the `llama-server` child) is missing — the
reconciliation below quantifies the gap. Trust cgroup `anon` for the absolute
number.

Two cross-checks back this up:

- **cgroup `memory.stat` breakdown** — `mem.tsv` records, per second for *both*
  stacks, the kernel's own accounting: `anon` (real process RSS), `shmem`
  (tmpfs/plasma — Ray's object store, charged to the cgroup but to no process),
  and `file` (reclaimable page cache). The memory table reports the peak of each,
  so the modelship-vs-baseline RSS gap is attributed to real memory vs plasma vs cache.
- **reporter-vs-cgroup reconciliation** — the per-component section compares the
  Ray reporter's Σ private/shared (a second-hand Prometheus gauge that can be
  stale) against cgroup `anon`/`shmem` (ground truth). A `⚠️ diverges` flag means
  the reporter numbers are suspect and shouldn't be quoted. (vLLM and llama-server
  each expose their own `/metrics` too, but only engine stats — no per-process
  memory — which is why the cgroup numbers are the only cross-stack memory signal.)

## Results

Both runs below: 1×RTX 5060 Ti (16 GB), 100 prompts @ concurrency 8, in/out
128/512, 20 warmups, median of 3, `--preflight on`, greedy (`--temperature 0`).

### vllm / GPU

Qwen2.5-7B-Instruct-AWQ, `num_gpus: 0.9`. Both arms launched with
`gpu_memory_utilization=0.9` and `max_model_len=-1` (vLLM's own auto-fit, which
settled on 32,768 tokens).

| metric | modelship | raw vllm | overhead |
| --- | ---: | ---: | ---: |
| completed / failed | 100 / 0 | 100 / 0 | — |
| throughput (req/s) | 1.213 | 1.214 | −0.1% |
| output (tok/s) | 620.91 | 621.70 | −0.1% |
| TTFT mean (ms) | 64.8 | 59.9 | +8.3% |
| TTFT p50 (ms) | 62.2 | 64.5 | −3.6% |
| TTFT p95 (ms) | 103.7 | 76.0 | +36.5% |
| TPOT mean (ms) | 12.3 | 12.3 | −0.0% |
| ITL mean (ms) | 12.66 | 12.27 | +3.2% |
| peak VRAM (MiB) | 15058 | 13294 | +1764 |
| anon / process RSS (MiB) | 5958 | 3696 | +2262 |

Notes:

- **Throughput and decode (TPOT) are at parity** — same vLLM wheel and GPU, so
  the engine's hot path is identical. modelship adds no per-token overhead.
- **TTFT** carries modelship's expected cost: the extra hop through the Ray Serve
  proxy/router adds a small *fixed* first-token latency (~5 ms). Negligible for a
  512-token response, and the p50 is inside the noise.
- **TTFT's tail is fatter than its mean** — p95 overhead (+36.5%) runs well above
  the mean (+8.3%): scheduling jitter through the proxy/router under concurrent
  load, not a fixed per-request cost. ~28 ms against a 6.3 s E2E latency.
- **Read host-RAM cost from `anon`**, not container RSS. The `file` (page cache)
  row swings multiple GB between arms depending on which cgroup faulted the
  weights first, and is not overhead.

### llama_server / GPU

Qwen2.5-7B-Instruct Q4_K_M GGUF, `num_gpus: 1`. `llama fit-params` returned
`-c 0 -ngl -1` on both arms — the whole model and its full context fit, so no
constraint is imposed.

| metric | modelship | vanilla llama-server | overhead |
| --- | ---: | ---: | ---: |
| completed / failed | **100 / 0** | 94 / 6 | — |
| throughput (req/s) | 0.524 | 0.528 | −0.6% |
| output (tok/s) | 268.51 | 270.17 | −0.6% |
| TTFT mean (ms) | 332.8 | 337.7 | **−1.5%** |
| TTFT p95 (ms) | 488.5 | 442.0 | +10.5% |
| TPOT mean (ms) | 28.5 | 28.4 | +0.4% |
| ITL mean (ms) | 28.54 | 28.51 | +0.1% |
| peak VRAM (MiB) | 6268 | 6268 | **+0** |
| anon / process RSS (MiB) | 11148 | 8457 | +2691 |

Notes:

- **Decode is at parity and VRAM is identical** — same binary, same GPU, and
  both arms fit from the same `fit-params` call, so they offload the same layers
  into the same buffers.
- **modelship completes every request; the raw baseline drops ~6%.** Vanilla
  `llama-server`'s failures are all `ServerDisconnectedError` — the bench client
  (aiohttp) hits `llama-server`'s cpp-httplib keep-alive close behaviour
  directly, a race that Ray Serve's uvicorn front door structurally absorbs. It
  is **not** tunable away via `llama-server` flags (`--threads-http` sizes the
  worker pool, not the keep-alive lifecycle). So the baseline's small throughput
  edge is partly **survivorship** — it decoded fewer requests. The
  survivorship-immune per-token metrics (TPOT/ITL) are the honest read.
- **The result-parity gate is relative between arms**: modelship dropping or
  truncating *more* than the baseline hard-fails the run; the baseline dropping
  more (as here) is reported as a **FINDING** and the run passes.
- **The load client runs greedy (`--temperature 0`).** Both arms then decode an
  identical deterministic token stream, so the A/B is reproducible and any
  *shared* engine-level in-band error appears symmetrically instead of landing on
  one arm by sampling luck.
- Numbers are illustrative; they vary with model, hardware, load, loader, and device.

## How the two phases stay comparable

- Both phases use the same image (same vLLM wheel / same `llama-server` binary),
  the same config file, and the same working-tree `modelship/` mount.
- Both phases are pinned to the same physical GPU(s) (`--gpu-device`, default 0) and
  resolve weights into the same mounted cache, so neither arm reads a different
  device or a different page cache than the arm it is compared against. The peak-VRAM
  sampler and the between-phase release gate read those same device ids, summed.
- **Both phases run preflight.** The baseline entrypoints
  ([`rawvllm_entrypoint.py`](rawvllm_entrypoint.py),
  [`rawllama_entrypoint.py`](rawllama_entrypoint.py)) run the same
  `run_preflight` → `merge_with_user_overrides` → `resolve_*` chain the loader
  actors run, then translate the result into `vllm serve` flags or a
  `llama-server` launch command. Set `--preflight off` to take both arms back to
  loader/pydantic defaults instead.
- **Phase B replays phase A's resolved engine args.** Preflight sizes itself from
  free RAM/VRAM, and the two phases deploy tens of minutes apart, so re-deriving
  can legitimately land on a different reservation — a difference that would make
  the two arms incomparable rather than reveal a defect. `pin_baseline_engine_args`
  extracts what the modelship arm actually launched with and passes it to the
  baseline, which logs both the pinned and the derived value — for vllm
  `gpu_memory_utilization` and `max_model_len`, for llama_server `-c`, `-ngl`
  and the `-ts` tensor split, all of which `fit-params` sizes from free VRAM.
  The flag
  translation still runs independently, so a translation bug still fails the run.
- **Launch parity check**: after both phases run, the harness extracts each
  phase's effective launch command from its container logs, normalizes
  legitimately-different tokens (ports, hostnames, api keys, weight paths), and
  fails the run if anything else differs. For vllm this compares what is actually
  handed to the engine, including the two values derived at the engine boundary
  (`gpu_memory_utilization` and the `max_model_len` auto-fit sentinel), not the
  config dump — those read `None` where the engine gets `-1`. The comparison
  covers every `vllm_engine_kwargs` key `VllmInfer` forwards, the multimodal
  `limit_mm_per_prompt`/`mm_processor_kwargs` included, so a vision config can't
  benchmark two different engines.
- vLLM tool-call and reasoning parsers are auto-detected from the model's chat
  template whenever the config leaves them unset, so the raw arm calls
  `resolve_tool_parser`/`resolve_reasoning_parser` itself and passes the result
  as `--enable-auto-tool-choice --tool-call-parser`/`--reasoning-parser`. Both
  bench configs hit this: Qwen2.5 resolves to `hermes`, Qwen3 to `hermes` plus
  `deepseek_r1`. The resolved names never reach the kwargs dump, so the actor
  logs them on their own line for the parity check to read. Note that the
  chat-template toggle defaults the actor pins into `chat_template_kwargs` are
  request-time, not launch flags — `vllm serve` has no equivalent, so a template
  whose `enable_thinking` default differs from the parser's is the one
  request-shaping difference this harness cannot equalize.
- **Tokenizer extraction**: GGUF configs can't be used as Hugging Face repo IDs by
  the bench client, so the harness reads a `# bench-tokenizer: <repo-id>` comment
  from the yaml (inert to modelship). `--tokenizer` overrides it.
- vLLM: `gpu_memory_utilization` is not a config key — `resolve_gpu_memory_utilization()`
  derives it (fractional `num_gpus` > preflight > loader default) and both arms
  call it. Setting it in the yaml is a hard error.
- vLLM, multi-slot (`tp × pp > 1`): modelship forces
  `distributed_executor_backend="ray"` because its actor sits in a 0-GPU
  placement-group bundle. vLLM's own default there is `mp`, so the raw arm is
  passed `--distributed-executor-backend ray` too — otherwise the sweep would
  compare two executors rather than two wrappers around one. It is also not a
  config key, so the actor logs it beside the kwargs dump for the parity check
  to read.
- On a multi-GPU host both raw entrypoints set `CUDA_VISIBLE_DEVICES` to exactly
  the devices Ray reserves for the modelship actor before exec'ing the server.
  Without this the raw phase inherits every GPU the container's `--gpus` flag
  exposed and llama.cpp auto-splits across all of them. For llama_server that
  reservation is `num_gpus`; for vllm it is `num_gpus` or, for a multi-slot
  deploy, `tensor_parallel_size × pipeline_parallel_size` — config validation
  collapses `num_gpus` to `1.0` once tp × pp carries the count, so reading
  `num_gpus` alone would pin a tp=2 baseline to one GPU. Pass every device to
  `--gpu-device` (e.g. `--gpu-device 0,1`) for such a run.
