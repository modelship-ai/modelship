# AGENTS.md

Operational notes for agents working in this repo. Read before making changes.

## Toolchain

- Python is pinned exactly to `3.12.10` (`requires-python = "==3.12.10"`). Not `>=3.12`. That applies to the engine; `bootstrap/` (published as `mship`) is `>=3.10` because it runs before the pinned environment exists.
- Dependency manager is **uv**, pinned exactly by `required-version` in `pyproject.toml`: CI installs that version, the `Dockerfile` copies the same tag (`tests/test_uv_pin.py` guards it), and any other uv refuses to run in the repo. Bump both together and re-run `make pins`.
- Never run `pip install`; always use `uv sync` / `uv run` / `uv lock`.
- `cuda` and `cpu` extras are mutually exclusive (declared in `[tool.uv] conflicts`). `torch` / `torchvision` come from different indexes per extra (`pytorch-cu130` vs `pytorch-cpu`). A third extra, `thin`, is empty (base deps only) — no torch/vllm, used by the thin control/coordinator image.

## Commands you'd otherwise guess wrong

```bash
# Install deps for development (choose cuda OR cpu, plus dev)
uv sync --extra dev --extra cuda   # what CI uses
uv sync --extra dev --extra cpu --extra vllm-cpu   # CPU-only dev (vllm-cpu pulls `openai`, which conftest needs)

# The canonical dev loop (mirrored in CI and Makefile)
make lint        # ruff check + ruff format --check + pyright  (all three MUST pass)
make lint-fix    # ruff check --fix + ruff format
make test        # uv run pytest tests/ -v

# Run a single test
uv run pytest tests/test_config.py::TestLlamaServerConfig::test_defaults -v
```

CI (`.github/workflows/ci.yml`) runs `uv sync --extra dev --extra cuda` on Linux, then `ruff check`, `ruff format --check`, `pyright`, and `pytest tests/ -v -m "not integration"` — same filter as the "skip integration" guidance below. Separate CI jobs cover the `bootstrap/` package (multi-Python-version matrix), lockfile/pins parity, and the Helm chart. Match the lint+test job locally before pushing.

`make lint` requires `--extra cuda` to be installed. Pyright resolves imports against the active venv, and `gguf`, `diffusers`, and `psutil` only ship under the cuda extra, so lint on a cpu-only sync fails with `reportMissingImports`. (`vllm` is importable under both extras as of the Stage E0 CPU wheel wiring — it's no longer cuda-only, just not enough on its own to make lint pass cpu-only.) Tests run fine on either extra (the cuda extra is a superset).

Agents: when running tests on your own initiative (sanity-checking a change, verifying a bump), skip the slow `integration`-marked suite by default — `uv run pytest tests/ -v -m "not integration"`. Only run the full `make test` (which includes integration) when explicitly requested.

Pre-commit only runs ruff; it does **not** run pyright or tests, so don't rely on the hook to catch type errors.

## OpenAI protocol fidelity

`modelship/openai/protocol.py` is the request/response surface clients see. When adding or changing models there:

- **Follow the official OpenAI API specification strictly.** Field names, types, defaults, optionality, and shape of nested objects must match what `platform.openai.com/docs` documents for the corresponding route.
- Do not invent fields to expose loader-specific knobs (Diffusers `strength`, vLLM `stop_reason`, etc.). Carry loader-specific defaults via the per-model `*_config` in `infer_config.py` instead.
- Missing optional OpenAI fields are fine when a feature is genuinely unsupported. Adding fields that aren't in OpenAI's spec is not — it locks clients into a modelship-specific dialect and breaks the drop-in-replacement guarantee.
- When OpenAI's spec evolves (new fields, new response_format values, new routes), update the protocol shapes before wiring the backend.

When in doubt, check OpenAI's reference for the exact route. Existing deviations are documented and tracked separately; do not add new ones.

## Lint / format / typecheck rules

- Line length **120** (not 88). Ruff handles formatting; `E501` is disabled because the formatter owns line length.
- Ruff rule set: `E, W, F, I, N, UP, B, SIM, RUF`. `I` means isort runs — don't hand-sort imports.
- Pyright `typeCheckingMode = "basic"`, scoped to `modelship`. Don't add `# type: ignore` without checking pyright actually complains in basic mode.

## Running the server

`mship start`, `mship join` and `mship deploy` are the engine commands (console script, installed via `pip`/`uv tool install "mship[metal]"`; `python -m modelship.launcher <command>` from source). `modelship/launcher.py` resolves the cache root, checks the Python version, detects the accelerator (`cuda`/`rocm`/`xpu`/`metal`/`cpu`, keyed on the installed torch build — see `modelship/utils/accelerator.py`), and on macOS auto-provisions `llama-server` before handing off to `modelship/driver.py:run`. The commands split by lifetime:

- `mship start` creates this machine's Ray head (sized from `MSHIP_NODE_NUM_CPUS`/`MSHIP_NODE_NUM_GPUS`, auto-detected if unset; metrics on `--metrics-port`, default 8079), brings up Serve and the gateway, sends any `--config`/`--model` as a deploy request without waiting for it, stays running and tears the cluster down on exit. It refuses when any Ray node already runs on the machine.
- `mship join --gcs-address HOST[:PORT]` starts this machine's Ray node as a worker, stays running and leaves on exit. Node only — no driver, no model changes.
- `mship deploy` attaches to the cluster of a node on this machine, sends a deploy request to the deploy coordinator and exits (with `--wait`, waits for it to succeed or fail and exits with the outcome). A failed request is rolled back; a signal only stops the `--wait`. It waits up to 5 min for a Ray node on this machine and for the deploy coordinator (`start` creates it before the gateway), then errors or exits 4.
- `mship deploy --ray-dashboard-url URL` (`modelship/remote.py`) runs that same `mship deploy` on another cluster's head instead, as a Ray job through the head's dashboard: the forwarded args minus its own flags, the `--config` file in the job's metadata (96 KiB cap, measured JSON-escaped as Ray sends it), the head's env, `MSHIP_RAY_AUTH_TOKEN` as an explicit Bearer header. It streams the job's log and exits with the job's exit code. It waits up to 5 min for the dashboard to answer and to take the job, then exits 3; a lost dashboard mid-deploy exits 3 at once, and nothing is resubmitted.
- `mship deploy --cancel ID` cancels a queued or running deploy request instead and rolls back what it submitted, then exits; with `--wait` it waits for the rollback and exits with the outcome (a signal only stops the wait).

`start` and `deploy` share the model handling:

1. Reads the models from `--config <path>` (a missing file is a hard error; examples in `config/examples/`) or `--model`; there is no default file. Absent both, it redeploys the gateway's committed models that are missing (nothing on a fresh `start`).
2. `deploy` adds models **additively** by default (each app is named `<gateway>.<model>-<config fingerprint>`, e.g. `modelship.qwen-3f9a0c12de`); `deploy --reconcile` instead makes the cluster match the config exactly (add/remove/replace) — it never tears the cluster down. `start` has no such flag: it always sends its models as a reconcile, so the gateway serves exactly them and any others its state store holds are removed.
3. The gateway is a FastAPI Ray Serve app named `modelship` (override with `--gateway-name`), listening on port `8000`, mounted at `/<slugified-gateway-name>` (e.g. `/modelship/v1/...`) since every gateway on a cluster shares one HTTP proxy/port. `deploy` creates one only for an explicit `--gateway-name` that doesn't exist yet.

The published images are built by running the native install: `uv tool install mship==<version>` then `mship bootstrap --<variant>`, both with `UV_FIND_LINKS` pointed at the release wheels built earlier in `release.yml` (so the images are proven against the exact artifacts PyPI later receives — `pypi` is gated on `docker`). Their ENTRYPOINT runs under `tini` (PID 1, reaps orphans) and prepends `mship`, so `docker run <img> start --config …` and `docker run <img> info` hit the same CLI as a native node. The engine lives at `/opt/mship/envs/<variant>/.venv`; there is no `/.venv` and no source tree. `mship bootstrap` never looks for the accelerator — the variant flag alone decides what to provision, so the cuda images build on GPU-less runners with no opt-out flag; it gates only on the variant's build prerequisites (`--cuda` needs `nvcc` and `ninja`, checkable anywhere), and the engine commands gate on the real accelerator. The `dev` target is the exception — it branches off `base`, syncs from `uv.lock` into `/.venv` with `--no-install-project`, and bakes no `llama-server`, so inside a Dev Container use `uv run python -m modelship.launcher start` (see `docs/development.md`).

Right after connecting to Ray, the driver logs the cluster's observed totals (`Connected to Ray: N node(s), X GPU / Y CPU total (Xa GPU / Ya CPU schedulable now)`) — useful for telling a legitimately-waiting head (0 schedulable resources, no workers joined yet) apart from a misconfigured one.

## Architecture quick map

- `modelship/driver.py` — the `start`/`join`/`deploy` phases plus the deploy loop they share (`_apply`). `build_deployment_options` (in `modelship/deploy/actor_options.py`) handles GPU allocation: multi-slot vLLM deploys (`tp*pp > 1`) always build a Ray Serve placement group (one whole-GPU bundle per slot, STRICT_PACK) that vLLM's ray executor inherits via `get_current_placement_group()`. Single-slot deploys use a scalar `num_gpus` on the outer actor. Fractional `num_gpus` (`<1`) is single-GPU only — combining it with TP/PP is rejected at config time (Ray packs fractional PG bundles onto the same physical GPU). `llama_server` and (Metal-only) `whispercpp` also accept a fractional `num_gpus`, sized by preflight against the declared share of the GPU's total capacity (see `docs/model-configuration.md`'s "Sharing one GPU" section); `sherpa_onnx` never touches CUDA so `num_gpus` is ignored. Every deploy also requests a `mship_<loader>` custom resource (`modelship/deploy/capabilities.py`), so it only schedules onto a node that actually has that loader's backend installed — see `CLAUDE.md`'s capability-aware scheduling sharp edge.
- `modelship/openai/api.py` — FastAPI gateway. Uses `RequestWatcher` + a single shared `DisconnectRegistry` Ray actor (keyed by request id) to propagate client disconnects across process boundaries.
- `modelship/infer/model_deployment.py` — the single `@serve.deployment` actor class; lazily imports the right backend based on `config.loader`.
- `modelship/infer/infer_config.py` — pydantic config schemas **and** `RawRequestProxy` / `DisconnectRegistry`. `RawRequestProxy` exists because FastAPI `Request` cannot cross Ray process boundaries; any new attribute vLLM reads from `raw_request` must be added there.
- `modelship/infer/{vllm,diffusers,llama_server,stable_diffusion_cpp,whispercpp,sherpa_onnx}/` — one subdir per loader. Each has an `*_infer.py` and an `openai/` adapter subpackage. `modelship/infer/llama_server/llama_server_infer.py` is the exception: a flat file with no `openai/` subpackage — it proxies a `llama-server` subprocess's own OpenAI-compatible HTTP API rather than parsing output in-process.

## Tests

- Under `tests/`. Use `pytest-asyncio` for async tests.
- The default suite mocks out Ray Serve; it does **not** spin up a real cluster. Pattern: access the wrapped class via `ModelshipAPI.func_or_class` to bypass the `@serve.deployment` wrapper (see `tests/test_api.py`).
- `tests/test_*_integration.py` are real end-to-end tests against a live cluster and real (small) models — opt-in via pytest markers (`integration` plus a per-loader/feature marker, e.g. `vllm`, `blue_green`, `cluster_join`; see `pyproject.toml`'s `markers`). Excluded by default (`-m "not integration"`); don't assume `make test`/CI coverage without them.

## Releases

`make release-{patch,minor,major}` is the only supported path. It refuses to run off `main` or with a dirty tree, bumps `pyproject.toml`, runs `uv lock`, generates a CHANGELOG entry from conventional commits (`feat:`, `fix:`, `refactor:|perf:|docs:|chore:|build:|ci:|style:|test:`), commits, tags `vX.Y.Z`, and pushes. The `release.yml` workflow publishes the Docker images and PyPI package. Do not bump the version by hand.

Commit messages matter: use Conventional Commits prefixes so the changelog generator picks them up.

## Working with git

- **The maintainer pushes; agents don't.** Create branches and commits locally, but leave `git push` to the human (this environment has no `ssh`, and the remote is SSH anyway). Hand back the branch name and let them push and open the PR.
- **Never amend; always add a new commit.** Don't `git commit --amend` (or rebase/squash) unless explicitly asked. Follow-up work — review feedback, refactors, even bug fixes to a just-made commit — goes in its own commit stacked on top of the original, so history stays reviewable.

## Gotchas

- There is no default models.yaml: models come only from `--config <path>` or `--model`, and with neither, `start` comes up with no models and `deploy` redeploys the gateway's committed ones. `config/models.yaml` stays gitignored for local configs.
- **A deploy request runs from one diff.** The deploy coordinator builds it (`diff` and `rollback_diff` in `modelship/infer/deploy_coordinator.py`, `build_diff` in `modelship/deploy/diff.py`) from the request, the gateway's committed version and Serve's apps (`serve_apps` in `modelship/deploy/strategy.py`: each app's status and the replica-count fields its deployment is set to, never its running replica count); the worker only executes it. Each model gets one action (add, replace, rescale, redeploy, remove, keep), and they run in the order of `Diff`'s fields: `leftovers` deleted, `scale_downs` (rescales that raise no replica limit, and under `blue_green` a replaced app whose count drops), `stop_start` deletes of `retired`, `deploys`, `scale_ups`, switch and commit, `blue_green` deletes of `retired`. A step before the commit must leave every committed model serving, so deletes stay where the strategy puts them. Replica fields are compared in Serve's shape (`serve_scaling` fills in Serve's autoscaling defaults), so an unset tunable is not a change. A rollback is the bare request's diff without its deploys. `get_serve_details` is a private Serve client call; `TestServeApps` fails loudly if a Ray bump changes what is read.
- **A replica-count change is applied to the live app.** `num_replicas` and `autoscaling_config` are outside the config fingerprint. When only they change, `rescale_app` (`modelship/deploy/strategy.py`) reads the app's deployment back from Serve (`live_app`: the bound config with its pinned source, the runtime_env, the version) and re-submits it with the new fields under that same version, so no running replica restarts; a rollback gives each live app its committed fields back (`restore_scaling` in `modelship/deploy/worker.py`). Don't set a version on an ordinary submit: a plain re-submit gets a random version, which is what restarts a `DEPLOY_FAILED` app on the worker's retry — under an unchanged version Serve leaves it failed. `Deployment._version` and the client's `get_deployment_info` are private Serve names; `TestRescaleApp` fails loudly if a Ray bump drops them.
- vLLM version is pinned (`vllm==0.28.0`). Do not bump casually — the TP scheduling logic in `build_deployment_options` (`modelship/deploy/actor_options.py`) defaults to the Ray V2 executor, and the loader imports vLLM-internal `entrypoints.*`/`renderers.*`/`parser.*` module paths that upstream restructures between minors (0.25 deleted `OpenAIServingRender`; the loader now builds `vllm.renderers.online_renderer.OnlineRenderer` directly, see `CLAUDE.md`).
- **The vLLM engine is given the reasoning parser before it starts.** `_resolve_chat_parsers` (`modelship/infer/vllm/vllm_infer.py`) reads the chat template from vLLM's cached tokenizer right after `create_engine_config`, resolves the tool and reasoning parser names (`modelship/infer/vllm/parsing/detect.py`) and sets `structured_outputs_config.reasoning_parser` on the built config; `init_serving_chat` reuses the names. vLLM reads that field once, at engine start, and holds a grammar (`response_format`, a `required` or named `tool_choice`) back until the reasoning ends only when it is set. Unset, the grammar applies from the first token, the model never closes its thinking, and the whole answer comes back as `reasoning` with `content` null; the per-request `reasoning_ended` (`derive_reasoning_ended` in `modelship/infer/vllm/engine_ops.py`) is ignored. `TestSignaturesGuardVllmBump` and `test_structured_output_manager_takes_a_reasoning_parser_set_on_the_built_config` fail loudly if a vLLM bump moves the field.
- **The vLLM reasoning parser is picked from vLLM's registry.** `resolve_reasoning_parser` (`modelship/infer/vllm/parsing/detect.py`) takes the first name in `ReasoningParserManager.list_registered()` found in `config.json`'s `model_type` or its architecture that builds on the tokenizer, implements `is_reasoning_end` (the engine holds a grammar back until it returns true) and whose markers, if it declares any, occur in the chat template; otherwise `deepseek_r1`. An explicit `reasoning_parser` always wins. `TestVllmReasoningRegistry` fails loudly if a vLLM bump drops the fallback name or the marker properties; `tests/test_vllm_reasoning_detection.py` (`integration` marker) holds the expected pick per model.
- **The vLLM tool parser is picked by trying vLLM's parsers.** `resolve_tool_parser` (`modelship/infer/vllm/parsing/detect.py`) renders an assistant tool call through the chat template (`tool_probe.py`) and takes the registered parsers that read it back; a tie goes to those whose forced-call grammar accepts the text, then to the name closest to `config.json`'s `model_type` or architecture. A template that writes no call falls back to a parser named there, then to the markers in `classify_tool_template`. Neither parser is picked from `models.yaml`'s `model` or `name`. `tests/test_vllm_tool_probe.py` runs vLLM's real parsers and fails loudly if a vLLM bump changes them; `tests/test_vllm_tool_detection.py` (`integration` marker) holds the expected pick per model.
- **GGUF is not supported on the `vllm` loader.** vLLM moved GGUF out of tree in 0.24 and it's stayed out since; the only external `vllm-gguf-plugin` (`0.0.2`) has a stale `override_quantization_method` signature incompatible with vLLM's current quantization API (it breaks *all* quantized models, not just GGUF), so it is deliberately not installed. `resolve_all_model_sources` rejects a `.gguf` on the vllm loader at driver preflight and points to `llama_server`. For GGUF use `loader: llama_server`; feed the vllm loader safetensors or an AWQ/GPTQ/FP8 quant.
- `llama_server` loader (GGUF) launches a `llama-server` subprocess — found via `MSHIP_LLAMA_SERVER_BIN`, pinned in the Docker images at `/opt/llama.cpp/llama-server.sh` — and proxies its native OpenAI API instead of parsing output in-process. `num_gpus` accepts `0`, a fraction `< 1` (shares one GPU; preflight sizes `n_ctx`/`n_gpu_layers` to `fraction × total VRAM`, not free VRAM), or a whole integer. `num_gpus > 0` honors `n_gpu_layers`; `sherpa_onnx` never touches CUDA so `num_gpus` is ignored (forced to `0`); `stable_diffusion_cpp` is forced to `num_gpus: 0` in `actor_options.py` everywhere except Darwin, where ggml picks up Metal on its own and `num_gpus` is honored. `--parallel` slots give real request concurrency instead of serializing behind a single lock. Tool-call/reasoning parsing is llama-server's own, auto-detected per chat template: named-function `tool_choice` forcing is unsupported globally (silently falls back to `auto`), and `tool_choice: required` is grammar-enforced for harmony-style templates but a silent no-op for hermes-style ones (e.g. Qwen3); bare `response_format: {"type": "json_object"}` (no `schema` key) is also unenforced despite llama-server's own docs claiming support — `type: json_schema` requests (what modelship sends whenever a schema is given) are unaffected. No persistent on-disk prompt cache, and the host-RAM one (`--cache-ram`) is off unless `cache_ram_mib` is set. See `docs/model-configuration.md`'s llama_server section for the full field table and examples. Binaries for every platform come from modelship's own `llama-cpp-builds` release of one llama.cpp tag (`.github/workflows/llama-cpp-build.yml`), consumed identically by the Docker images and by `launcher.py`'s native auto-provisioning; a native CUDA host also fetches `libggml-cuda.so`.
- **Ray sums per-node resource reports with zero cross-node hardware awareness** — not IP-based, not GPU-UUID-based, nothing. Every raylet self-reports (auto-detected or explicit `--node-num-cpus`/`--node-num-gpus`/`--node-memory`) and the cluster total is a blind sum. This only bites when 2+ modelship containers actually share physical hardware — separate hosts, and k8s pods with correctly-set `resources.requests/limits` (the NVIDIA device plugin hands out disjoint GPU UUIDs; kubelet enforces real cgroup CPU/memory quotas), are unaffected. Co-locating containers on one host requires manually fencing disjoint hardware *before* Ray starts: `--gpus device=0` / `device=1` (not `--gpus=all` on more than one container — CUDA_VISIBLE_DEVICES has no cross-container coordination either, so two `--node-num-gpus=1` containers both sharing `--gpus=all` will both land their actor on physical GPU 0, not split 0/1), and a real per-container memory limit (`docker run --memory=`) or `--node-memory` (splits into Ray's `object_store_memory`/30% + schedulable `memory`/70%, matching Ray's own auto-detect proportion) so each container's memory auto-detect doesn't independently claim the whole host. CPU has no such fencing flag beyond `--node-num-cpus` itself — pair it with `--cpuset-cpus` if you need real isolation, not just accounting. Separately, a single container with no `--memory` limit and no `--node-memory` set no longer blindly trusts Ray's own uncapped-cgroup estimate (host total minus only this container's usage) — it auto-sizes from actually-free host RAM instead, so a non-Docker consumer on the same host (e.g. a co-resident VM) is correctly accounted for. That auto-detect is still per-container and point-in-time, so it does *not* replace explicit `--node-memory` when co-locating multiple modelship containers as described above.
- Metrics are on by default on port **8079** (not 8000). Disable with `--no-metrics` or `MSHIP_METRICS=false`.
- Preflight hardware auto-sizing is on by default. Disable with `--no-preflight` or `MSHIP_PREFLIGHT=false` to run models on loader/library defaults plus explicit `models.yaml` config only — useful for benchmarking across hardware.
- Log level `TRACE` (below `DEBUG`) is a custom level and logs full request/response payloads.
- Three images are published from the unified `Dockerfile` (`--build-arg MSHIP_VARIANT=thin|cpu|cuda`), all under `ghcr.io/modelship-ai/modelship`:

  | Variant | Tag | Platforms | Contains |
  |---|---|---|---|
  | thin (control/coordinator) | `:X.Y.Z`, `:latest` | amd64, arm64 | base only — no torch/vllm |
  | cuda (GPU node) | `:X.Y.Z-cuda`, `:latest-cuda` | amd64 | torch cu130 + vllm + CUDA runtime |
  | cpu (CPU node) | `:X.Y.Z-cpu`, `:latest-cpu` | amd64, arm64 | torch CPU + vllm CPU wheel |

  Floating tags (`:latest*`) are single-node only — Ray refuses to form a cluster across mismatched
  versions, so any multi-node deployment pins every node to the same `X.Y.Z` (or `-cuda`/`-cpu`) tag.
  For the thin variant the bootstrapper sets `MSHIP_NODE_NUM_CPUS=0`/`MSHIP_NODE_NUM_GPUS=0` when it
  launches the engine (they are not image `ENV`), so a thin container never advertises
  capacity it can't serve — it's a driver/coordinator role, not a compute node.

## Further reading

Prefer these over re-reading source when orienting:

- `docs/architecture.md` — request lifecycle, loaders
- `docs/development.md` — full dev-container + manual-Docker setup, env vars
- `docs/model-configuration.md` — `models.yaml` reference
- `config/examples/` — working `models.yaml` files for each backend
