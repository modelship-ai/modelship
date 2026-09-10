# AGENTS.md

Operational notes for agents working in this repo. Read before making changes.

## Toolchain

- Python is pinned exactly to `3.12.10` (`requires-python = "==3.12.10"`). Not `>=3.12`. That applies to the engine; `bootstrap/` (published as `mship`) is `>=3.10` because it runs before the pinned environment exists.
- Dependency manager is **uv**.
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
- Pyright `typeCheckingMode = "basic"`, scoped to `modelship`, `mship_deploy.py`. Don't add `# type: ignore` without checking pyright actually complains in basic mode.

## Running the server

`mship deploy` (console script, installed via `pip`/`uv tool install "mship[metal]"`) is the entry point; `modelship/launcher.py` resolves the cache root, checks the Python version, detects the accelerator (`cuda`/`rocm`/`xpu`/`metal`/`cpu`, keyed on the installed torch build — see `modelship/utils/accelerator.py`), and on macOS auto-provisions `llama-server` before handing off to `modelship/driver.py:main`. `mship_deploy.py` survives as a 3-line back-compat shim to `modelship.driver.main`, for source runs. The driver itself:

1. Reads `config/models.yaml` (gitignored — copy one from `config/examples/`). An explicit `--config <path>` that doesn't exist is a hard error; absent both `--config` and the default file, it bootstraps an empty coordinator (gateway up, no models) that waits for a later `--config`/`--reconcile` or a joining node.
2. Starts its **own** Ray head by default (sized from `MSHIP_NODE_NUM_CPUS`/`MSHIP_NODE_NUM_GPUS`, auto-detected if unset; metrics on `RAY_METRICS_EXPORT_PORT`) and tears it down on exit. With `--use-existing-ray-cluster` it instead connects to a cluster you manage via `ray.init(address="auto")` and deploys-and-exits without teardown — the driver must run **on** a cluster node (Docker co-located / k8s RayJob / bare-metal node); it cannot attach from off-cluster.
3. Deploys models **additively** by default (new deployments get a random suffix, e.g. `qwen-a3f9k`). Pass `--reconcile` to instead make the cluster match the config exactly (add/remove/replace) — it never tears the cluster down.
4. Starts a FastAPI gateway Ray Serve app named `modelship` (override with `--gateway-name`), listening on port `8000`, mounted at `/<slugified-gateway-name>` (e.g. `/modelship/v1/...`) since every gateway on a cluster shares one HTTP proxy/port.

The published images are built by running the native install: `uv tool install mship==<version>` then `mship bootstrap --<variant>`, both with `UV_FIND_LINKS` pointed at the release wheels built earlier in `release.yml` (so the images are proven against the exact artifacts PyPI later receives — `pypi` is gated on `docker`). Their ENTRYPOINT prepends `mship`, so `docker run <img> deploy --config …` and `docker run <img> info` hit the same CLI as a native node. The engine lives at `/opt/mship/envs/<variant>/.venv`; there is no `/.venv` and no source tree. `mship bootstrap` never looks for the accelerator — the variant flag alone decides what to provision, so the cuda images build on GPU-less runners with no opt-out flag; it gates only on the variant's build prerequisites (`--cuda` needs `nvcc` and `ninja`, checkable anywhere), and `deploy` gates on the real accelerator. The `dev` target is the exception — it branches off `base`, syncs from `uv.lock` into `/.venv` with `--no-install-project`, and bakes no `llama-server`, so inside a Dev Container use `uv run mship_deploy.py` or `uv run python -m modelship.launcher deploy` (see `docs/development.md`).

Right after connecting to Ray, the driver logs the cluster's observed totals (`Connected to Ray: N node(s), X GPU / Y CPU total (Xa GPU / Ya CPU schedulable now)`) — useful for telling a legitimately-waiting head (0 schedulable resources, no workers joined yet) apart from a misconfigured one.

## Architecture quick map

- `modelship/driver.py` — Ray init + deploy loop (`mship_deploy.py`'s former contents; that file is now a 3-line shim). `build_deployment_options` (in `modelship/deploy/actor_options.py`) handles GPU allocation: multi-slot vLLM deploys (`tp*pp > 1`) always build a Ray Serve placement group (one whole-GPU bundle per slot, STRICT_PACK) that vLLM's ray executor inherits via `get_current_placement_group()`. Single-slot deploys use a scalar `num_gpus` on the outer actor. Fractional `num_gpus` (`<1`) is single-GPU only — combining it with TP/PP is rejected at config time (Ray packs fractional PG bundles onto the same physical GPU). `llama_server` and (Metal-only) `whispercpp` also accept a fractional `num_gpus`, sized by preflight against the declared share of the GPU's total capacity (see `docs/model-configuration.md`'s "Sharing one GPU" section); `sherpa_onnx` never touches CUDA so `num_gpus` is ignored. Every deploy also requests a `mship_<loader>` custom resource (`modelship/deploy/capabilities.py`), so it only schedules onto a node that actually has that loader's backend installed — see `CLAUDE.md`'s capability-aware scheduling sharp edge.
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

- `config/models.yaml` is gitignored — copy one from `config/examples/`. A missing default config no longer errors (bootstraps an empty coordinator instead); a missing *explicit* `--config <path>` still does.
- vLLM version is pinned (`vllm==0.28.0`). Do not bump casually — the TP scheduling logic in `build_deployment_options` (`modelship/deploy/actor_options.py`) defaults to the Ray V2 executor, and the loader imports vLLM-internal `entrypoints.*`/`renderers.*`/`parser.*` module paths that upstream restructures between minors (0.25 deleted `OpenAIServingRender`; the loader now builds `vllm.renderers.online_renderer.OnlineRenderer` directly, see `CLAUDE.md`).
- **GGUF is not supported on the `vllm` loader.** vLLM moved GGUF out of tree in 0.24 and it's stayed out since; the only external `vllm-gguf-plugin` (`0.0.2`) has a stale `override_quantization_method` signature incompatible with vLLM's current quantization API (it breaks *all* quantized models, not just GGUF), so it is deliberately not installed. `resolve_all_model_sources` rejects a `.gguf` on the vllm loader at driver preflight and points to `llama_server`. For GGUF use `loader: llama_server`; feed the vllm loader safetensors or an AWQ/GPTQ/FP8 quant.
- `llama_server` loader (GGUF) launches a `llama-server` subprocess — found via `MSHIP_LLAMA_SERVER_BIN`, pinned in the Docker images at `/opt/llama.cpp/llama-server.sh` — and proxies its native OpenAI API instead of parsing output in-process. `num_gpus` accepts `0`, a fraction `< 1` (shares one GPU; preflight sizes `n_ctx`/`n_gpu_layers` to `fraction × total VRAM`, not free VRAM), or a whole integer. `num_gpus > 0` honors `n_gpu_layers`; `sherpa_onnx` never touches CUDA so `num_gpus` is ignored (forced to `0`); `stable_diffusion_cpp` is forced to `num_gpus: 0` in `actor_options.py` everywhere except Darwin, where ggml picks up Metal on its own and `num_gpus` is honored. `--parallel` slots give real request concurrency instead of serializing behind a single lock. Tool-call/reasoning parsing is llama-server's own, auto-detected per chat template: named-function `tool_choice` forcing is unsupported globally (silently falls back to `auto`), and `tool_choice: required` is grammar-enforced for harmony-style templates but a silent no-op for hermes-style ones (e.g. Qwen3); bare `response_format: {"type": "json_object"}` (no `schema` key) is also unenforced despite llama-server's own docs claiming support — `type: json_schema` requests (what modelship sends whenever a schema is given) are unaffected. No persistent on-disk prompt cache. See `docs/model-configuration.md`'s llama_server section for the full field table and examples. Binaries for every platform come from modelship's own `llama-cpp-builds` release of one llama.cpp tag (`.github/workflows/llama-cpp-build.yml`), consumed identically by the Docker images and by `launcher.py`'s native auto-provisioning; a native CUDA host also fetches `libggml-cuda.so`.
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
  A thin container bakes `MSHIP_NODE_NUM_CPUS=0`/`MSHIP_NODE_NUM_GPUS=0` so it never advertises
  capacity it can't serve — it's a driver/coordinator role, not a compute node.

## Further reading

Prefer these over re-reading source when orienting:

- `docs/architecture.md` — request lifecycle, loaders
- `docs/development.md` — full dev-container + manual-Docker setup, env vars
- `docs/model-configuration.md` — `models.yaml` reference
- `config/examples/` — working `models.yaml` files for each backend
