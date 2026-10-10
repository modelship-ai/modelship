# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.8.0] - 2026-10-10

### Added
- a deploy request runs from one diff built by the deploy coordinator
- deploy through a Job running mship deploy --ray-dashboard-url
- mship deploy waits up to 5 min for the head
- name resources after the release alone
- the head turns Ready only once mship start is done
- drop the default config/models.yaml
- mship deploy --ray-dashboard-url deploys to a remote cluster
- the Ray token comes only from env; start warns about an exposed dashboard
- start --enable-ray-auth; the chart always enables it
- remove MSHIP_API_KEYS gateway auth
- mship deploy --cancel replaces mship stop
- the chart sets the gateway's autoscaling range instead of a replica count
- every gateway autoscales, sized by mship start's --gateway-* flags
- mship stop --wait waits for the cancelled deploy's rollback
- mship deploy sends the request and exits; --wait waits for the outcome
- the deploy worker logs when a rollback finishes
- mship deploy sends a request and waits for it; mship stop --deploy-id cancels one
- the gateway coordinator routes by the deploy coordinator's routing version and reports switches
- the deploy coordinator queues requests per gateway and runs their workers
- a deploy worker actor plans, submits, switches, commits and rolls back one request
- keep each gateway's committed and previous deploy version in the state store
- deploy waits for the replica coordinator to remove retired apps
- the gateway answers 503 for a configured model with nothing serving
- the replica coordinator derives routing instead of keeping a registry
- routing rules as a pure function
- deploy hands models to Serve instead of gating on free capacity
- serialise model loads per node with a deploy-lease actor
- pre-delete hook deletes the RayCluster before its Secrets, then the release's state
- python -m modelship.uninstall deletes a release's RayCluster, then its state
- namespace modelship's Redis keys per release
- namespace Redis state keys with ?namespace= on the URI
- run mship start and join as the Ray nodes
- add --ray-dashboard-host and --metrics-port to the node commands
- split mship deploy into start, join and deploy
- forward cluster settings to the actors that read them
- hold a cluster-wide lease while downloading model sources
- add per-source cached check and leftover removal

### Fixed
- a model that reserves no GPU sees none
- start serves exactly the models it is given
- the vLLM reasoning fallback is deepseek_r1 only
- a vLLM request without tools parses no tool calls
- a vLLM reasoning parser without is_reasoning_end is not picked
- the vLLM parsers are not picked from models.yaml names
- the vLLM tool parser is picked by trying vLLM's parsers
- the vLLM benchmark resolves the reasoning parser with the new arguments
- the vLLM reasoning parser is picked from vLLM's registry
- tool_choice none drops the tools for every loader
- the vLLM engine gets the reasoning parser before it starts
- a num_replicas or autoscaling_config change rescales the live deployment
- a num_replicas or autoscaling_config change replaces the deployment
- a CPU llama_server model with n_ctx, n_gpu_layers and tensor_split set is still checked
- the not-sized message names memory and n_ctx before --no-preflight
- a CPU llama_server model that preflight cannot size fails its deploy
- llama_server prompt cache is off by default
- llama_server CPU preflight fits against free RAM
- a model with replicas or autoscaling must set its context length
- kokoro bundles download from modelship's model-bundles release
- kokoro-multi-lang-v1_0 pin matches the re-uploaded bundle
- start refuses a malformed Redis state-store URL before starting its head
- start refuses a bad state store or gateway sizing without a traceback
- the disconnect and heartbeat registries restart in place
- pin the disconnect and heartbeat registries to the head
- accept every reasoning_effort value OpenAI defines
- model and gateway names reject '#'
- Ray auth stays off without --enable-ray-auth on Ray 2.59
- a remote deploy waits for a dashboard that's down at connect or submit
- the gateway Service routes only to pods whose Serve proxy answers
- cap a remote deploy's config by its JSON-escaped size
- a local deploy with no Ray token exits with the token hint
- a gateway replica being stopped waits for its background responses
- drop the unread RAY_LOG_LEVEL and correct the documented library log levels
- log level, target and OTLP endpoint are set per node on mship start and join
- an empty cluster setting stays empty instead of taking its default
- the bench's modelship arm runs mship start
- logging and metrics are set once on mship start for the whole cluster
- the head forwards the default state store explicitly
- mship deploy uses the head's state store
- a deploy that is already rolling back refuses a cancel
- llama_server rejects an mmproj that isn't a GGUF file at driver preflight
- a repeat SIGINT/SIGTERM no longer re-enters the stop handler mid-teardown
- llama_server rejects a source that isn't a GGUF file at driver preflight
- a dead worker's late switch or commit is refused during its rollback
- a rollback deletes against the version the deploy coordinator holds
- a gateway without a committed version routes nothing
- the gateway coordinator never creates the deploy coordinator
- a deploy stops waiting on a model whose app was deleted elsewhere
- models autoscaled to zero stay routed
- every configured model with nothing serving answers 503, not 404
- every delete of a gateway's app runs under its deploy lease, re-checked first
- an interrupted or failed deploy no longer deletes the apps it submitted
- a gateway's deploys take turns under its deploy lease
- the gateway takes expected models from a snapshot even when its routes fail to apply
- the deploy loop reports Serve's deployment message as the pending reason
- a new deploy-lease actor grants at once when only the gateway exists
- the deploy loop leaves an UNHEALTHY app pending
- the deploy "add" count covers only models that came up
- a failed app is redeployed and a live unrouted app is routed
- deploy-lease renewal retries a failed call until the lease would have expired
- blue_green keeps the old app serving until its replacement is ready
- deploy loop never blocks on removal and fails models still waiting to retry at the deadline
- model replicas register their own deployment once loaded
- a replacement deploy-lease actor waits out a lease period before granting
- a lost deploy lease no longer kills the loading replica
- omit workerGroupSpecs when there are no worker groups
- always declare the metrics port, require metrics for the PodMonitor
- create the RayCluster with KubeRay's Redis cleanup finalizer
- run the image entrypoint under tini
- keep worker groups as KubeRay writes them back, size memory from requests
- pin the deploy RayJob to the head by resource
- keep apps and pass Redis credentials when the GCS is Redis-backed
- create the operator probe only when a deploy has models to add
- carry models.yaml in the chart's deploy RayJob
- run the chart on KubeRay 1.7.1
- relay the state-store URI through model replicas
- carry the state-store URI through the deploy coordinator
- expand the state-store URI before checking it for a password
- keep the redis password out of the forwarded state-store URI
- pin the cluster-wide actors to the head node
- configure logging in vLLM's engine and worker processes
- bound the lease release call
- pin HF_HUB_CACHE under MSHIP_CACHE_DIR for replicas
- time out stalled archive downloads
- create the lease actor outside the caller's placement group
- keep the download lease until the held work ends, even when cancelled
- report unrecoverable on-disk source state as fatal, not a download retry
- check a local model path exists on the replica's node
- stop double-counting snapshot download progress
- remove the downloaded archive when extraction fails
- give each sherpa_onnx bundle download its own archive path
- reject a llama-server binary this user can't execute
- reject a llama-server wrapper this user can't execute
- run the container as the owner of the mounted cache
- stop mship deploy rewriting the llama-server wrapper
- stop forwarding the driver's HF settings to replicas
- resolve whispercpp built-in names through the HF cache
- expand cache placeholders in the bench's raw entrypoints
- resolve replica cache paths on each node, not the driver
- apply cache-dir flags before the driver sets its cache env
- chown only the image's fixed cache paths in the entrypoint
- keep vLLM/Triton/FlashInfer caches node-local under MSHIP_NODE_CACHE_DIR

### Changed
- num_gpus 0 hides the node's GPUs from the model
- start and join set Ray's zero-GPU override
- seed the two llama_server tool_choice fallback requests
- start serves exactly its config
- start sends a reconcile and takes no --reconcile
- the diffusers integration tests use sd-turbo
- the diffusers example uses sd-turbo
- uv is pinned by required-version
- the Dockerfile's uv tag matches required-version
- the release wheels job installs the pinned uv
- pin uv to 0.12.23
- the tool_choice comment states vLLM's default for an absent key
- the vLLM reasoning fallback is deepseek_r1
- one reasoning fallback
- inline the shared-run threshold
- reasoning parsers must implement is_reasoning_end
- reasoning parsers without is_reasoning_end, tool_choice defaults
- the vLLM parsers are picked from config.json
- parser detection ignores the names in models.yaml
- how the vLLM tool parser is picked
- tool parser detection by trying vLLM's parsers
- shorter reasoning parser comments and notes
- how the vLLM reasoning parser is picked
- reasoning parser detection from vLLM's registry, thinking switched off
- the vLLM reasoning parser and tool_choice none
- constrained output on a reasoning model, tool_choice none on both loaders
- what a deploy changes and in what order
- a scale-down makes room for the same request's new apps
- a replica-count change is applied to the running deployment
- a rescale keeps the running replicas under traffic
- test_cancel_on_completed_is_idempotent sends temperature=0
- README states loader parity in one sentence
- the examples README says what the per-loader files are for
- restore the example config entries
- correct the token path, port table, stop timeout and example configs
- the quick start quotes the real ready log line; multi-node pins use X.Y.Z
- cap the autoscale test model's n_ctx
- CLAUDE.md notes resolve_ray_auth_env's disabled default
- guard every private ray.init kwarg start_head passes
- bump Ray to 2.59
- install notes watch the pods, not the RayCluster
- clarify the deploy Job's retry comment
- drop API-key leftovers from chart comments and docs
- a remote deploy retries nothing
- trim the identity comments
- live remote deploy and cancel with token auth
- join --gcs-address, start --gcs-port
- MSHIP_API_KEYS removed
- mship deploy --cancel replaces mship stop
- trim the gateway autoscaling test's module docstring
- load past its target scales the gateway out and the new replica serves the model
- gateway autoscaling settings replace --gateway-replicas
- a replica on a joined node follows that node's logging and the head's format and metrics
- Serve's HTTP proxy and a replica's first startup lines log at the head's level
- clarify /readyz stays 503 with no models configured
- the gateway coordinator is created without the state store
- the broken deploy fixture fails in its replica via a text model's GGUF as mmproj
- only mship start creates the deploy coordinator
- drop the deploy lease startup window
- the deploy coordinator releases a deploy lease once its replica is RUNNING or gone
- a committed deployment whose backend keeps dying is kept
- silence Ray's own deprecation warning when the deploy coordinator checks for a restart
- integration tests for the deploy ledger
- the deploy ledger
- remove the gateway deploy lease and the code only it needed
- integration tests for concurrent, pending and multi-gateway deploys, gateway coordinator restart and replica retirement
- name both coordinators in the driver's actor comment
- describe the gateway's deploy lease, and drop the registry and deploy-lock mentions
- say which coordinator wherever a comment, doc or message said only "coordinator"
- the deploy coordinator grants the per-node deploy leases
- name the GatewayCoordinator in the architecture doc
- rename the replica coordinator to the gateway coordinator
- routing is derived by the replica coordinator, not kept in a registry
- app names carry the gateway as a prefix
- the replica-coordinator and logging tests stop leaking a non-propagating modelship logger
- trim the run_deploy_loop docstring
- cover what _apply deploys, routes and removes
- sync the chart dashboard and document --deploy-timeout
- retry backoff uses None when unset, and the first poll logs what is outstanding
- drop the GPU-footprint deploy ordering
- trim comments and stale retry wording
- correct what --no-metrics disables
- note worker-group removal and Redis keys left by a disabled cleanup
- validate the rendered chart with a server-side dry-run apply
- run the chart on Helm 4 and test a server-side upgrade
- rename --dashboard-port to --ray-dashboard-port
- correct the chart's Serve-proxy readiness claims
- document start, join and deploy
- install the locked kuberay-operator tarball for the server dry-run
- raise max_tokens on the response-format tool_choice=none test
- raise max_tokens on the llama_server tool-call tests to fit the reasoning preamble
- remove the init_serving_* startup logs
- name the HF cache settings replicas override
- skip the download lease once the source is cached while waiting
- trim lease comments
- log the request body limit in MiB
- stub configure_logging when constructing the lease actor
- describe download leases
- route sherpa_onnx bundles through a shared model-sources package
- add Discord community links across README, docs, and package metadata
- raise the huggingface-hub floor to 1.18.0
- record the 2026-09-13 llama_server GPU bench results on Qwen3.5-9B
- disable llama.cpp's prompt cache in the llama bench configs

## [0.7.15] - 2026-09-13

### Added
- fail fast on AMD and Intel GPUs when num_gpus > 0
- let vLLM auto-fit the GPU context length
- report what the vllm engine actually deployed

### Fixed
- reuse the llama-server connection across streaming chat requests
- detect the raw vllm parsers from the configured tokenizer
- tolerate a missing -ts when pinning the llama baseline
- probe the bench gpu device with a valid nvidia-smi query
- match the vllm tool and reasoning parsers across both bench arms
- match the vllm executor backend across both bench arms
- replay the llama tensor split in the bench baseline
- pin the raw vllm arm to every GPU the deploy reserves
- compare the vllm engine args a vision config actually changes
- sample VRAM on the GPUs the bench is pinned to
- quote the bench --gpus device list for docker's CSV parser
- honour MSHIP_GATEWAY_NAME in the bench harness
- pair llama flags with negative numeric values
- stop bench cleanup masking the run's exit status
- log the vllm engine args the actor actually passes
- stop the test suite from silently booting a real Ray head
- let vllm auto-fit the context on cpu instead of dictating it
- reject a max_model_len that is neither a length nor the sentinel
- stop reading the auto-fit sentinel as a context length
- submit the death report instead of awaiting it
- drop a model from the readiness set when it never registered at all
- name the reason when the vllm output handler exits without raising
- count replica deaths for the deployment's life, not in a window
- retire a deployment whose backend keeps dying
- drop a model from the readiness set when its last deployment goes
- stop retrying a failing deploy forever
- deploy a model with no chat template instead of crashing
- install torchaudio from the accelerator indexes
- size the MLA gmu haircut against device total
- stop the driver reserving a CUDA context on every GPU
- divide the CPU gmu by the cgroup-clamped RAM total
- reserve process RSS in the CPU gpu_memory_utilization
- stop subtracting a cudagraph estimate from the preflight budget
- correct three misreadings in the deployment summary
- tighten llama_server preflight guards and config validation
- delegate llama_server preflight sizing to llama fit-params
- normalize every image content-part shape to the nested chat form

### Changed
- drop review-flagged comments in bench/run.sh and bench/README.md
- record the 2026-09-12 vllm GPU bench results on Qwen3.5-9B
- trim the bench config comments
- move the GPU bench configs to Qwen3.5-9B
- explain the baseline-only OMP_NUM_THREADS, and floor it
- correct the llama preflight -c 0 claim
- record the 2026-09-10 loader-parity audit
- correct the llama preflight n_gpu_layers claim
- strip the commentary from the example configs
- run preflight in both bench arms
- drop ms-python.python from the dev container
- bump vllm to 0.28.0
- make the auto-fit sentinel derived instead of user input
- pin that the Serve replica context crosses threads
- cut the commentary explaining Ray rather than this code
- move deployment teardown out of serve_utils
- consolidate the comments in strategy.py
- allow torchaudio to differ across bootstrap variants
- regenerate the pins for the torchaudio index move
- drop the unread cpu_count and unified_memory
- trim the comments in preflight/base.py
- read free VRAM from NVML instead of a CUDA context
- force both fractional tenants onto one GPU
- describe num_gpus as a GPU count or a share of one
- trim the comments added in the auto-fit rewrite
- fix an inaccurate docstring in the deployment summary
- correct a misleading comment in bench/rawllama_entrypoint.py
- add a canary test for llama.cpp's own CLI flags
- regenerate pins after removing the gguf dependency
- trim oversized comments across bench/
- drop the backend-rejection comments on image url nesting
- name both backends' rejection of a bare-string image url

## [0.7.14] - 2026-08-27

### Added
- set the nested loader-config blocks from mship deploy flags
- aggregate HF model download progress and add a heartbeat
- provision llama-server in the dev container

### Fixed
- derive gpu_memory_utilization in the bench harness too
- reject out-of-range resource values, and name the flags in the tuning-flag error
- use explicit None checks for total_bytes instead of truthiness
- move HF_HUB_DISABLE_XET default out of model_resolver's import side effect
- drop private filter_repo_objects dep, don't log success on a failed download
- disable hf_xet by default to avoid its intermittent download stalls
- derive the mid-stream response.failed event from the frames already sent
- emit an SSE error chunk on mid-stream failures instead of aborting
- log structured fields instead of raw inference error messages
- log silent ErrorResponse paths and validate tool_choice at the source
- read tensor names from every shard of a sharded GGUF
- fall back safely when a GGUF's tensor list is incomplete
- count only attention blocks when sizing GGUF KV cache
- size hybrid models correctly in vLLM preflight
- mirror vLLM's max_num_seqs floors in the MLA workspace estimate
- shard the MLA prefill workspace estimate by tensor_parallel_size
- account for MLA attention in llama_server preflight sizing
- account for MLA attention in vllm preflight sizing
- account for sliding-window attention in llama_server GGUF preflight
- clamp _fit_len_with_sliding to ctx_cap on every return path
- regenerate bootstrap pins for the transformers 5.14.1 bump
- pin transformers to 5.14.1, newest version Gemma 4 works with
- account for sliding-window attention in vLLM KV-cache preflight

### Changed
- cover the preflight shapes that recommend a derived gpu_memory_utilization
- drop the vllm engine kwargs modelship sets or ignores itself
- bump devcontainer node feature to 22
- skip oversized arrays when reading GGUF metadata
- make the unreadable-tensor-list fallback actually unreadable
- consolidate comments in llama_cpp.py to house length
- trim comments in the sliding-window preflight fix to house length
- cover llama_server GGUF sliding-window preflight
- extract sliding-window KV math into a shared preflight module
- cover the CPU sliding-window preflight path
- update Open Responses conformance result to Gemma 4

## [0.7.13] - 2026-08-19

### Added
- deploy a single model from CLI flags instead of models.yaml
- gate bootstrap --cuda on the CUDA toolchain, not the GPU
- split provisioning into `mship bootstrap`

### Fixed
- reject --model with --config at parse time, not after Ray starts
- point integration test base URLs at the /modelship route prefix
- pass the missing match count to the sharded-GGUF log call

### Changed
- lead the README with an agentic hero and a real quickstart
- trim the comments added with the model flags
- trim verbose comments across the test suite
- trim the comments added by the config_schema split
- split the models.yaml schemas out of infer_config
- require MSHIP_VERSION explicitly in the image build
- drop the stale docker entry-point note from AGENTS.md
- make the bootstrap hardware gate a flag, not an env var
- split installation by method and drop stale entry points
- install the release wheel in the images
- audit and trim docs for accuracy and brevity

## [0.7.12] - 2026-08-15

### Added
- two-stage bootstrapper with pinned Python and hash-pinned deps
- warn when llama-server lists no CUDA device
- offload llama_server to CUDA on a native Linux install
- make mship[cpu] torch-free, split vLLM into mship[vllm-cpu]
- auto-provision llama-server on native Linux, not just Metal

### Fixed
- apply the data filter's checks by hand where the interpreter lacks it
- let a zero-capacity coordinator hold a config it can't run itself
- resolve torch from the index the lock pinned it to
- extract archives on Pythons without tarfile filters
- bound the bootstrapper's downloads with a socket timeout
- unbreak the chart job — version-agnostic image assertion and missing capabilities
- keep pins diffable so a ray or torch bump is reviewable
- pin vllm-cpu extra to +cpu local versions, not bare 0.26.0

### Changed
- skip the filter branch where the interpreter has no filter
- ignore coverage artifacts
- trim bootstrapper comments and READMEs to the mechanical facts
- fail fast when the cuda variant is built for arm64
- drop the CUDA device check from provisioning
- native CUDA hosts now offload GGUF too
- cut the Dockerfile comments back to mechanical facts
- reopen the pin-bump PR after a llama.cpp build
- source every llama.cpp binary from our own b10375 release
- document the native Linux CUDA install
- build llama.cpp arm64 with gcc-14
- point make llama-cpp-bump at the unified build workflow
- build the CUDA backend with the stock jammy toolchain
- trim comments in the llama.cpp build workflow
- build llama.cpp for every platform from one tag
- raise the CUDA build's ccache limit to 2G
- fail fast on a bad publish token, keep the built artifact
- add Linux CUDA llama.cpp backend build workflow
- fix development.md's stale macOS-only llama-server claim
- trim comments and install prose to mechanical facts
- fix native Linux vllm-cpu install recipe

## [0.7.11] - 2026-08-11

### Fixed
- route whispercpp download progress through the pywhispercpp logger
- stop whispercpp loader's native/download logs bleeding into console
- trim comment verbosity in download-progress throttling change
- throttle HF download progress logging instead of spamming a line per tick
- drop CoreML support from sherpa_onnx loader, CPU only

## [0.7.10] - 2026-08-11

### Added
- reject explicit vllm gpu_memory_utilization, derive it always
- support fractional GPU sharing on llama_server and whispercpp

### Fixed
- update stale hosted-tool-rejection integration test to match drop behavior
- compute real GPU demand/ordering from deployment options, not raw num_gpus

### Changed
- split test_integration.py into per-loader/per-concern files
- pin temperature=0 and case-insensitive match in GPU-sharing test
- trim over-long comment in _gpu_footprint sort key
- strip chat-register justification from GPU-sharing comments/docs
- add integration coverage for cross-loader fractional GPU sharing
- document GPU-sharing semantics, drop stale fractional-GPU claims

## [0.7.9] - 2026-08-10

### Added
- add sherpa_onnx TTS loader (kokoro, CPU/CoreML, streaming)
- promote whispercpp to a first-class in-process loader

### Fixed
- clear a stale extract_dir before re-fetching llama-server
- single-flight sherpa_onnx bundle fetches with an flock
- give each sherpa_onnx bundle fetch a unique archive path
- clear a stale cached bundle before re-fetching
- recognize whispercpp built-in model names on a pywhispercpp-less driver
- correct whispercpp ggml magic and English-only language detection
- forward the source-language hint on whispercpp translations
- raise onnxruntime-gpu floor to 1.28.0 for CUDA 13 compatibility
- type the vllm output_handler callback correctly and tighten the process match
- crash the replica actor when a loader's backend dies unexpectedly
- refresh stale and broken model references in example configs
- close vLLM cross-identity prefix-cache timing side channel
- correct compute_lib_level docstring to match its DEBUG floor
- cut library log noise, prefix and de-duplicate startup log lines

### Changed
- remove the plugin system
- validate sherpa_onnx speech output via real STT transcription
- add end-to-end integration coverage for the sherpa_onnx loader
- promote is_pathy to a shared utility
- shorten fetch_and_extract_archive's docstring
- derive the sherpa OfflineTtsConfig sub-config from entry.family
- trim rationale out of two sherpa_onnx docstrings
- drop per-file size/sha256 pins from the sherpa_onnx registry
- unify tarball fetch/verify/extract into modelship.utils.fetch_and_extract_archive
- drop the CUDA-wheel aside from sherpa_onnx scope notes
- remove the whispercpp plugin, superseded by loader: whispercpp
- remove unnecessary comment
- clarify enable_prefix_caching None passthrough is intentional
- trim comments to 1-2 lines

## [0.7.8] - 2026-08-03

### Added
- capability-aware Ray scheduling for loaders and accelerators

### Fixed
- match unquoted exec target in llama-server wrapper detection
- numeric card ordering + full loader coverage in capability parity check

### Changed
- make effective_config.merge() O(n) instead of O(n^2)
- avoid mutable default argument in _hammer
- enforce unique model names in ModelshipConfig, not effective_config
- add sharded GGUF example to llama-server.yaml
- replace gateway round-robin with atomic per-name deployment cutover

## [0.7.7] - 2026-08-01

### Changed
- point repository references at the modelship-ai org

## [0.7.6] - 2026-07-31

### Fixed
- parse MSHIP_NODE_MEMORY with unit suffixes, not bare int()
- auto-detect free host RAM for Ray node memory sizing
- adapt vllm loader to 0.26's OnlineRenderer replacing OpenAIServingRender
- reject malformed mcp policy shapes hidden behind truthiness
- reject malformed mcp policy shapes before the stream opens

### Changed
- re-verify llama.cpp sharp edges against b10200
- bump llama.cpp to b10200
- bump vllm to 0.26.0
- trim comments on the mcp policy-shape validation
- assert launcher env guard inside its patch.dict scope
- add CLA, CODEOWNERS, and governance docs for contributors

## [0.7.5] - 2026-07-30

### Added
- server-side MCP tool execution on /v1/responses
- add background+stream live tailing and resume (Phase E2) to /v1/responses
- add background mode (Phase E1) to /v1/responses

### Fixed
- turn McpToolSpec validation errors into a clean 400
- declare httpx2 as a direct dependency
- drop unsupported hosted Responses tools and merge system-level messages
- require previous_response_id to resume an mcp approval
- validate allowed_tools plain-list form as list[str] too
- synthesized response.created for a fresh tail must be an opening envelope
- validate approval-resume tool name against real discovery, not client input
- normalize mcp_call output/error through _text_of like function_call_output
- reject non-list tool_names in mcp spec instead of matching as a string
- synthesize response.created for a fresh background stream tail
- move background-run liveness into a dedicated HeartbeatRegistry actor
- close touch()'s race against a concurrent terminal write
- return 503 instead of an unhandled 500 when the store fails during staleness reconciliation
- return 503 instead of an unhandled 500 when the store fails during cancel
- preserve original output-item order for executed mcp_call in the turn loop
- emit mcp_call_arguments.delta/.done for approved approval-resume calls
- emit response.mcp_call.in_progress/.failed for rejected approval resumes
- validate mcp_call input items before translating to chat messages
- reject mcp server_url with no hostname
- set output_index for approval-bound mcp_call buffers in _flush()
- reject malformed require_approval bucket shapes instead of crashing
- serialize background+stream buffer appends and flush before discard
- don't let unhandled errors escape background-stream buffering/tailing
- background terminal persistence must let cancel/delete win races
- touch() must not regress a terminal snapshot's status

### Changed
- cover background mode + cancel for server-side MCP tool execution
- consolidate long comments back to 1-2 lines in E2 background-streaming work
- consolidate comments in background-mode PR to 1-2 lines

## [0.7.4] - 2026-07-27

### Fixed
- pin numba, ray, fastapi, and protobuf to unbreak fresh mship installs

## [0.7.3] - 2026-07-27

### Added
- trigger llama.cpp Metal builds on demand via make llama-cpp-bump
- native mship CLI with Metal support for Apple Silicon

### Fixed
- bump Dockerfile llama.cpp Docker image pins in the Metal bump workflow
- publish under PyPI project name mship, not modelship

### Changed
- bump llama.cpp to b9859
- publish llama.cpp Metal builds to a dedicated releases repo
- publish to PyPI without waiting on Docker builds
- fold llama.cpp Metal provisioning into launcher, drop _pins.py

## [0.7.2] - 2026-07-24

### Added
- add WebSocket transport for /v1/responses
- add /v1/responses/compact endpoint

### Fixed
- share the compaction key across processes via the state store
- wire compact instructions through, drop unused prompt_cache_key
- harden compaction_crypto against misconfigured keys and non-ASCII blobs
- reject nested compaction items instead of recursing unbounded
- correct dark-mode diagram switching and drop redundant nav tabs
- use theme-adaptive favicon instead of near-white logo variant
- reject non-list/non-string content instead of silently stringifying it
- close Open Responses conformance gaps in schema, vision input, and SSE termination

### Changed
- trim oversized comments/docstrings and fix WS error-message leaks
- update Open Responses conformance results to 17/17
- avoid re-normalizing already-resolved input on HTTP too
- unify HTTP/WS error handling behind ResponsesApiError
- make the streaming translator transport-neutral
- add Open Responses conformance results to README
- consolidate verbose comments in the responses adapter

## [0.7.1] - 2026-07-23

### Changed
- scope GHA build cache per image variant and make export failures non-fatal

## [0.7.0] - 2026-07-22

### Added
- log a ready-to-paste join command at own-head startup
- wire Ray cluster token auth across head, workers, and RayJob submitter
- make co-located Ray nodes easy and verifiable instead of refusing them
- add --node-memory flag to fence per-container RAM under host co-location
- let docker run pass mship_deploy.py flags directly
- bootstrap an empty coordinator when no --config is present on own-head
- join a Ray cluster as a compute node via --address/--token
- add --ray-port to pin the own-head GCS server port
- download model weights actor-side instead of on the deploy driver
- split Docker images into thin/cuda/cpu variants, fix CI's drifted CUDA pin
- always-on Ray dashboard, opt-in cluster auth, node-scoped resource flags

### Fixed
- fix unreadable tabs/links/buttons in dark mode
- render Material icon shortcodes instead of literal text
- pin docs workflow to exact Python 3.12.10
- don't claim durable state when the default store is in-memory
- diagram caption undersold gateway scaling, oversold state durability
- expand ~ in parse_model_ref so local tilde paths resolve
- never pin a joining node's Ray metrics export port
- treat a pathy model ref as local even when it doesn't exist yet
- stop permanently evicting fatally-failed models from effective config
- bail loudly when --ray-auth=token can't be honored safely

### Changed
- link the hosted docs site from the README
- fix installation.md Python version and CPU image contents
- move docs site to docs.model-ship.ai, not the apex
- build MkDocs Material site for GitHub Pages / model-ship.ai
- reposition README around the self-hosted agent backend
- trim mship_deploy.py comments to at most 2 lines
- reframe production-readiness's Compose item as non-K8s docker run
- add multi-node-without-Kubernetes guide, fix stale cross-node download docs
- drop dead ray_auth_is_safe() guard, resolve auth env unconditionally
- resolve HF repo listing and revision via a single model_info call

## [0.6.5] - 2026-07-16

### Added
- add CLI args for responses TTL and state sweep interval
- add server-side conversation state to /v1/responses

### Fixed
- stop the request watcher on any history-resolution failure
- guard against null response payloads in Responses state handling
- reclaim expired keys in the memory state store
- reject memory:// URIs with a path, not just a host
- make memory:// state store cluster-scoped via a detached Ray actor

### Changed
- cache the sweep interval once per MemoryStoreActor instance
- cover responses state gaps found by ad-hoc endpoint probing
- split openai chat_utils into utils/{chat,responses}; extract Responses gateway helpers from api.py
- move /v1/responses state domain module into openai/state package

## [0.6.4] - 2026-07-15

### Fixed
- tolerate uninspectable apply_chat_template in toggle detection
- harden reasoning-trap reconcile against parse failures and malformed templates
- stop vLLM reasoning models trapping their answer in the reasoning field

## [0.6.3] - 2026-07-14

### Fixed
- catch OSError, not just FileNotFoundError, in weight-footprint listdir
- drop dead itemsize fallback in mamba state-size calc
- account for mamba/SSM recurrent-state cache in vllm preflight

## [0.6.2] - 2026-07-13

### Added
- harden state store for async, TTL, listing, and availability errors
- add identity_key() caller-identity primitive for log correlation

### Fixed
- redirect vLLM usage-stats config dir to MSHIP_CACHE_DIR
- redirect Triton JIT cache to MSHIP_CACHE_DIR
- normalize list() prefix so a trailing slash doesn't break segment matching
- clean up leaked tmp file when FileStateStore.set() fails mid-write
- enforce segment boundaries in StateStore prefix listing, avoid tmp filename collisions
- reject exact "." and ".." trusted-header identity values
- guard resolve_identity() against request-like objects with no state attribute
- use theme-specific SVG variants for README logo

### Changed
- cache identity resolution to avoid redundant env parsing and header lookups
- add project logo to README

## [0.6.1] - 2026-07-10

### Added
- expose llama-server prompt-cache tuning flags

### Fixed
- rename init_serving_embeding to init_serving_embedding
- correct docstring reference to vllm.parser.Parser after alias rename
- validate registry/expected shape when loading replica coordinator state
- degrade _poll_disconnected_ids on any exception, not just RayActorError
- cancel work in run_cancellable when the caller itself is cancelled
- drop stale EmbeddingRequest re-validation workaround
- close chunks generator on every exit path of _stream_responses

### Changed
- unify infer stream/no-stream seams and error handling
- prefix all third-party vLLM imports with Vllm/vllm_
- split deploy coordinator into mutex and replica-routing actors
- batch client-disconnect polling per replica instead of per request
- consolidate the /v1/responses streaming and non-stream lifecycle

## [0.6.0] - 2026-07-07

### Added
- pin greedy load client and make bench result-parity gate relative
- extend bench to llama_server/CPU and add --no-preflight for fair A/B
- fool-proof preflight sizing for vllm CPU deploys and llama_server GPU offload
- make vllm loader installable on the cpu extra
- add native streaming support to /v1/responses for vllm and llama_server
- shape /v1/responses natively from ParsedChatOutput for vllm and llama_server
- rewire vLLM streaming chat onto engine_ops
- abort llama_server non-stream requests on client disconnect
- rewire vLLM non-stream chat onto engine_ops
- quarantine vLLM-internal touchpoints behind engine_ops
- implement embeddings, vision, logprobs, and concurrency coupling for llama_server loader (Stage B4)
- extract 3-field DTO and rewire llama_server non-stream projection (Stage B3)
- ship llama-server in Docker images and wire GPU offload (Stage B2)
- add llama_server loader (Stage B1 of the parser-migration roadmap)

### Fixed
- harden bench A/B harness for fair modelship-vs-raw comparisons
- cancel the in-flight next_item before closing work on teardown
- close stream generators and cancel embeddings on disconnect
- guard vllm CPU preflight against an undiscoverable host RAM probe
- count the output layer's weight in llama_server CPU-resident RAM sizing
- don't let thread-alignment preflight starve declared parallel slots
- vllm embed init, transcription/translation request construction, and Responses streaming usage
- wire llama_server streaming chat onto client disconnect and stamp chunk ids
- guard against out-of-bounds top_logprobs index in vLLM logprobs projection
- replace assert with a defensive check for vLLM prompt_token_ids
- resolve mmproj once on the driver instead of again in the actor
- guard against malformed and non-object JSON responses from llama-server
- derive finish_reason for out-of-range list entries instead of hardcoding stop
- use explicit None check for created timestamp fallback in embeddings projection
- suppress interpreter teardown exceptions inside __del__
- secure pending_client_closes against python interpreter teardown
- intercept and handle mid-stream JSON error payloads from llama-server
- intercept and parse inline JSON error payloads on 2xx responses in llama_server loader
- call self.shutdown() on any exception during llama-server startup
- address concurrency, early-crash thread leaks, and closed-loop shutdown issues in llama_server loader
- assert self._proc is not None to resolve pyright type-checking error
- optimize llama_server loader concurrency, timeouts, and protocol alignment
- reject non-positive parallel and close httpx client on shutdown
- harden llama_server loader streaming and subprocess log draining

### Changed
- remove Home Assistant/Wyoming integration doc
- resolve vllm gpu_memory_utilization default lazily instead of auto-flagging it
- remove one-click profiles system
- repoint driver preflight and the vllm actor at the new parser module
- move vLLM parser detection out of driver preflight into the actor
- delete dead raw-text parser engine from openai/parsers/
- delete vLLM OpenAIServingChat monolith usage
- repoint llama_cpp non-stream chat onto build_from_parsed
- add missing llama_server integration coverage
- document the llama_server loader
- reposition Modelship around its agentic + GPU-sharing wedge

## [0.5.8] - 2026-07-01

### Added
- add GPU offload support to the llama_cpp loader

### Fixed
- gate llama_cpp GPU warning correctly and unbreak CI import of cu130 wheel
- match .GGUF extension case-insensitively in vllm loader guard

### Changed
- bump vllm to 0.24.0

## [0.5.7] - 2026-06-30

### Added
- drive tool calling from per-request tool_choice and infer reasoning state

### Fixed
- harden reasoning probe and auto-path reasoning check
- guard get_parser in reasoning probe against startup crash

## [0.5.6] - 2026-06-29

### Added
- enforce required arguments in Gemma tool-call GBNF grammar
- allow optional leading/trailing whitespace in Gemma GBNF tool call grammar
- enforce schema order and uniqueness in Gemma tool call GBNF grammar
- back require_tool_call with LlamaCppConfig, imply constraining
- constrain FunctionGemma/Gemma4 tool calls with a GBNF grammar
- add llama.cpp native prompt cache support

### Fixed
- robustly filter required schema property elements
- guard empty type list in Gemma value emitter
- emit generic recursive value rules for free-form Gemma tool args
- scope disk-cache replica guard to the llama_cpp loader
- key llama.cpp disk cache by deployment name, not model name
- reject llama.cpp disk cache with multiple replicas
- isolate llama.cpp disk cache per-model under MSHIP_CACHE_DIR
- drop stale registry entries for resurrected deployments on reconcile

### Changed
- simplify Gemma GBNF grammar generator and add defensive schema handling
- set _require_tool_call on __new__-built serving chat in trace test
- simplify multi-replica check for disk cache guard

## [0.5.5] - 2026-06-24

### Added
- add generic chat_template_kwargs for text loaders

### Fixed
- reserve "conversation" key on transformers chat_template_kwargs
- reserve "messages" key in chat_template_kwargs
- honor chat_template_kwargs on transformers streaming path
- drop reserved keys from chat_template_kwargs
- skip tool-call grammar on reasoning deployments
- tolerate null assistant content in prompt rendering

### Changed
- merge chat_template_kwargs before building vllm request

## [0.5.4] - 2026-06-24

### Added
- grammar-constrained tool calling (constrain_tool_calls)
- TRACE-log parsed tool calls handed to client

### Fixed
- allow conversational text around grammar-constrained tool calls
- finalize tool calls in ChatOutputStreamer.finalize()
- fall back to default voice for unknown voice names
- require string call_id and name when building tool-name map
- guard tool-name backfill against malformed messages
- backfill tool-message name for strict chat templates

### Changed
- skip tool-call summary build when TRACE disabled

## [0.5.3] - 2026-06-23

### Added
- log chat request/response payloads at TRACE

### Fixed
- forward passthrough env vars and configure logging in gateway replica

### Changed
- cover streaming TRACE response logging

## [0.5.2] - 2026-06-22

### Fixed
- recover disconnect registry from actor death
- TTL-evict disconnect entries instead of clearing on teardown
- stop request watcher on cancellation during initial response
- time streaming generation/request duration after the stream drains
- count GPU models' host RAM against the RAM budget
- avoid double-counting reclaimable cache on cgroup v1
- refuse cleanly when a capability has no catalog models
- size stacks against free RAM with a weighted knapsack selector

### Changed
- keep gateway route tests off a real Ray cluster
- build and push CPU image before GPU
- add integration tests deploying profiles at cpu/gpu stages

## [0.5.1] - 2026-06-20

### Fixed
- make download atomic to avoid corrupt cached files

### Changed
- only run the chart job when the Helm chart changes

## [0.5.0] - 2026-06-19

### Added
- HA control-plane metrics, per-gateway dimension, chart-shipped alerts/dashboard
- reclaim Ray temp disk on restart and drop --redeploy

### Fixed
- warn when backed by a non-durable memory state store
- retry routing reconcile when a deployment handle isn't ready yet
- wire the Model dropdown into per-model panels
- never let metric emission mask a state-store error
- forward MSHIP_METRICS to replicas so --no-metrics is cluster-wide
- export Ray metrics on the declared port

### Changed
- correct gateway-tagged HA metric count (six, not three)
- make metric/state-store proxies transparent via __getattr__

## [0.4.0] - 2026-06-18

### Added
- image.variant selector (gpu default) for cpu/gpu image tags
- redis-backed GCS fault tolerance, retire reassert cron
- durable coordinator state for head-restart self-heal
- URI-selectable state stores (memory/file/redis)
- multi-node ingress — proxy on every node, Service spans all pods
- coordinator-driven watch model for multi-replica routing
- per-model autoscaling + --state-dir flag
- self-heal CronJob reconciling to the effective config
- durable per-gateway effective config for self-heal
- per-group image/runtimeClassName override; empty workerGroups default
- deploy via RayJob on the cluster, not an off-cluster Job
- KubeRay chart — RayCluster + deploy Job, /readyz gating
- k8s/KubeRay readiness — gateway self-heal, ownership registry, /readyz

### Fixed
- reject file:// URIs with a non-empty host
- opt out of vLLM dictConfig so head-restart recovery doesn't crash GPU replicas
- resolve coordinator off the event loop in the watch loop
- drop stale coordinator handle so the watch loop recovers
- address code-review feedback
- declare head dashboard port (8265) for RayJob submission
- RayJob clusterSelector rejects backoffLimit; drop head /readyz probe
- exit instead of blocking/teardown on an external cluster
- compare MSHIP_RAY_DASHBOARD case-insensitively

### Changed
- chart OCI install + image.variant
- validate Helm chart (lint, kubeconform, kind server dry-run)
- publish Helm chart as OCI artifact to GHCR
- stamp Helm chart version/appVersion + image tag from release tag
- head-node HA, state-store URI, reassert cron removal
- per-model autoscaling, gateway HA, state-dir/self-heal
- mship_deploy owns its Ray head; --use-existing connects
- disable Ray dashboard by default to cut host RAM

## [0.3.0] - 2026-06-14

### Added
- one-click model stacks via MSHIP_MODEL_STACK
- add stable_diffusion_cpp CPU image-generation loader
- streaming event protocol for /v1/responses (Phase A2)
- stateless /v1/responses endpoint (Phase A)

### Fixed
- treat cgroup v1 unlimited sentinel as no-limit at the source
- give CPU leftover to a single anchor, not every generate
- exit cleanly when the stack file can't be written
- keep scaled-down CPU allocation within the budget
- check per-model VRAM, not the sum, on multi-GPU boxes
- create parent dir before writing generated stack yaml
- guard _is_moe against non-dict config sections
- fall back to cgroup limit when psutil RAM probe fails
- validate MSHIP_MODEL_STACK before any filesystem op
- allocate whole-integer num_gpus on multi-GPU boxes
- parse all bundled SSE messages per chat chunk
- guard None tool_calls in streaming delta loop
- validate tool-call association ids on input items
- robust error mapping and full usage-detail propagation
- robust status_code handling and list default_factory
- drop logprobs/top_logprobs defaults from completion kwargs
- reject logprobs explicitly instead of silently dropping

### Changed
- use asyncio.get_running_loop() in async paths
- rebind handle to None instead of del on shutdown
- warm up + repeat sweeps for stable A/B numbers
- update _parse_chat_sse tests for multi-message parsing
- document /v1/responses endpoint and streaming
- split protocol.py into a protocol package

## [0.2.0] - 2026-06-06

### Added
- make Ray Serve concurrency caps configurable
- default usecase to image and reject non-image
- /v1/images/edits and /v1/images/variations
- vision / image_url input support
- structured outputs + tools/response_format compat gate
- enforce tools-supersede-response_format precedence
- extend hardware-aware preflight to llama_cpp loader
- qwen3-coder tool parser and custom-loader preflight

### Fixed
- robust memory-unit parsing for docker stats output
- tolerate single/unquoted yaml values when parsing bench.yaml
- guard nvidia-smi/docker stats against pipefail+set -e
- validate gateway concurrency env vars are positive ints
- accept Open WebUI image[] edit uploads and log 422s
- guard getextrema and soften alpha mask edges
- serialize GPU inference with an asyncio lock
- force image decode so truncated uploads error cleanly
- tolerate from_pipe failures for img2img/inpaint
- decode images in executor and honor alpha masks
- release all shared pipelines on teardown
- swap edit/variation default strengths
- close audio upload files after reading
- close image upload files after reading
- use input image alpha as mask on edits
- emit task field on verbose transcription/translation responses
- drop max_completion_tokens after mapping to max_tokens
- unwrap numpy arrays from gguf ReaderField.contents()
- cap llama_cpp n_ctx when GGUF omits context_length
- accept `call <name>` (whitespace) in addition to `call:<name>`
- account for PG bundle CPUs in coordinator reservation
- strip only structural JSON suffix in streamed args

### Changed
- add modelship vs raw vLLM A/B benchmark harness
- record push ownership and no-amend commit policy
- compute _world_size once in build_deployment_options
- declare image[] as an explicit aliased field
- move teardown into shutdown(), delegate from __del__
- decode edit input image only once
- load uploaded edit mask as grayscale
- run image PNG/base64 encoding in the executor
- tighten protocol shapes to OpenAI spec
- cache the json_object LlamaGrammar
- drop tools/response_format precedence validator
- always use ray placement groups for multi-slot deploys

## [0.1.36] - 2026-05-14

### Added
- include PyTorch .bin/.pt weights in footprint estimate
- preflight estimator, pipeline parallelism, and runtime hardening
- add Gemma 4 and FunctionGemma tool/reasoning parsers
- llama3_json tool-call parser
- mistral tool-call parser
- transformers reasoning content
- llama_cpp reasoning content + parser unification
- vllm reasoning content + auto-detect
- llama_cpp tool calling + cross-loader auto-detection
- auto-detect tool-call parser for transformers loader
- incremental streaming for tool-call parsing
- cross-loader tool-calling toolkit + transformers wiring
- add integration testing suite for OpenAI endpoints

### Fixed
- decouple multimodal max_num_batched_tokens from max_model_len
- restore envelope-} strip and harden Gemma parsers
- consider reasoning parsers when resolving skip_special_tokens
- preserve preamble whitespace next to tool_calls
- case-insensitive .gguf suffix check in chat-template reader
- maintain consistent created timestamp in chat streaming
- make tool-call finalization robust to skipped blocks
- integration tests

### Changed
- add bitsandbytes to gpu extra
- narrow exception scope in Gemma args parser
- optimize noise stripping using delta processing and regex
- improve noise-stripping robustness and reasoning support
- update testing target score to 9/10
- remove python sdk example from quick start
- optimize README for user adoption and clarify production readiness
- refresh roadmap and production readiness state
- tighten agent notes wording
- bump transformers to 5.8.0 and llama-cpp-python to 0.3.22
- unify openai parsers under modelship.openai.parsers
- per-model deploy/reconcile in integration tests
- bump vllm to 0.20.1
- simplify tool-parser detection warning and file open
- optimize transformers stream complexity
- remove unused `_content_parts_len` attribute from ToolCallStreamer

## [0.1.35] - 2026-05-01

### Added
- centralize model source resolution on the driver
- add reconciliation logic for deployments based on models.yaml
- add max_num_batched_tokens to VllmEngineConfig
- add flatten_message_content utility and integrate into llama_cpp
- upgrade vLLM to 0.20.0 and harden inference loaders

### Fixed
- resolver returns file paths for GGUF and sets HF_HOME pre-import
- defer ray cluster env var checks to avoid key error in auto mode
- handle direct Response chunks and improve vLLM embedding error conversion
- remove non-existent io_processor argument from OpenAIServingRender
- type llama_cpp stream iterator so pyright accepts run_in_executor
- tag REQUEST_TOTAL by outcome instead of marking every request processed

### Changed
- simplify LlamaCpp plugin by delegating resolution to the driver
- extract deployment components into modelship.deploy
- fix unused import in mship_deploy.py
- simplify mship_deploy.py by extracting logic
- detect capabilities and emit OpenAI-compliant chat responses for transformers loader
- split llama_cpp into per-surface OpenAI serving handlers

## [0.1.34] - 2026-04-27

### Added
- upgrade vllm to 0.19.1

### Fixed
- propagate log levels before ray import and add pip for runtime_env
- reap orphan vLLM workers on actor death and quiet shutdown noise
- use async fatal error reporting and unique deployment keys
- handle fatal deployment initialization errors to prevent infinite retries
- harden orphan reaping and vllm audio response handling
- vllm 0.19.1 response types, tp>1 init, orphan workers

## [0.1.33] - 2026-04-25

### Fixed
- propagate UID/GID ARGs to all Dockerfile stages

## [0.1.32] - 2026-04-25

### Added
- make Ray CPU/GPU allocation auto-detect by default
- implement dynamic wheel-based plugin deployment

### Fixed
- restrict plugin discovery to directories in Makefile
- normalize plugin wheel names to match PEP 427

### Changed
- refresh roadmap and remove stale MSHIP_PLUGINS references
- use Bash arrays for safe argument handling in scripts
- unify GPU/CPU Dockerfiles and update docs for dynamic extras
- dynamically load plugin extras in dev docker stage

## [0.1.31] - 2026-04-24

### Changed
- drop --compile-bytecode from uv sync in Docker builds

## [0.1.30] - 2026-04-23

### Added
- cluster-wide deploy coordinator and retry-pass deploy loop
- /status readiness endpoint with per-model load timings

### Changed
- slim CUDA runtime, MSHIP_SKIP_SYNC fast-path, misc

## [0.1.29] - 2026-04-20

### Added
- make kokoroonnx plugin engine-agnostic
- add whispercpp STT plugin, expand custom plugin system to all usecases

### Changed
- relicense from MIT to Apache-2.0

## [0.1.28] - 2026-04-19

## [0.1.27] - 2026-04-19

### Changed
- consolidated documentation

## [0.1.26] - 2026-04-19

### Fixed
- incorrect syntax on github release

## [0.1.25] - 2026-04-19

### Fixed
- resolve UnboundLocalError and enable arm64 builds

### Changed
- fix cache volume mounts and update llama_cpp example
- clean up env var building and enable arm64 builds

## [0.1.24] - 2026-04-18

### Added
- add llama_cpp loader for cpu-only gguf inference

## [0.1.23] - 2026-04-17

### Added
- migrate cache to /.cache, fix CUDA 12 mismatch, and logging typos
- add --openai-api-port flag and run container as non-root user

### Fixed
- update for ci
- update for ci
- update for ci

### Changed
- decouple OpenAI protocol models from vLLM
- improve quick start with correct docker env vars and CPU-first example
- add public roadmap
- add badges and "Why Modelship?" section to README

## [0.1.22] - 2026-04-15

### Added
- add transformers CPU inference, TRACE logging, and fix audio resampling

### Fixed
- resolve pyright type errors across serving modules

## [0.1.21] - 2026-04-13

### Fixed
- remove dockerfile old config folder setup

## [0.1.20] - 2026-04-13

### Added
- auto-generate changelog from conventional commits during release
- add Prometheus alerting rules, Grafana alerts row, and monitoring docs
- add syslog and OpenTelemetry log export
- additive deploys with --redeploy flag and multi-gateway support

### Fixed
- makefile fix for multi-line changelog

## [0.1.11] - 2025-06-20

### Fixed
- Makefile release process

### Changed
- Consolidated environment variables

## [0.1.10] - 2025-06-19

### Added
- Security policy and vulnerability reporting guidelines

## [0.1.8] - 2025-06-18

### Fixed
- Production Docker build

## [0.1.7] - 2025-06-17

### Changed
- Upgraded plugin system
- Migrated Orpheus to new plugin architecture

## [0.1.6] - 2025-06-16

### Fixed
- GitHub Actions release workflow

## [0.1.5] - 2025-06-15

### Added
- Multi-GPU fractional model support
- Sequential Ray deployment to prevent model load memory spikes
- Kokoro plugin configuration
- Fine-tuned example configs for various GPU sizes

### Fixed
- Tool calling bugfix

## [0.1.4] - 2025-06-14

### Added
- Per-actor cache environment variables
- Dedicated Ray actor for each model
- Cache folder for downloaded models

### Fixed
- Type fix and stability improvement

## [0.1.3] - 2025-06-13

### Fixed
- uv lock file

## [0.1.2] - 2025-06-12

### Added
- Lock file for reproducible builds

## [0.1.1] - 2025-06-11

### Added
- Initial release with GitHub Actions CI/CD
