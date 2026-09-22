"""Shared test fixtures, including the integration suite's session-scoped cluster
infrastructure shared by every `@pytest.mark.integration` file."""

import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml

import modelship.infer.infer_config as infer_config
from modelship.utils.model_flags import GENERATED_MODEL_ARGS, MODEL_ARG_KEYS
from openai import OpenAI


@pytest.fixture(autouse=True)
def neutralize_request_watcher():
    """Stubs the disconnect registry and watch loop so route-handler tests don't spin
    up a real Ray cluster; a test module needing the real watcher can override this fixture."""

    async def _noop_watch(self):
        return

    with (
        patch.object(infer_config, "get_disconnect_registry", return_value=MagicMock()),
        patch.object(infer_config.RequestWatcher, "_watch", _noop_watch),
    ):
        yield


# ---------------------------------------------------------------------------
# Integration suite: real Ray cluster + real models, `@pytest.mark.integration`.
# ---------------------------------------------------------------------------

OPENAI_API_BASE = "http://localhost:8000/modelship/v1"
HEALTH_URL = "http://localhost:8000/modelship/health"

# Per-model configs; Deployer.deploy(*names) writes a subset into a one-shot
# models.yaml and runs `mship deploy --reconcile` to swap the deployed set.
# 64x64 solid red PNG — one unambiguous color, large enough to survive image preprocessing.
RED_IMAGE_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAUElEQVR42u3PQREAAAQAMGTQP5kwUni42xos"
    "pzs+q3hOQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA4N4CQ5MBhMgt2OkA"
    "AAAASUVORK5CYII="
)

MODEL_CONFIGS: dict[str, dict] = {
    "chat-capable": {
        "name": "chat-capable",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "usecase": "generate",
        "loader": "vllm",
        # num_gpus is also wired into vllm's gpu_memory_utilization; 0.1 leaves no room for KV cache
        "num_gpus": 0.5,
        "vllm_engine_kwargs": {
            "max_model_len": 2048,
            "enforce_eager": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        },
    },
    "chat-reasoning": {
        "name": "chat-reasoning",
        # Qwen3-0.6B natively emits <think>...</think>; max_model_len is bumped for reasoning headroom.
        "model": "Qwen/Qwen3-0.6B",
        "usecase": "generate",
        "loader": "vllm",
        "num_gpus": 0.5,
        "vllm_engine_kwargs": {
            "max_model_len": 4096,
            "enforce_eager": True,
            "enable_reasoning": True,
            "reasoning_parser": "deepseek_r1",
        },
    },
    "chat-vlm": {
        "name": "chat-vlm",
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "usecase": "generate",
        "loader": "vllm",
        "num_gpus": 1,
        "vllm_engine_kwargs": {
            "max_model_len": 8192,
            "enforce_eager": True,
            "limit_mm_per_prompt": {"image": 2},
            "mm_processor_kwargs": {"min_pixels": 50176, "max_pixels": 200704},
        },
    },
    "autoscale-llama": {
        "name": "autoscale-llama",
        # Tiny CPU GGUF so the host can hold several replicas at once;
        # target_ongoing_requests=1 drives scale-out under light concurrent load.
        "model": "lmstudio-community/Qwen3-0.6B-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_cpus": 1,
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 3,
            "target_ongoing_requests": 1,
            "upscale_delay_s": 2,
            "downscale_delay_s": 10,
        },
    },
    "chat-llama-server": {
        "name": "chat-llama-server",
        # parallel: 4 gives multi-slot concurrency; n_ctx is per-slot (launched with
        # -c n_ctx*parallel).
        "model": "lmstudio-community/Qwen3-0.6B-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_cpus": 2,
        "llama_server_config": {
            "n_ctx": 4096,
            "parallel": 4,
        },
    },
    "chat-vlm-llama-server": {
        "name": "chat-vlm-llama-server",
        # Smallest GGUF VLM that fits a CPU deploy; mmproj is what gates image input.
        "model": "ggml-org/SmolVLM-256M-Instruct-GGUF:SmolVLM-256M-Instruct-Q8_0.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_cpus": 2,
        "llama_server_config": {
            "mmproj": "ggml-org/SmolVLM-256M-Instruct-GGUF:mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
            "n_ctx": 4096,
        },
    },
    "chat-llama-server-plain": {
        "name": "chat-llama-server-plain",
        # Same GGUF as chat-llama-server but without a <think> preamble; needed for response_format tests.
        "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_cpus": 1,
    },
    "chat-llama-server-gpu": {
        "name": "chat-llama-server-gpu",
        # Same GGUF as chat-llama-server-plain, on a whole GPU (exercises the -ngl offload path).
        "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_gpus": 1,
        "num_cpus": 1,
    },
    "frac-share-vllm": {
        # Paired with frac-share-llama-server (0.6 + 0.3 = 0.9). No vllm_engine_kwargs
        # — preflight sizes max_model_len from num_gpus alone.
        "name": "frac-share-vllm",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "usecase": "generate",
        "loader": "vllm",
        "num_gpus": 0.6,
    },
    "frac-share-llama-server": {
        # Paired with frac-share-vllm. Same GGUF as chat-llama-server-gpu, fractional.
        "name": "frac-share-llama-server",
        "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_gpus": 0.3,
    },
    "embed-model-llama-server": {
        "name": "embed-model-llama-server",
        # Real embeddings via llama-server subprocess (--embedding), distinct from
        # the vllm-loader embed-model below.
        "model": "nomic-ai/nomic-embed-text-v1.5-GGUF:nomic-embed-text-v1.5.Q4_K_M.gguf",
        "usecase": "embed",
        "loader": "llama_server",
        "num_cpus": 1,
    },
    "embed-model": {
        "name": "embed-model",
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "usecase": "embed",
        "loader": "vllm",
        "num_gpus": 0.15,
        "vllm_engine_kwargs": {
            "trust_remote_code": True,
        },
    },
    "stt-model": {
        "name": "stt-model",
        "model": "openai/whisper-tiny",
        "usecase": "transcription",
        "loader": "vllm",
        "num_gpus": 0.15,
        "vllm_engine_kwargs": {
            "trust_remote_code": True,
        },
    },
    "stt-cpp-model": {
        "name": "stt-cpp-model",
        "model": "tiny.en",
        "usecase": "transcription",
        "loader": "whispercpp",
        "num_cpus": 1,
    },
    "stt-cpp-multilingual": {
        "name": "stt-cpp-multilingual",
        # Multilingual counterpart of stt-cpp-model: exercises auto-detection and translate.
        "model": "tiny",
        "usecase": "transcription",
        "loader": "whispercpp",
        "num_cpus": 1,
    },
    "tts-model": {
        "name": "tts-model",
        "model": "kokoro-en-v0_19",
        "usecase": "tts",
        "loader": "sherpa_onnx",
        "num_cpus": 1,
    },
    "image-model": {
        "name": "image-model",
        "model": "stabilityai/sdxl-turbo",
        "usecase": "image",
        "loader": "diffusers",
        "num_gpus": 1,
        "diffusers_config": {"num_inference_steps": 2, "guidance_scale": 0.0},
    },
    "image-cpu-model": {
        "name": "image-cpu-model",
        "model": "gpustack/stable-diffusion-v2-1-turbo-GGUF:*Q4_1.gguf",
        "usecase": "image",
        "loader": "stable_diffusion_cpp",
        "num_cpus": 4,
        "stable_diffusion_cpp_config": {"sample_steps": 4, "cfg_scale": 1.0},
    },
}


# Half the single-model call sites deploy through the CLI flags and half through a
# models.yaml, so the same integration suite covers both surfaces. Split on sorted
# position rather than hash(): str hashing is salted per process.
CLI_ROUTED = frozenset(name for i, name in enumerate(sorted(MODEL_CONFIGS)) if i % 2 == 0)

_OPTION_BY_PATH = {arg.path: arg.option for arg in GENERATED_MODEL_ARGS}


def cli_routed(config: dict) -> bool:
    """Whether a lone model deploys via the `--model` flags rather than a config file."""
    return config["name"] in CLI_ROUTED


def _model_flags(config: dict) -> list[str]:
    """The `mship deploy` flags for one models.yaml entry: root keys by name, a
    nested block one flag per leaf. Generated flags take YAML text, and json.dumps
    emits valid YAML for every value shape here."""
    flags: list[str] = []
    for key, value in config.items():
        if key in MODEL_ARG_KEYS:
            flags += [f"--{key.replace('_', '-')}", str(value)]
        elif (key,) in _OPTION_BY_PATH:
            flags += [_OPTION_BY_PATH[(key,)], json.dumps(value)]
        else:
            for sub_key, sub_value in value.items():
                flags += [_OPTION_BY_PATH[(key, sub_key)], json.dumps(sub_value)]
    return flags


class _Deployer:
    """Runs `mship deploy --reconcile` against the running gateway to swap the
    deployed set; re-deploying the same set is a no-op. A lone CLI-expressible model
    goes through the `--model` flags, everything else through a one-shot models.yaml.
    """

    def __init__(self, tmp_dir: Path) -> None:
        self._tmp = tmp_dir
        self._current: frozenset[str] = frozenset()

    def deploy(self, *model_names: str) -> None:
        wanted = frozenset(model_names)
        if wanted == self._current:
            return

        slug = "+".join(sorted(wanted)) or "empty"
        configs = [MODEL_CONFIGS[n] for n in sorted(wanted)]
        if len(configs) == 1 and cli_routed(configs[0]):
            input_args = _model_flags(configs[0])
        else:
            input_args = ["--config", str(self._write_config(f"models-{slug}.yaml", configs))]

        self._run(input_args, slug, "stop_start")
        self._current = wanted

    def deploy_raw(self, models: list[dict], *, replace_strategy: str = "stop_start") -> None:
        """Like deploy(), but takes raw model dicts directly and lets the caller pick
        --replace-strategy; resets self._current since this bypasses the by-name cache."""
        slug = "raw-" + ("+".join(sorted(m["name"] for m in models)) or "empty")
        config_path = self._write_config(f"models-{slug}-{replace_strategy}.yaml", models)
        self._current = frozenset()
        self._run(["--config", str(config_path)], slug, replace_strategy)

    def deploy_cli(self, *flags: str) -> None:
        """Deploy straight from `--model` flags. Resets the by-name cache, as
        deploy_raw does."""
        self._current = frozenset()
        self._run(list(flags), "cli", "stop_start")

    def _write_config(self, filename: str, models: list[dict]) -> Path:
        config_path = self._tmp / filename
        with open(config_path, "w") as f:
            yaml.dump({"models": models}, f)
        return config_path

    def _run(self, input_args: list[str], slug: str, replace_strategy: str) -> None:
        log_path = self._tmp / f"reconcile-{slug}-{replace_strategy}.log"
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "modelship.launcher",
                    "deploy",
                    *input_args,
                    "--reconcile",
                    "--replace-strategy",
                    replace_strategy,
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=900,
            )
        if result.returncode != 0:
            tail = log_path.read_text()[-4000:]
            pytest.fail(
                f"mship deploy --reconcile failed for {slug} ({replace_strategy}, exit {result.returncode}).\n"
                f"Log file: {log_path}\nLast 4KB:\n{tail}"
            )


@pytest.fixture(scope="session")
def mship_cluster(tmp_path_factory):
    """Starts a cluster with a long-lived `mship start` process bound to an empty
    models.yaml; per-test code deploys models additively via `_Deployer.deploy(...)`."""
    tmp_dir = tmp_path_factory.mktemp("mship_integration")
    empty_config = tmp_dir / "empty-models.yaml"
    log_path = tmp_dir / "mship_start.log"

    # `start` refuses while another Ray node runs on this machine.
    subprocess.run(["ray", "stop", "--force"], check=False)

    with open(empty_config, "w") as f:
        yaml.dump({"models": []}, f)

    log_file = open(log_path, "w")  # noqa: SIM115 — kept open for subprocess lifetime, closed in cleanup
    proc = subprocess.Popen(
        # 2 gateway replicas exercise multi-replica convergence; num_cpus=0 makes this cheap.
        [
            "uv",
            "run",
            "python",
            "-m",
            "modelship.launcher",
            "start",
            "--config",
            str(empty_config),
            "--gateway-replicas",
            "2",
            "--prune-ray-sessions",
            "false",
        ],
        # MSHIP_TRUSTED_IDENTITY_HEADER lets tests simulate distinct identities via
        # extra_headers; MSHIP_RAY_DASHBOARD_HOST binds the dashboard to 0.0.0.0.
        env={
            **os.environ,
            "MSHIP_TRUSTED_IDENTITY_HEADER": "X-Mship-Test-Identity",
            "MSHIP_RAY_DASHBOARD_HOST": "0.0.0.0",
            # One visible card, so Ray has nowhere to spread fractional deploys.
            # No fixture uses tensor/pipeline parallelism, so nothing needs a second.
            "CUDA_VISIBLE_DEVICES": "0",
        },
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def cleanup():
        log_file.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        subprocess.run(["ray", "stop", "--force"], check=False)

    try:
        deadline = time.time() + 120
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                if httpx.get(HEALTH_URL).status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(2)

        if not ready:
            tail = log_path.read_text()[-4000:] if log_path.exists() else "<no log>"
            cleanup()
            pytest.fail(f"Gateway failed to become ready within timeout.\nLog file: {log_path}\nLast 4KB:\n{tail}")

        yield tmp_dir
    finally:
        cleanup()


@pytest.fixture(scope="session")
def model_deployer(mship_cluster) -> _Deployer:
    return _Deployer(mship_cluster)


@pytest.fixture(scope="session")
def client(mship_cluster) -> OpenAI:
    return OpenAI(base_url=OPENAI_API_BASE, api_key="not-needed")
