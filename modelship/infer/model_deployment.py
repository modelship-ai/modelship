import contextlib
import os
import platform
import subprocess
import sys
import textwrap
import time
import traceback
from collections.abc import AsyncGenerator
from typing import Any

from ray import serve

from modelship.infer.base_infer import BaseInfer
from modelship.infer.deploy_leases import DeployLeaseError, deploy_lease
from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, RawRequestProxy
from modelship.infer.sources import ModelDownloadError
from modelship.logging import configure_logging, get_logger
from modelship.metrics import (
    EMBEDDING_DURATION_SECONDS,
    GENERATION_DURATION_SECONDS,
    IMAGE_GENERATION_DURATION_SECONDS,
    MODEL_LOAD_DURATION_SECONDS,
    MODEL_LOAD_FAILURES_TOTAL,
    TRANSCRIPTION_DURATION_SECONDS,
    TTS_GENERATION_DURATION_SECONDS,
    stamp_gateway,
)
from modelship.openai.protocol import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageVariationRequest,
    ResponsesRequest,
    SpeechRequest,
    TranscriptionRequest,
    TranslationRequest,
)
from modelship.utils.accelerator import detect_accelerator
from modelship.utils.cache import reject_unset_cache_roots

logger = get_logger("infer.deployment")

# llama_server, stable_diffusion_cpp, and whispercpp all have Metal backends; sherpa_onnx is
# CPU-only; vllm/diffusers don't run on Darwin at all.
_DARWIN_UNSUPPORTED_LOADERS = {ModelLoader.vllm, ModelLoader.diffusers}


def _reject_unsupported_darwin_loader(config: ModelshipModelConfig) -> None:
    """Checked on the actor's platform, not the driver's."""
    if platform.system() != "Darwin":
        return
    if config.loader in _DARWIN_UNSUPPORTED_LOADERS:
        raise RuntimeError(
            f"loader={config.loader.value} is not supported on macOS/Apple Silicon (model '{config.name}'). "
            "Use loader: llama_server for GGUF models on Metal."
        )


# No loader offloads to AMD or Intel GPUs: vllm ships CUDA-only wheels here, and the
# llama.cpp/ggml binaries are built for CPU, CUDA and Metal.
_UNSUPPORTED_ACCELERATORS = {"rocm": "AMD (ROCm)", "xpu": "Intel (XPU)"}


def _reject_unsupported_accelerator(config: ModelshipModelConfig) -> None:
    """Checked on the actor's platform, not the driver's."""
    if config.num_gpus <= 0:
        return
    vendor = _UNSUPPORTED_ACCELERATORS.get(detect_accelerator())
    if vendor is not None:
        raise RuntimeError(
            f"this node's GPU is {vendor}, which no modelship loader can offload to "
            f"(model '{config.name}' asks for num_gpus={config.num_gpus}). "
            "Set num_gpus: 0 to run on CPU."
        )


def _reap_child_processes() -> None:
    """Kill any subprocesses still alive in this actor process.

    Some loaders (and vLLM's engine core process) fork helper subprocesses
    before constructors return. If init then raises (e.g. CUDA OOM during
    graph capture), those workers never get reaped — they reparent to PID 1
    and hold their full GPU allocation until manually killed.
    """
    try:
        import psutil

        children = psutil.Process().children(recursive=True)
        if not children:
            return
        logger.warning(
            "Reaping %d orphan subprocess(es): %s",
            len(children),
            [c.pid for c in children],
        )
        for c in children:
            with contextlib.suppress(psutil.NoSuchProcess):
                c.terminate()
        _, alive = psutil.wait_procs(children, timeout=5)
        for c in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                c.kill()
    except Exception:
        logger.exception("Failed to reap child subprocesses")


# Sidecar program: tracks the actor's descendants while the actor is alive,
# then SIGKILLs them once the actor's process disappears. Run as a fresh
# Python interpreter via `python -c` so it has no inherited imports beyond
# what it imports here. Survives `ray stop` because it's a plain OS
# subprocess in a new session, not a Ray-managed actor.
_ORPHAN_REAPER_SOURCE = textwrap.dedent(
    """
    import os, signal, sys, time
    import psutil

    parent_pid = int(sys.argv[1])
    self_pid = os.getpid()

    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        sys.exit(0)

    # Track (pid, create_time) so we don't kill an unrelated process that
    # happened to inherit a recycled PID after our descendant exited.
    tracked: dict[int, float] = {}
    while True:
        try:
            if not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE:
                break
            for c in parent.children(recursive=True):
                if c.pid == self_pid:
                    continue
                try:
                    tracked[c.pid] = c.create_time()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            break
        time.sleep(0.5)

    for pid, ctime in tracked.items():
        try:
            proc = psutil.Process(pid)
            if proc.create_time() != ctime:
                continue  # PID was recycled — not our process
            proc.kill()
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass
    """
).strip()


def _spawn_orphan_reaper() -> subprocess.Popen | None:
    """Fork a sidecar that SIGKILLs our descendants if we die ungracefully.

    Python signal handlers can't preempt C-extension code, so SIGTERM
    received during vLLM CUDA init is queued and never serviced before
    raylet escalates to SIGKILL — leaving Worker_TP* subprocesses
    orphaned with full GPU memory mapped. The sidecar runs in its own
    session and watches our PID from the kernel's perspective, so it
    outlives our death (graceful or not) and reaps the workers.
    """
    try:
        return subprocess.Popen(
            [sys.executable, "-c", _ORPHAN_REAPER_SOURCE, str(os.getpid())],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except Exception:
        logger.exception("Failed to spawn orphan reaper sidecar")
        return None


@serve.deployment
class ModelDeployment:
    async def __init__(self, config: ModelshipModelConfig):
        configure_logging()
        # MSHIP_GATEWAY_NAME is forwarded to this actor via runtime_env (see
        # actor_options), so the gateway tag is set without threading a param in.
        stamp_gateway(os.environ.get("MSHIP_GATEWAY_NAME", ""))
        self.config = config
        # Spawn before loader init so the sidecar exists even if init blocks
        # in C code and the actor gets SIGKILL'd before __init__ completes.
        self._orphan_reaper = _spawn_orphan_reaper()
        start = time.monotonic()
        self.infer: BaseInfer
        try:
            reject_unset_cache_roots()
            _reject_unsupported_darwin_loader(config)
            _reject_unsupported_accelerator(config)
            # Must run before the loader is constructed: preflight (which
            # needs the model file on disk) runs synchronously inside a
            # loader's own __init__, so `_resolved_path` has to be populated
            # ahead of that, not lazily inside `infer.start()`.
            await BaseInfer.ensure_downloaded(config)

            # Loading holds this node's lease; downloading must not, or a slow
            # fetch blocks every other replica on the node.
            async with deploy_lease(config.name):
                if config.loader == ModelLoader.vllm:
                    from modelship.infer.vllm.vllm_infer import VllmInfer

                    self.infer = VllmInfer(config)
                elif config.loader == ModelLoader.diffusers:
                    from modelship.infer.diffusers.diffusers_infer import DiffusersInfer

                    self.infer = DiffusersInfer(config)
                elif config.loader == ModelLoader.llama_server:
                    from modelship.infer.llama_server.llama_server_infer import LlamaServerInfer

                    self.infer = LlamaServerInfer(config)
                elif config.loader == ModelLoader.stable_diffusion_cpp:
                    from modelship.infer.stable_diffusion_cpp.stable_diffusion_cpp_infer import (
                        StableDiffusionCppInfer,
                    )

                    self.infer = StableDiffusionCppInfer(config)
                elif config.loader == ModelLoader.whispercpp:
                    from modelship.infer.whispercpp.whispercpp_infer import WhispercppInfer

                    self.infer = WhispercppInfer(config)
                elif config.loader == ModelLoader.sherpa_onnx:
                    from modelship.infer.sherpa_onnx.sherpa_onnx_infer import SherpaOnnxInfer

                    self.infer = SherpaOnnxInfer(config)

                await self.infer.start()
                await self.infer.warmup()
        except (ModelDownloadError, DeployLeaseError) as e:
            # Deliberately NOT reported to the coordinator as fatal (see the
            # except Exception branch below): a download or lease blip should
            # retry next pass, not permanently evict an otherwise-good model.
            MODEL_LOAD_FAILURES_TOTAL.inc(tags={"model": config.name, "loader": config.loader.value})
            self._graceful_teardown()
            logger.warning("Load failed for '%s', will retry next pass: %s", config.name, e)
            raise
        except Exception as e:
            MODEL_LOAD_FAILURES_TOTAL.inc(tags={"model": config.name, "loader": config.loader.value})
            self._graceful_teardown()

            logger.exception("Engine init failed for '%s'", config.name)
            tb = traceback.format_exc()
            err_msg = f"{config.loader.value} engine init failed for '{config.name}': {e}"
            try:
                from modelship.infer.deploy_coordinator import get_or_create_coordinator

                coordinator = get_or_create_coordinator()
                app_name = serve.get_replica_context().app_name
                await coordinator.report_fatal_error.remote(app_name, f"{err_msg}\n{tb}")
            except Exception:
                logger.exception("Failed to report fatal error to coordinator for %s", config.name)

            raise RuntimeError(err_msg) from e
        finally:
            MODEL_LOAD_DURATION_SECONDS.observe(
                time.monotonic() - start, tags={"model": config.name, "loader": config.loader.value}
            )

    def __del__(self):
        self._graceful_teardown()

    def _graceful_teardown(self) -> None:
        if infer := getattr(self, "infer", None):
            try:
                infer.shutdown()
            except Exception:
                logger.exception("Failed to shutdown infer for %s", self.config.name)
        _reap_child_processes()
        # Cleanup is done — let the sidecar exit. It would auto-die on actor
        # exit anyway (parent gone), but kill it explicitly so it doesn't sit
        # around for its next 0.5s poll tick.
        if reaper := getattr(self, "_orphan_reaper", None):
            with contextlib.suppress(Exception):
                reaper.terminate()

    @staticmethod
    def _set_request_id(request_id: str | None) -> None:
        from modelship.logging import request_id_var

        request_id_var.set(request_id)

    @staticmethod
    def _set_identity(identity: str | None) -> None:
        from modelship.logging import identity_var

        identity_var.set(identity)

    async def generate(
        self,
        request: ChatCompletionRequest,
        request_headers: dict[str, str],
        disconnect_registry: Any,
        request_id: str | None = None,
        identity: str | None = None,
    ):
        self._set_request_id(request_id)
        self._set_identity(identity)
        proxy = RawRequestProxy(disconnect_registry, request_headers, request_id, identity)
        start = time.monotonic()
        result = await self.infer.create_chat_completion(request, proxy)
        if isinstance(result, AsyncGenerator):
            # Streaming: tokens are produced lazily while we iterate, so observe
            # after the generator drains (try/finally also captures a mid-stream
            # client disconnect / cancellation).
            try:
                async for chunk in result:
                    yield chunk
            finally:
                GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        else:
            GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
            yield result

    async def respond(
        self,
        request: ResponsesRequest,
        request_headers: dict[str, str],
        disconnect_registry: Any,
        request_id: str | None = None,
        identity: str | None = None,
        response_id: str | None = None,
    ):
        self._set_request_id(request_id)
        self._set_identity(identity)
        proxy = RawRequestProxy(disconnect_registry, request_headers, request_id, identity)
        start = time.monotonic()
        result = await self.infer.create_response(request, proxy, response_id=response_id)
        if isinstance(result, AsyncGenerator):
            try:
                async for chunk in result:
                    yield chunk
            finally:
                GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        else:
            GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
            yield result

    async def embed(
        self,
        request: EmbeddingRequest,
        request_headers: dict[str, str],
        disconnect_registry: Any,
        request_id: str | None = None,
        identity: str | None = None,
    ):
        self._set_request_id(request_id)
        self._set_identity(identity)
        proxy = RawRequestProxy(disconnect_registry, request_headers, request_id, identity)
        start = time.monotonic()
        result = await self.infer.create_embedding(request, proxy)
        EMBEDDING_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def transcribe(
        self,
        audio_data: bytes,
        request: TranscriptionRequest,
        request_headers: dict[str, str],
        disconnect_registry: Any,
        request_id: str | None = None,
        identity: str | None = None,
    ):
        self._set_request_id(request_id)
        self._set_identity(identity)
        proxy = RawRequestProxy(disconnect_registry, request_headers, request_id, identity)
        start = time.monotonic()
        result = await self.infer.create_transcription(audio_data, request, proxy)
        TRANSCRIPTION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def translate(
        self,
        audio_data: bytes,
        request: TranslationRequest,
        request_headers: dict[str, str],
        disconnect_registry: Any,
        request_id: str | None = None,
        identity: str | None = None,
    ):
        self._set_request_id(request_id)
        self._set_identity(identity)
        proxy = RawRequestProxy(disconnect_registry, request_headers, request_id, identity)
        start = time.monotonic()
        result = await self.infer.create_translation(audio_data, request, proxy)
        TRANSCRIPTION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def speak(
        self,
        request: SpeechRequest,
        request_headers: dict[str, str],
        disconnect_registry: Any,
        request_id: str | None = None,
        identity: str | None = None,
    ):
        self._set_request_id(request_id)
        self._set_identity(identity)
        proxy = RawRequestProxy(disconnect_registry, request_headers, request_id, identity)
        start = time.monotonic()
        result = await self.infer.create_speech(request, proxy)
        TTS_GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def imagine(
        self,
        request: ImageGenerationRequest,
        request_headers: dict[str, str],
        disconnect_registry: Any,
        request_id: str | None = None,
        identity: str | None = None,
    ):
        self._set_request_id(request_id)
        self._set_identity(identity)
        proxy = RawRequestProxy(disconnect_registry, request_headers, request_id, identity)
        start = time.monotonic()
        result = await self.infer.create_image_generation(request, proxy)
        IMAGE_GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def edit_image(
        self,
        image_data: bytes,
        mask_data: bytes | None,
        request: ImageEditRequest,
        request_headers: dict[str, str],
        disconnect_registry: Any,
        request_id: str | None = None,
        identity: str | None = None,
    ):
        self._set_request_id(request_id)
        self._set_identity(identity)
        proxy = RawRequestProxy(disconnect_registry, request_headers, request_id, identity)
        start = time.monotonic()
        result = await self.infer.create_image_edit(image_data, mask_data, request, proxy)
        IMAGE_GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def vary_image(
        self,
        image_data: bytes,
        request: ImageVariationRequest,
        request_headers: dict[str, str],
        disconnect_registry: Any,
        request_id: str | None = None,
        identity: str | None = None,
    ):
        self._set_request_id(request_id)
        self._set_identity(identity)
        proxy = RawRequestProxy(disconnect_registry, request_headers, request_id, identity)
        start = time.monotonic()
        result = await self.infer.create_image_variation(image_data, request, proxy)
        IMAGE_GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result
