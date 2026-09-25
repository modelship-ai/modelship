"""End-to-end deployment crash recovery: kill a backend engine subprocess
directly and verify Ray Serve respawns a working replica, until repeated deaths
retire the deployment."""

import time

import pytest

import openai
from modelship.infer.deploy_coordinator import _DEATHS_PER_REPLICA
from tests.conftest import serve_apps

_PING_PROMPT = [{"role": "user", "content": "hi"}]


def _find_and_kill_vllm_engine_core(deadline_s: float = 30) -> int:
    """SIGKILL the vLLM engine-core subprocess (titled `VLLM::EngineCore` via
    setproctitle) and return its PID."""
    import psutil

    end = time.time() + deadline_s
    while time.time() < end:
        for proc in psutil.process_iter():
            try:
                cmdline = " ".join(proc.cmdline())
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            if "VLLM::EngineCore" in cmdline:
                pid = proc.pid
                proc.kill()
                return pid
        time.sleep(1)
    pytest.fail("Could not find a vLLM EngineCore subprocess to kill within the deadline")


def _poll(predicate, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if predicate():
            return True
        time.sleep(1)
    return False


@pytest.mark.integration
@pytest.mark.vllm
class TestVllmCrashRecovery:
    """Kills the vLLM engine-core subprocess directly and verifies the replica recovers."""

    MODEL = "chat-capable"

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy(self.MODEL)

    def test_recovers_after_engine_core_crash(self, client):
        # Sanity: the deployment is actually serving before we kill anything.
        client.chat.completions.create(model=self.MODEL, messages=_PING_PROMPT, max_tokens=4)

        _find_and_kill_vllm_engine_core()

        def _request_succeeds() -> bool:
            try:
                client.chat.completions.create(model=self.MODEL, messages=_PING_PROMPT, max_tokens=4, timeout=15)
                return True
            except Exception:
                return False

        recovered = _poll(_request_succeeds, deadline_s=90)
        assert recovered, (
            "expected the deployment to recover (replica respawn) after its engine core crashed; requests kept failing"
        )


def _kill_llama_server(gguf: str, spared: set[int], deadline_s: float = 30) -> int:
    """SIGKILL the `llama serve` subprocess serving `gguf`, skipping pids in `spared`; returns its pid."""
    import psutil

    end = time.time() + deadline_s
    while time.time() < end:
        for proc in psutil.process_iter():
            try:
                cmdline = proc.cmdline()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            if cmdline[1:2] == ["serve"] and gguf in " ".join(cmdline) and proc.pid not in spared:
                proc.kill()
                return proc.pid
        time.sleep(1)
    pytest.fail(f"Could not find a new llama serve subprocess for {gguf} within the deadline")


@pytest.mark.integration
@pytest.mark.llama_server
class TestRepeatedBackendDeaths:
    MODEL = "chat-llama-server-plain"
    GGUF = "Qwen2.5-0.5B-Instruct"

    def test_a_deployment_whose_backend_keeps_dying_is_retired(self, client, model_deployer):
        model_deployer.deploy(self.MODEL)
        (app,) = (name for name in serve_apps() if name.startswith(f"modelship.{self.MODEL}-"))

        def request_succeeds() -> bool:
            try:
                client.chat.completions.create(model=self.MODEL, messages=_PING_PROMPT, max_tokens=4, timeout=15)
                return True
            except Exception:
                return False

        killed: set[int] = set()
        # num_replicas is 1, so the deploy coordinator retires the app at this many deaths
        for death in range(1, _DEATHS_PER_REPLICA + 1):
            assert _poll(request_succeeds, deadline_s=120), f"no working replica before death {death}"
            killed.add(_kill_llama_server(self.GGUF, killed))

        model_deployer.forget()
        assert _poll(lambda: app not in serve_apps(), deadline_s=120), "the deployment was not retired"
        # the effective config still lists the model, so the gateway answers 503
        no_retries = client.with_options(max_retries=0)
        for _ in range(20):
            with pytest.raises(openai.InternalServerError) as raised:
                no_retries.chat.completions.create(model=self.MODEL, messages=_PING_PROMPT, max_tokens=4)
            assert raised.value.status_code == 503
