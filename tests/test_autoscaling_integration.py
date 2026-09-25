"""End-to-end check that autoscaling_config actually drives Ray Serve replica
counts, exercised via the llama_server loader."""

import concurrent.futures
import threading
import time

import pytest

from openai import OpenAI
from tests.conftest import serve_apps


def _running_replicas(model_name: str) -> int:
    """Counts RUNNING replicas of the deployment serving `model_name`; app names are
    `modelship.<model_name>-<fingerprint>`, matched by prefix."""
    for app_name, app in serve_apps().items():
        if app_name.startswith(f"modelship.{model_name}-"):
            for dep in app.get("deployments", {}).values():
                return sum(1 for r in dep.get("replicas", []) if r.get("state") == "RUNNING")
    return 0


def _wait_for_replicas(model_name: str, predicate, deadline_s: float) -> int:
    """Poll replica count until `predicate(count)` holds or the deadline passes.
    Returns the last observed count either way (caller asserts)."""
    end = time.time() + deadline_s
    count = _running_replicas(model_name)
    while time.time() < end:
        count = _running_replicas(model_name)
        if predicate(count):
            return count
        time.sleep(2)
    return count


_LOAD_PROMPT = [{"role": "user", "content": "Write a long, detailed story about a curious robot."}]


def _hammer(
    client: OpenAI,
    model: str,
    stop: threading.Event,
    errors: list,
    *,
    messages: list[dict] | None = None,
    max_tokens: int = 256,
) -> None:
    """Keeps one request in flight until `stop` is set; run several concurrently to push
    load past the autoscaler's per-replica setpoint, or pass a cheap prompt for liveness only."""
    messages = messages if messages is not None else _LOAD_PROMPT
    while not stop.is_set():
        try:
            client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        except Exception as exc:
            # Surfaced via the shared list, not raised in the worker thread.
            errors.append(exc)


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.autoscaling
class TestAutoscaling:
    """Replicas scale out under sustained concurrent load (bounded by max_replicas) and
    scale back to min_replicas once the load stops."""

    MODEL = "autoscale-llama"

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy(self.MODEL)

    def test_scales_out_under_load_then_back_to_min(self, client):
        # Idle baseline: the deployment sits at min_replicas (1).
        baseline = _wait_for_replicas(self.MODEL, lambda n: n == 1, deadline_s=60)
        assert baseline == 1, f"expected to start at min_replicas=1, saw {baseline}"

        stop = threading.Event()
        errors: list[Exception] = []
        # 8 concurrent in-flight requests vs target_ongoing_requests=1 asks the
        # autoscaler for ~8 replicas, capped at max_replicas=3.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for _ in range(8):
                pool.submit(_hammer, client, self.MODEL, stop, errors)
            try:
                # Autoscaler needs a look-back window of load metrics; allow generous time.
                peak = _wait_for_replicas(self.MODEL, lambda n: n > 1, deadline_s=120)
            finally:
                stop.set()

        assert peak > 1, f"expected scale-out under load, replicas stayed at {peak}"
        assert peak <= 3, f"replicas {peak} exceeded max_replicas=3"
        assert not errors, f"load requests errored during scale-out: {errors[:3]}"

        # Load stopped: scale back in to min_replicas within the downscale window + slack.
        settled = _wait_for_replicas(self.MODEL, lambda n: n == 1, deadline_s=180)
        assert settled == 1, f"expected scale-in to min_replicas=1 after load, saw {settled}"


_SCALE_TO_ZERO_MODEL = "scale-to-zero"
_PING_PROMPT = [{"role": "user", "content": "hi"}]


def _scale_to_zero_config(n_ctx: int) -> dict:
    # min_replicas 0 with no initial_replicas deploys at zero replicas; n_ctx changes the fingerprint.
    return {
        "name": _SCALE_TO_ZERO_MODEL,
        "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_cpus": 1,
        "llama_server_config": {"n_ctx": n_ctx},
        "autoscaling_config": {"min_replicas": 0, "max_replicas": 1, "downscale_delay_s": 5},
    }


def _scale_to_zero_apps() -> set[str]:
    return {name for name in serve_apps() if name.startswith(f"modelship.{_SCALE_TO_ZERO_MODEL}-")}


def _listed(client: OpenAI, model: str) -> bool:
    return model in {m.id for m in client.models.list().data}


def _poll(predicate, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if predicate():
            return True
        time.sleep(1)
    return False


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.autoscaling
class TestScaleToZero:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy_raw([_scale_to_zero_config(n_ctx=2048)])

    def _ping(self, client: OpenAI) -> None:
        client.chat.completions.create(model=_SCALE_TO_ZERO_MODEL, messages=_PING_PROMPT, max_tokens=4)

    def test_a_model_deployed_at_zero_replicas_answers(self, client):
        assert _running_replicas(_SCALE_TO_ZERO_MODEL) == 0
        assert _poll(lambda: _listed(client, _SCALE_TO_ZERO_MODEL), deadline_s=30), (
            "the gateway does not list a model at zero replicas"
        )
        self._ping(client)

    def test_a_model_scaled_back_to_zero_answers_again(self, client):
        settled = _wait_for_replicas(_SCALE_TO_ZERO_MODEL, lambda n: n == 0, deadline_s=120)
        assert settled == 0, f"expected scale-in to zero replicas, saw {settled}"
        self._ping(client)

    def test_a_changed_model_at_zero_replicas_retires_the_old_app(self, client, model_deployer):
        self._ping(client)
        old = _scale_to_zero_apps()
        model_deployer.deploy_raw([_scale_to_zero_config(n_ctx=4096)], replace_strategy="blue_green")
        assert _poll(lambda: not (old & _scale_to_zero_apps()), deadline_s=60), "the old app was not retired"
        self._ping(client)
