"""End-to-end blue_green replace-strategy cutover: no dropped requests during
the swap, and the old deployment is fully torn down rather than left running
as a zombie."""

import concurrent.futures
import threading
import time

import pytest

from openai import OpenAI
from tests.conftest import serve_apps

_PING_PROMPT = [{"role": "user", "content": "hi"}]


def _hammer(
    client: OpenAI,
    model: str,
    stop: threading.Event,
    errors: list,
    *,
    messages: list[dict] | None = None,
    max_tokens: int = 256,
) -> None:
    """Keep one request in flight at a time until `stop` is set."""
    messages = messages if messages is not None else _PING_PROMPT
    while not stop.is_set():
        try:
            client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        except Exception as exc:
            # Surfaced via the shared list, not raised in the worker thread.
            errors.append(exc)


def _model_in_all_samples(client: OpenAI, model: str, samples: int = 20) -> bool:
    return all(model in {m.id for m in client.models.list().data} for _ in range(samples))


def _poll(predicate, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if predicate():
            return True
        time.sleep(1)
    return False


_BLUE_GREEN_MODEL = "blue-green-cutover"
_BLUE_GREEN_SOURCE = "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf"


def _blue_green_config(n_ctx: int) -> dict:
    # n_ctx forces a new fingerprint between "versions" without a new download; parallel
    # must cover the 2 hammer workers below, or a queued request can hang until client timeout.
    return {
        "name": _BLUE_GREEN_MODEL,
        "model": _BLUE_GREEN_SOURCE,
        "usecase": "generate",
        "loader": "llama_server",
        "num_cpus": 1,
        "llama_server_config": {"n_ctx": n_ctx, "parallel": 2},
    }


def _app_names_for(model_name: str) -> set[str]:
    """Live Serve app names currently routing `model_name` (app name is
    `modelship.<model_name>-<fingerprint>`)."""
    return {name for name in serve_apps() if name.startswith(f"modelship.{model_name}-")}


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.blue_green
class TestBlueGreenReplace:
    """--replace-strategy blue_green (the default) cuts a model's config change over
    atomically, with the old deployment fully torn down rather than left running."""

    def test_cutover_has_no_request_loss_and_leaves_no_zombie(self, client, model_deployer):
        model_deployer.deploy_raw([_blue_green_config(n_ctx=2048)], replace_strategy="blue_green")
        assert _poll(lambda: _model_in_all_samples(client, _BLUE_GREEN_MODEL), deadline_s=60), (
            "initial deployment did not become routable on all gateway replicas"
        )
        old_apps = _app_names_for(_BLUE_GREEN_MODEL)
        assert len(old_apps) == 1, f"expected exactly one deployment before cutover, saw {old_apps}"

        stop = threading.Event()
        errors: list[Exception] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_hammer, client, _BLUE_GREEN_MODEL, stop, errors, messages=_PING_PROMPT, max_tokens=5)
                for _ in range(2)
            ]
            try:
                # Blocks until the new deployment is live and the old one is dropped; hammering
                # concurrently exposes any gap where the model would 404 or 5xx.
                model_deployer.deploy_raw([_blue_green_config(n_ctx=4096)], replace_strategy="blue_green")
            finally:
                stop.set()
                concurrent.futures.wait(futures)

        assert not errors, f"requests failed during blue_green cutover: {errors[:3]}"

        new_apps = _app_names_for(_BLUE_GREEN_MODEL)
        assert len(new_apps) == 1, f"expected exactly one deployment after cutover, saw {new_apps}"
        assert new_apps != old_apps, "app name did not change even though the config did"
        assert not (old_apps & new_apps), "old deployment still present after cutover"
