"""End-to-end `mship deploy` on the session cluster: additive deploys, models still coming up
when deploy exits, and two gateways side by side."""

import time
from functools import partial

import httpx
import pytest

from modelship.deploy.strategy import _POLL_SECONDS
from modelship.infer.gateway_coordinator import _UNUSED_GRACE_SECONDS
from openai import OpenAI
from tests.conftest import OPENAI_API_BASE, run_on_cluster, serve_apps

_SOURCE = "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf"
# the gateway coordinator deletes an app the effective config doesn't target once it has been unused this long
_PAST_GRACE_S = _UNUSED_GRACE_SECONDS + 5
_PING = [{"role": "user", "content": "hi"}]
_READYZ_URL = "http://localhost:8000/modelship/readyz"


def _flags(name: str, *, num_cpus: int = 1) -> list[str]:
    return [
        "--name",
        name,
        "--model",
        _SOURCE,
        "--usecase",
        "generate",
        "--loader",
        "llama_server",
        "--num-cpus",
        str(num_cpus),
    ]


def _poll(predicate, deadline_s: float, interval_s: float = 1.0) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def _apps_for(gateway: str, model: str) -> set[str]:
    return {name for name in serve_apps() if name.startswith(f"{gateway}.{model}-")}


def _listed_everywhere(client: OpenAI, model: str, samples: int = 20) -> bool:
    """True iff every sampled /v1/models call lists `model`; a stale gateway replica would omit it."""
    return all(model in {m.id for m in client.models.list().data} for _ in range(samples))


def _listed_nowhere(client: OpenAI, model: str, samples: int = 20) -> bool:
    return all(model not in {m.id for m in client.models.list().data} for _ in range(samples))


def _answers(model: str, status: int, samples: int = 20) -> bool:
    """True iff every sampled chat completion for `model` gets `status`."""
    return all(
        httpx.post(
            f"{OPENAI_API_BASE}/chat/completions", json={"model": model, "messages": _PING, "max_tokens": 1}, timeout=30
        ).status_code
        == status
        for _ in range(samples)
    )


def _pending_everywhere(model: str, samples: int = 20) -> bool:
    """True iff every sampled /readyz is 503 with `model` pending."""
    for _ in range(samples):
        resp = httpx.get(_READYZ_URL, timeout=10)
        if resp.status_code != 503 or model not in resp.json()["models_pending"]:
            return False
    return True


def _chat(client: OpenAI, model: str) -> str | None:
    return client.chat.completions.create(model=model, messages=_PING, max_tokens=5).choices[0].message.content


@pytest.fixture(autouse=True)
def _empty_gateway(model_deployer):
    model_deployer.deploy()
    yield
    model_deployer.forget()


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.deploy
class TestAdditiveDeploys:
    def test_two_concurrent_deploys_both_land_and_stay(self, client, model_deployer):
        names = ("additive-a", "additive-b")
        deploys = [model_deployer.spawn(*_flags(name), log_name=name) for name in names]
        for deploy in deploys:
            deploy.wait()

        for name in names:
            assert _poll(partial(_listed_everywhere, client, name), deadline_s=60), (
                f"{name} did not become routable on every gateway replica"
            )
        # a model missing from the effective config would lose its app to the gateway coordinator by now
        time.sleep(_PAST_GRACE_S)
        for name in names:
            assert _apps_for("modelship", name), f"{name}'s app was deleted after its deploy succeeded"
            assert _listed_everywhere(client, name)
            assert _chat(client, name) is not None


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.deploy
class TestModelsStillComingUp:
    def test_an_unplaceable_model_is_left_pending_and_answers_503(self, model_deployer):
        log = model_deployer.run(
            *_flags("pending-model", num_cpus=1000), "--deploy-timeout", "0", log_name="pending-model"
        )
        assert "Model 'pending-model' is still coming up" in log
        (app,) = _apps_for("modelship", "pending-model")
        assert serve_apps()[app]["status"] == "DEPLOYING"
        assert _poll(lambda: _answers("pending-model", 503), deadline_s=30), (
            "a configured model with nothing serving did not answer 503 on every gateway replica"
        )
        assert _poll(lambda: _pending_everywhere("pending-model"), deadline_s=30)

        model_deployer.deploy()
        assert not _apps_for("modelship", "pending-model")
        assert _poll(lambda: _answers("pending-model", 404), deadline_s=30)

    def test_a_model_another_deploy_removes_is_no_longer_waited_for(self, model_deployer):
        deploy = model_deployer.spawn(*_flags("removed-model", num_cpus=1000), log_name="removed-model")
        assert _poll(lambda: _apps_for("modelship", "removed-model"), deadline_s=120), "the model was never submitted"
        # the deploy only stops waiting on an app one of its polls has seen
        time.sleep(2 * _POLL_SECONDS + 1)
        model_deployer.deploy()
        log = deploy.wait(timeout=60)
        assert "Model 'removed-model' was removed before it came up" in log
        assert "will land on its own" not in log

    def test_a_model_still_loading_when_deploy_exits_is_routed_once_up(self, client, model_deployer):
        log = model_deployer.run(*_flags("late-model"), "--deploy-timeout", "0", log_name="late-model")
        assert "Model 'late-model' is still coming up" in log
        assert _poll(lambda: _listed_everywhere(client, "late-model"), deadline_s=120), (
            "a model that came up after its deploy exited was never routed"
        )
        assert _chat(client, "late-model") is not None


_OTHER_GATEWAY = "other-gateway"


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.deploy
class TestTwoGateways:
    def test_each_gateway_keeps_its_own_apps(self, client, model_deployer, tmp_path):
        other = OpenAI(base_url=f"http://localhost:8000/{_OTHER_GATEWAY}/v1", api_key="not-needed")
        empty_config = tmp_path / "empty-models.yaml"
        empty_config.write_text("models: []\n")
        try:
            model_deployer.run(*_flags("shared-name"), log_name="shared-name-modelship")
            model_deployer.run("--gateway-name", _OTHER_GATEWAY, *_flags("shared-name"), log_name="shared-name-other")
            for gateway_client in (client, other):
                assert _poll(partial(_listed_everywhere, gateway_client, "shared-name"), deadline_s=120)
            (theirs,) = _apps_for(_OTHER_GATEWAY, "shared-name")
            assert len(_apps_for("modelship", "shared-name")) == 1

            model_deployer.deploy()
            assert _poll(lambda: _listed_nowhere(client, "shared-name"), deadline_s=60)
            time.sleep(_PAST_GRACE_S)
            assert _apps_for(_OTHER_GATEWAY, "shared-name") == {theirs}, (
                "removing a model from one gateway touched another's"
            )
            assert _chat(other, "shared-name") is not None

            model_deployer.run(
                "--gateway-name",
                _OTHER_GATEWAY,
                "--config",
                str(empty_config),
                "--reconcile",
                "--replace-strategy",
                "stop_start",
                log_name="other-gateway-empty",
            )
            assert not _apps_for(_OTHER_GATEWAY, "shared-name")
        finally:
            run_on_cluster(f"""
                from ray import serve
                for name in [n for n in serve.status().applications if n.split(".")[0] == {_OTHER_GATEWAY!r}]:
                    serve.delete(name)
            """)
