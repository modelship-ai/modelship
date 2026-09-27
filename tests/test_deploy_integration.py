"""End-to-end `mship deploy` on the session cluster: deploy requests queued per gateway, all-or-nothing
outcomes, cancel, rollback, and a deploy coordinator restart."""

import time
from functools import partial
from pathlib import Path

import httpx
import pytest

from openai import OpenAI
from tests.conftest import OPENAI_API_BASE, run_on_cluster, serve_apps

_SOURCE = "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf"
# passes the source check, then fails to load in its replica
_BROKEN_SOURCE = "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:README.md"
# long enough for anything that deletes apps in the background to have acted
_PAST_GRACE_S = 15
_PING = [{"role": "user", "content": "hi"}]
_OTHER_GATEWAY = "other-gateway"
_FIRST_GATEWAY = "first-gateway"


def _flags(name: str, *, num_cpus: int = 1, source: str = _SOURCE) -> list[str]:
    return [
        "--name",
        name,
        "--model",
        source,
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


def _running(gateway: str, model: str) -> bool:
    apps = serve_apps()
    return any(apps[app]["status"] == "RUNNING" for app in _apps_for(gateway, model))


def _half_placeable(tmp_path) -> Path:
    config = tmp_path / "half-placeable.yaml"
    config.write_text(
        "models:\n"
        f"  - {{name: up-model, model: {_SOURCE!r}, usecase: generate, loader: llama_server, num_cpus: 1}}\n"
        f"  - {{name: never-model, model: {_SOURCE!r}, usecase: generate, loader: llama_server, num_cpus: 1000}}\n"
    )
    return config


def _status(model: str, base: str = OPENAI_API_BASE) -> int:
    return httpx.post(
        f"{base}/chat/completions", json={"model": model, "messages": _PING, "max_tokens": 1}, timeout=30
    ).status_code


def _answers(model: str, status: int, samples: int = 20) -> bool:
    """True iff every sampled chat completion for `model` gets `status`."""
    return all(_status(model) == status for _ in range(samples))


def _chat(client: OpenAI, model: str) -> str | None:
    return client.chat.completions.create(model=model, messages=_PING, max_tokens=5).choices[0].message.content


def _kill_replicas(app: str) -> int:
    """Kills every replica actor of *app* without letting Ray restart it; Serve starts replacements."""
    out = run_on_cluster(f"""
        killed = 0
        for actor in ray.util.list_named_actors(all_namespaces=True):
            if actor["name"].startswith("SERVE_REPLICA::{app}#"):
                ray.kill(ray.get_actor(actor["name"], namespace=actor["namespace"]), no_restart=True)
                killed += 1
        print(killed)
    """)
    return int(out.strip().splitlines()[-1])


def _delete_gateway(gateway: str) -> None:
    run_on_cluster(f"""
        from ray import serve
        for name in [n for n in serve.status().applications if n.split(".")[0] == {gateway!r}]:
            serve.delete(name)
    """)


@pytest.fixture(autouse=True)
def _empty_gateway(model_deployer):
    model_deployer.deploy()
    yield
    model_deployer.forget()


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.deploy
class TestQueue:
    def test_two_concurrent_deploys_both_land_and_stay(self, client, model_deployer):
        names = ("additive-a", "additive-b")
        deploys = [model_deployer.spawn(*_flags(name), log_name=name) for name in names]
        logs = [deploy.wait() for deploy in deploys]
        assert any("queued behind deploy" in log for log in logs)

        for name in names:
            assert _poll(partial(_listed_everywhere, client, name), deadline_s=60), (
                f"{name} did not become routable on every gateway replica"
            )
        time.sleep(_PAST_GRACE_S)
        for name in names:
            assert _apps_for("modelship", name), f"{name}'s app was deleted after its deploy succeeded"
            assert _chat(client, name) is not None

    def test_a_deploy_waits_behind_a_pending_one_on_its_gateway_but_not_on_another(self, model_deployer):
        pending = model_deployer.spawn(*_flags("blocker", num_cpus=1000), log_name="blocker")
        blocker_id = pending.request_id()
        assert _poll(lambda: _apps_for("modelship", "blocker"), deadline_s=120), "the blocker was never submitted"
        try:
            queued = model_deployer.spawn(*_flags("behind"), log_name="behind")
            queued.request_id()
            assert f"queued behind deploy {blocker_id}" in queued.log()
            model_deployer.run("--gateway-name", _OTHER_GATEWAY, *_flags("elsewhere"), log_name="elsewhere")
            assert not _apps_for("modelship", "behind"), "a deploy ran while another held its gateway's queue"

            model_deployer.stop(blocker_id, log_name="stop-blocker")
            pending.wait(expect_code=1)
            queued.wait()
            assert _apps_for("modelship", "behind")
        finally:
            _delete_gateway(_OTHER_GATEWAY)


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.deploy
class TestCancel:
    def test_a_cancelled_deploy_is_rolled_back_including_models_already_up(self, model_deployer, tmp_path):
        deploy = model_deployer.spawn("--config", str(_half_placeable(tmp_path)), log_name="half-placeable")
        request_id = deploy.request_id()
        assert _poll(partial(_running, "modelship", "up-model"), deadline_s=180), "the placeable model never came up"
        assert _answers("up-model", 404), "a model was routed before its deploy committed"

        log = model_deployer.stop(request_id, log_name="stop-half-placeable")
        assert f"Deploy {request_id} is being cancelled and rolled back." in log
        assert f"Deploy {request_id} cancelled: cancelled" in deploy.wait(expect_code=1)
        assert _poll(
            lambda: not _apps_for("modelship", "up-model") and not _apps_for("modelship", "never-model"),
            deadline_s=60,
        )

    def test_an_unknown_deploy_cannot_be_cancelled(self, model_deployer):
        log = model_deployer.stop("nosuchdeploy", log_name="stop-unknown", expect_code=1)
        assert "no queued or running deploy nosuchdeploy" in log


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.deploy
class TestFirstDeploy:
    def test_a_gateways_first_deploy_is_not_routed_before_it_commits(self, model_deployer, tmp_path):
        base = f"http://localhost:8000/{_FIRST_GATEWAY}/v1"
        deploy = model_deployer.spawn(
            "--gateway-name", _FIRST_GATEWAY, "--config", str(_half_placeable(tmp_path)), log_name="first-deploy"
        )
        request_id = deploy.request_id()
        try:
            assert _poll(partial(_running, _FIRST_GATEWAY, "up-model"), deadline_s=180), (
                "the placeable model never came up"
            )
            assert _poll(lambda: httpx.get(f"{base}/models", timeout=30).status_code == 200, deadline_s=60)
            assert not _poll(lambda: _status("up-model", base) != 404, deadline_s=10), (
                "a gateway's first deploy was routed before it committed"
            )
        finally:
            model_deployer.stop(request_id, log_name="stop-first-deploy")
            deploy.wait(expect_code=1)
            _delete_gateway(_FIRST_GATEWAY)


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.deploy
class TestClientGone:
    def test_a_killed_client_does_not_stop_its_deploy(self, client, model_deployer):
        deploy = model_deployer.spawn(*_flags("orphan-model"), log_name="orphan-model")
        deploy.request_id()
        deploy.kill()
        assert _poll(partial(_listed_everywhere, client, "orphan-model"), deadline_s=180), (
            "the deploy stopped with its client"
        )
        assert _chat(client, "orphan-model") is not None


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.deploy
class TestFailedChange:
    def test_a_failed_change_keeps_the_old_app_through_a_replica_restart(self, model_deployer):
        model_deployer.run(*_flags("survivor"), log_name="survivor")
        (old,) = _apps_for("modelship", "survivor")

        log = model_deployer.run(*_flags("survivor", source=_BROKEN_SOURCE), log_name="survivor-broken", expect_code=1)
        assert "failed" in log
        assert _apps_for("modelship", "survivor") == {old}, "the failed change was not rolled back"

        assert _kill_replicas(old) >= 1
        time.sleep(_PAST_GRACE_S)
        assert _apps_for("modelship", "survivor") == {old}, "the old app was deleted while its replica restarted"
        assert _poll(lambda: _status("survivor") == 200, deadline_s=180), "the old app never answered again"

    def test_a_failed_stop_start_leaves_a_gap_that_a_bare_deploy_fills(self, model_deployer):
        model_deployer.run(*_flags("gap-model"), log_name="gap-model")
        (old,) = _apps_for("modelship", "gap-model")

        model_deployer.run(
            *_flags("gap-model", source=_BROKEN_SOURCE),
            "--replace-strategy",
            "stop_start",
            log_name="gap-model-broken",
            expect_code=1,
        )
        assert not _apps_for("modelship", "gap-model")
        assert _poll(lambda: _answers("gap-model", 503), deadline_s=30), "a committed model with no app is not 503"

        model_deployer.run("--reconcile", log_name="gap-model-bare")
        assert _apps_for("modelship", "gap-model") == {old}
        assert _poll(lambda: _status("gap-model") == 200, deadline_s=60)


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
            assert not _apps_for("modelship", "shared-name")
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
            _delete_gateway(_OTHER_GATEWAY)


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.deploy
class TestDeployCoordinatorRestart:
    def test_a_restart_rolls_back_a_deploy_that_had_not_committed(self, model_deployer):
        deploy = model_deployer.spawn(*_flags("uncommitted", num_cpus=1000), log_name="uncommitted")
        request_id = deploy.request_id()
        assert _poll(lambda: _apps_for("modelship", "uncommitted"), deadline_s=120), "the model was never submitted"

        run_on_cluster("""
            ray.kill(ray.get_actor("modelship-deploy-coordinator", namespace="modelship"), no_restart=False)
        """)
        log = deploy.wait(expect_code=1)
        assert f"Deploy {request_id} was lost: the deploy coordinator restarted." in log
        assert _poll(lambda: not _apps_for("modelship", "uncommitted"), deadline_s=120), (
            "the restarted deploy coordinator did not roll the deploy back"
        )
