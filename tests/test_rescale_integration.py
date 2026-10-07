import concurrent.futures
import threading

import pytest
import yaml

from tests.conftest import run_on_cluster, serve_apps
from tests.test_blue_green_integration import _PING_PROMPT, _hammer, _model_in_all_samples, _poll

_MODEL = "rescale-in-place"
_SOURCE = "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf"


def _config(num_cpus: int = 1, **scaling) -> dict:
    return {
        "name": _MODEL,
        "model": _SOURCE,
        "usecase": "generate",
        "loader": "llama_server",
        "num_cpus": num_cpus,
        "llama_server_config": {"n_ctx": 2048, "parallel": 2},
        **scaling,
    }


def _cpu_share() -> int:
    """A CPU count per replica that two replicas fit in and a third doesn't."""
    cpus = int(run_on_cluster("print(int(ray.cluster_resources()['CPU']))"))
    share = (cpus - 1) // 2
    if cpus - 2 * share >= share:
        pytest.skip(f"two replicas must leave no room for a third; the cluster has {cpus} CPUs")
    return share


def _free_cpus() -> float:
    return float(run_on_cluster("print(ray.available_resources().get('CPU', 0))"))


def _running() -> dict[str, set[str]]:
    """The model's apps, each with the ids of its RUNNING replicas."""
    return {
        name: {
            replica["replica_id"]
            for deployment in app.get("deployments", {}).values()
            for replica in deployment.get("replicas", [])
            if replica.get("state") == "RUNNING"
        }
        for name, app in serve_apps().items()
        if name.startswith(f"modelship.{_MODEL}-")
    }


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.rescale
class TestInPlaceRescale:
    def test_a_replica_count_change_keeps_the_running_replicas_and_fails_no_request(self, client, model_deployer):
        model_deployer.deploy_raw([_config()])
        assert _poll(lambda: _model_in_all_samples(client, _MODEL), deadline_s=60)
        one = _running()
        ((app, original),) = one.items()
        assert len(original) == 1

        stop = threading.Event()
        errors: list[Exception] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_hammer, client, _MODEL, stop, errors, messages=_PING_PROMPT, max_tokens=5)
                for _ in range(2)
            ]
            try:
                model_deployer.deploy_raw([_config(num_replicas=2)])
                two = _running()
                model_deployer.deploy_raw([_config(autoscaling_config={"min_replicas": 1, "max_replicas": 2})])
                autoscaled = _running()
                model_deployer.deploy_raw([_config()])
                back_to_one = _running()
            finally:
                stop.set()
                concurrent.futures.wait(futures)

        assert not errors, f"requests failed during a rescale: {errors[:3]}"
        assert set(two) == {app}, f"the app changed on a replica-count change: {two}"
        assert original < two[app] and len(two[app]) == 2
        assert autoscaled == two
        assert set(back_to_one) == {app}
        assert len(back_to_one[app]) == 1 and back_to_one[app] < two[app]

    def test_a_cancelled_scale_up_goes_back_to_the_committed_replica_count(self, client, model_deployer, tmp_path):
        more_than_half_the_cpus = int(run_on_cluster("print(int(ray.cluster_resources()['CPU']))")) // 2 + 1
        config = _config(num_cpus=more_than_half_the_cpus)
        model_deployer.deploy_raw([config])
        assert _poll(lambda: _model_in_all_samples(client, _MODEL), deadline_s=60)
        before = _running()
        (app,) = before

        scale_up = tmp_path / "scale-up.yaml"
        scale_up.write_text(yaml.dump({"models": [{**config, "num_replicas": 2}]}))
        deploy = model_deployer.spawn("--config", str(scale_up), "--reconcile", log_name="rescale-pending")
        request_id = deploy.request_id()
        assert _poll(lambda: serve_apps()[app]["status"] == "DEPLOYING", deadline_s=60), "the scale-up never started"

        model_deployer.cancel(request_id, "--wait", log_name="cancel-rescale")
        assert f"Deploy {request_id} cancelled." in deploy.wait(expect_code=1)

        assert _poll(lambda: serve_apps()[app]["status"] == "RUNNING", deadline_s=60)
        assert _running() == before
        assert client.chat.completions.create(model=_MODEL, messages=_PING_PROMPT, max_tokens=5).choices

    def test_a_replica_count_changed_in_serve_is_put_back_by_the_same_config(self, model_deployer, tmp_path):
        model_deployer.deploy_raw([_config()])
        (app,) = _running()
        run_on_cluster(
            f"""
            from modelship.deploy.strategy import live_app, rescale_app
            from modelship.logging import serve_logging_config

            scaling = {{"num_replicas": 2, "autoscaling_config": None}}
            rescale_app({app!r}, live_app({app!r}), scaling, serve_logging_config())
            """
        )
        assert _poll(lambda: len(_running()[app]) == 2, deadline_s=120), "the app never got its second replica"

        same = tmp_path / "same.yaml"
        same.write_text(yaml.dump({"models": [_config()]}))
        log = model_deployer.run("--config", str(same), "--reconcile", log_name="same-config")

        assert f"rescale {_MODEL}: num_replicas: 2 -> 1 (Serve differs from the committed version)" in log
        assert len(_running()[app]) == 1


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.rescale
class TestScaleDownFirst:
    def test_a_scale_down_frees_the_room_a_new_model_needs(self, client, model_deployer):
        share = _cpu_share()
        model_deployer.deploy_raw([_config(num_cpus=share, num_replicas=2)])
        ((app, two),) = _running().items()
        assert _free_cpus() < share
        neighbour = {**_config(num_cpus=share), "name": "rescale-neighbour"}

        model_deployer.deploy_raw([_config(num_cpus=share), neighbour], replace_strategy="blue_green")

        after = _running()
        assert set(after) == {app}, f"the app changed on a replica-count change: {after}"
        assert len(after[app]) == 1 and after[app] < two
        assert _poll(lambda: _model_in_all_samples(client, neighbour["name"]), deadline_s=60)

    def test_a_replaced_app_is_scaled_down_to_make_room_for_its_replacement(self, client, model_deployer):
        share = _cpu_share()
        model_deployer.deploy_raw([_config(num_cpus=share, num_replicas=2)])
        (old_app,) = _running()
        assert _free_cpus() < share
        replacement = {**_config(num_cpus=share), "llama_server_config": {"n_ctx": 4096, "parallel": 2}}

        model_deployer.deploy_raw([replacement], replace_strategy="blue_green")

        ((app, replicas),) = _running().items()
        assert app != old_app
        assert len(replicas) == 1
        assert client.chat.completions.create(model=_MODEL, messages=_PING_PROMPT, max_tokens=5).choices
