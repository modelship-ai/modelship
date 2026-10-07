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
