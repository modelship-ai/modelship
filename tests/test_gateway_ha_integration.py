"""End-to-end multi-replica gateway routing consistency: a deployed/removed
model must converge on every gateway replica, not just the one a direct push
would have hit, and routing must survive a gateway coordinator restart."""

import concurrent.futures
import json
import threading
import time

import pytest

from modelship.infer.deploy_coordinator import COORDINATOR_NAMESPACE
from modelship.infer.gateway_coordinator import GATEWAY_COORDINATOR_ACTOR_NAME
from openai import OpenAI
from tests.conftest import run_on_cluster


def _model_in_all_samples(client: OpenAI, model: str, samples: int = 20) -> bool:
    """True iff `model` appears on every sampled /v1/models call — a stale
    replica would omit it on some."""
    return all(model in {m.id for m in client.models.list().data} for _ in range(samples))


def _model_in_no_samples(client: OpenAI, model: str, samples: int = 20) -> bool:
    return all(model not in {m.id for m in client.models.list().data} for _ in range(samples))


def _poll(predicate, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if predicate():
            return True
        time.sleep(1)
    return False


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.gateway_ha
class TestGatewayReplicaConsistency:
    """With 2 gateway replicas, a deployed model must become routable on both,
    and a removed one must stop routing on both."""

    def test_add_and_remove_propagate_to_all_replicas(self, client, model_deployer):
        # Warm both replicas (spread requests so each starts its watch loop).
        for _ in range(10):
            client.models.list()

        model_deployer.deploy("chat-llama-server-plain")
        assert _poll(lambda: _model_in_all_samples(client, "chat-llama-server-plain"), deadline_s=60), (
            "deployed model did not become routable on all gateway replicas"
        )
        completion = client.chat.completions.create(
            model="chat-llama-server-plain", messages=[{"role": "user", "content": "hi"}], max_tokens=5
        )
        assert completion.choices[0].message.content is not None

        # Reconcile to a different model — chat-llama-server-plain is removed everywhere.
        model_deployer.deploy("chat-llama-server")
        assert _poll(lambda: _model_in_no_samples(client, "chat-llama-server-plain"), deadline_s=60), (
            "removed model still routable on some gateway replica"
        )

        # Requests to the removed model now 404 on every replica — none route into
        # the torn-down deployment (which would surface as a 5xx, not a 404).
        import openai

        for _ in range(20):
            with pytest.raises(openai.NotFoundError):
                client.chat.completions.create(
                    model="chat-llama-server-plain", messages=[{"role": "user", "content": "hi"}], max_tokens=5
                )


_RESTART_GATEWAY_COORDINATOR = f"""
    import json, time
    coordinator = ray.get_actor({GATEWAY_COORDINATOR_ACTOR_NAME!r}, namespace={COORDINATOR_NAMESPACE!r})
    before = ray.get(coordinator.get_routing.remote("modelship"))
    ray.kill(coordinator, no_restart=False)
    # the kill is asynchronous; the old actor may answer a call or two first
    after = before
    deadline = time.monotonic() + 60
    while after["generation"] == before["generation"] and time.monotonic() < deadline:
        time.sleep(0.5)
        try:
            after = ray.get(coordinator.get_routing.remote("modelship"), timeout=10)
        except Exception:
            pass
    print(json.dumps({{"before": before, "after": after}}))
"""


def _hammer(client: OpenAI, model: str, stop: threading.Event, errors: list) -> None:
    while not stop.is_set():
        try:
            client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=5)
        except Exception as exc:
            errors.append(exc)


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.gateway_ha
class TestGatewayCoordinatorRestart:
    MODEL = "chat-llama-server-plain"

    def test_routing_survives_a_restart_and_later_changes_still_propagate(self, client, model_deployer):
        model_deployer.deploy(self.MODEL)
        assert _poll(lambda: _model_in_all_samples(client, self.MODEL), deadline_s=60)

        stop = threading.Event()
        errors: list[Exception] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_hammer, client, self.MODEL, stop, errors) for _ in range(2)]
            try:
                routing = json.loads(run_on_cluster(_RESTART_GATEWAY_COORDINATOR))
                # gateway replicas re-resolve the restarted actor and resync within this
                time.sleep(10)
            finally:
                stop.set()
                concurrent.futures.wait(futures)

        assert routing["after"]["generation"] > routing["before"]["generation"], (
            "the gateway coordinator never restarted"
        )
        assert routing["after"]["models"] == routing["before"]["models"], "the restarted table differs"
        assert not errors, f"requests failed across the gateway coordinator restart: {errors[:3]}"
        assert _model_in_all_samples(client, self.MODEL)

        model_deployer.deploy()
        assert _poll(lambda: _model_in_no_samples(client, self.MODEL), deadline_s=60), (
            "a removal after the restart did not reach every gateway replica"
        )
