import asyncio
import collections
import sys

import pytest
import ray
from ray import cloudpickle, serve
from ray.serve.config import RequestRouterConfig

from modelship.deploy.request_router import AffinityRouter
from modelship.deploy.serve_utils import local_ray_clusters

pytestmark = pytest.mark.integration

_MODEL_APP = "gw.m-0123456789"


@serve.deployment(
    num_replicas=3,
    max_ongoing_requests=5,
    ray_actor_options={"num_cpus": 0},
    request_router_config=RequestRouterConfig(request_router_class=AffinityRouter),
)
class _Model:
    async def generate(self, delay_s: float) -> str:
        await asyncio.sleep(delay_s)
        return serve.get_replica_context().replica_id.unique_id


@serve.deployment(num_replicas=2, max_ongoing_requests=100, ray_actor_options={"num_cpus": 0})
class _Gateway:
    def __init__(self):
        self._model = serve.get_app_handle(_MODEL_APP)

    async def call(self, key: str, delay_s: float) -> tuple[str, str]:
        replica = await self._model.generate.options(session_id=key).remote(delay_s)
        return serve.get_replica_context().replica_id.unique_id, replica


@pytest.fixture(scope="module")
def gateway():
    if running := local_ray_clusters():
        pytest.skip(f"a Ray node already runs on this machine (GCS at {', '.join(sorted(running))}); run this alone")
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False)
    try:
        serve.start(proxy_location="Disabled")
        serve.run(_Model.bind(), name=_MODEL_APP, route_prefix=None)
        yield serve.run(_Gateway.bind(), name="gateway", route_prefix=None)
    finally:
        serve.shutdown()
        ray.shutdown()
        cloudpickle.unregister_pickle_by_value(sys.modules[__name__])


def _calls(gateway, key: str, delay_s: float, count: int) -> list[tuple[str, str]]:
    responses = [gateway.call.remote(key, delay_s) for _ in range(count)]
    return [response.result(timeout_s=60) for response in responses]


def test_a_conversations_turns_stay_on_one_replica_through_either_gateway(gateway):
    seen = collections.defaultdict(set)
    gateways = set()
    for n in range(20):
        for _ in range(5):
            [(gateway_replica, replica)] = _calls(gateway, f"conversation-{n}", 0, 1)
            seen[n].add(replica)
            gateways.add(gateway_replica)

    assert all(len(replicas) == 1 for replicas in seen.values())
    assert len(gateways) == 2


def test_conversations_spread_over_the_replicas(gateway):
    firsts = collections.Counter(_calls(gateway, f"conversation-{n}", 0, 1)[0][1] for n in range(60))
    assert len(firsts) == 3 and min(firsts.values()) >= 10


def test_a_busy_conversation_spills_to_one_second_choice(gateway):
    [(_, first)] = _calls(gateway, "busy", 0, 1)
    served = collections.Counter(replica for _, replica in _calls(gateway, "busy", 2, 8))

    assert served[first] == max(served.values())
    assert len(served) == 2


def test_a_burst_beyond_every_replicas_cap_completes(gateway):
    responses = [gateway.call.remote(f"burst-{n}", 1) for n in range(30)]
    served = collections.Counter(response.result(timeout_s=60)[1] for response in responses)
    assert sum(served.values()) == 30
