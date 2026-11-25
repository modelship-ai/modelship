import hashlib
import random
from collections import Counter
from dataclasses import dataclass

from ray.serve._private.replica_result import ReplicaResult
from ray.serve.config import RequestRouterConfig
from ray.serve.request_router import LocalityMixin, PendingRequest, ReplicaID, RequestRouter, RunningReplica

from modelship import metrics
from modelship.utils.config_schema import parse_deployment_name


def _score(key: str, replica_id: ReplicaID) -> bytes:
    return hashlib.blake2b(f"{key}\0{replica_id.unique_id}".encode(), digest_size=8).digest()


class AffinityRouter(LocalityMixin, RequestRouter):
    def initialize_state(self, balance_factor: float = 1.25, spare_requests: int = 2) -> None:
        self._balance_factor = balance_factor
        self._spare_requests = spare_requests
        parsed = parse_deployment_name(self.app_name)
        self._model = parsed[1] if parsed else self.app_name

    async def choose_replicas(
        self, candidate_replicas: list[RunningReplica], pending_request: PendingRequest | None = None
    ) -> list[list[RunningReplica]]:
        if pending_request is not None:
            pending_request.routing_context.should_backoff = True
        if pending_request is None or not pending_request.metadata.session_id or not candidate_replicas:
            return self._two_random(pending_request)
        key = pending_request.metadata.session_id
        mean = sum(self._load(r) for r in candidate_replicas) / len(candidate_replicas)
        limit = self._balance_factor * mean + self._spare_requests
        ranked = sorted(candidate_replicas, key=lambda r: _score(key, r.replica_id), reverse=True)
        within = [r for r in ranked if self._load(r) <= limit]
        over = [r for r in ranked if self._load(r) > limit]
        return [[r] for r in within + over]

    def on_request_routed(self, pending_request: PendingRequest, replica_id: ReplicaID, result: ReplicaResult) -> None:
        key = pending_request.metadata.session_id
        if not key:
            choice = "no_key"
        elif replica_id == max(self.curr_replicas, key=lambda r: _score(key, r), default=None):
            choice = "first"
        else:
            choice = "spilled"
        metrics.AFFINITY_ROUTED_TOTAL.inc(tags={"model": self._model, "choice": choice})

    def _load(self, replica: RunningReplica) -> int:
        return self.replica_queue_len_cache.get(replica.replica_id) or 0

    def _two_random(self, pending_request: PendingRequest | None) -> list[list[RunningReplica]]:
        ids = list(self.apply_locality_routing(pending_request))
        return [[self.curr_replicas[i] for i in random.sample(ids, min(2, len(ids)))]]


class StickyRouter(LocalityMixin, RequestRouter):
    """A call to one of the *pins* methods with a session id goes to the replica its session's first such call went
    to, never another while that replica runs, waiting while it is busy. The first call picks the replica holding the
    fewest sessions, on the caller's node when one is there; a session whose replica is gone gets a new one. A call to
    one of the *unpins* methods goes to its session's replica when it has one, and unpins it. Every other call goes to
    any replica, on the caller's node when one is there."""

    def initialize_state(self, pins: list[str] | None = None, unpins: list[str] | None = None) -> None:
        self._pin_methods = frozenset(pins or ())
        self._unpin_methods = frozenset(unpins or ())
        # session id -> its replica
        self._pins: dict[str, ReplicaID] = {}

    async def choose_replicas(
        self, candidate_replicas: list[RunningReplica], pending_request: PendingRequest | None = None
    ) -> list[list[RunningReplica]]:
        if pending_request is not None:
            pending_request.routing_context.should_backoff = True
        key = pending_request.metadata.session_id if pending_request is not None else ""
        method = pending_request.metadata.call_method if pending_request is not None else ""
        follows = method in self._pin_methods or method in self._unpin_methods
        if not key or not follows or not candidate_replicas:
            return self._two_random(pending_request)
        pinned = self._pins.get(key)
        for replica in candidate_replicas:
            if replica.replica_id == pinned:
                return [[replica]]
        if method not in self._pin_methods:
            return self._two_random(pending_request)
        local = self.apply_locality_routing(pending_request)
        pool = [r for r in candidate_replicas if r.replica_id in local] or candidate_replicas
        held = Counter(self._pins.values())
        fewest = min(held[r.replica_id] for r in pool)
        choice = random.choice([r for r in pool if held[r.replica_id] == fewest])
        self._pins[key] = choice.replica_id
        return [[choice]]

    def on_request_routed(self, pending_request: PendingRequest, replica_id: ReplicaID, result: ReplicaResult) -> None:
        if pending_request.metadata.call_method in self._unpin_methods:
            self._pins.pop(pending_request.metadata.session_id, None)

    def _two_random(self, pending_request: PendingRequest | None) -> list[list[RunningReplica]]:
        ids = list(self.apply_locality_routing(pending_request))
        return [[self.curr_replicas[i] for i in random.sample(ids, min(2, len(ids)))]]


@dataclass(frozen=True)
class Routing:
    """The request router a deployment's app takes, and for `StickyRouter` the methods that pin a session and the
    ones that unpin it."""

    router: type[RequestRouter]
    pins: tuple[str, ...] = ()
    unpins: tuple[str, ...] = ()

    def config(self) -> RequestRouterConfig:
        kwargs = {"pins": list(self.pins), "unpins": list(self.unpins)} if self.pins or self.unpins else {}
        return RequestRouterConfig(request_router_class=self.router, request_router_kwargs=kwargs)  # type: ignore[arg-type]
