"""Cluster-wide bookkeeping for model deploys.

`DeployCoordinator` is a detached, named Ray actor on the head node, created by
the first caller and looked up by name afterwards. It holds what no single driver
or replica can:

- each gateway's deploy queue: requests run one at a time per gateway, each in its
  own worker actor (`deploy/worker.py`), and a gateway's version is committed only
  when one succeeds;
- each gateway's routing version, which the gateway coordinator routes by;
- one deploy lease per node, held by a replica for the duration of its load, so
  loads on one node run one at a time (the holder's side is `deploy_leases.py`);
- a per-deployment backend-death count, which fails a deploy whose model keeps dying;
- fatal init errors, reported by a replica and read back by the worker, which
  is how a permanently-broken model is told apart from a transient failure.

Requests and routing versions live in memory: after a restart every gateway with
model apps is rolled back to its committed version.
"""

import asyncio
import contextlib
import os
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import ray
from ray import serve
from ray.exceptions import RayActorError

from modelship.deploy.ledger import DeployRequest, Version, commit_version, read_versions
from modelship.logging import configure_logging, get_logger
from modelship.state import get_state_store, state_store_env_var
from modelship.utils import head_node_options, random_uuid
from modelship.utils.config_schema import parse_deployment_name
from modelship.utils.runtime_env import DEPLOY_COORDINATOR_ENV_VARS, build_env_vars

logger = get_logger("deploy_coordinator")

COORDINATOR_ACTOR_NAME = "modelship-deploy-coordinator"
COORDINATOR_NAMESPACE = "modelship"

LEASE_SECONDS = 30.0
RENEW_SECONDS = 10.0
POLL_SECONDS = 2.0
_REAP_INTERVAL_SECONDS = 1.0
_DEATHS_PER_REPLICA = 3
_SWITCH_TIMEOUT_ENV = "MSHIP_DEPLOY_SWITCH_TIMEOUT_S"
_CANCEL_GRACE_ENV = "MSHIP_DEPLOY_CANCEL_GRACE_S"
_DEFAULT_SWITCH_TIMEOUT_S = 60.0
_DEFAULT_CANCEL_GRACE_S = 30.0


class _Lease(NamedTuple):
    holder: str
    expires_at: float


class _Routing(NamedTuple):
    # changes on every switch and switch back, and never repeats
    seq: int
    # None: the gateway has no committed version
    models: list[dict] | None
    # model -> app, from models; empty without a committed version
    apps: dict[str, str]


@dataclass
class _Entry:
    request: DeployRequest
    done: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())
    # queued | applying | switching | committing | retiring | rolling_back
    state: str = "queued"
    cancelled: bool = False
    # set once the worker has been told of the cancel
    cancel_seen: bool = False
    worker: Any = None


def _seconds_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = 0.0
    if value <= 0:
        logger.warning("Ignoring %s=%r, not a positive number; using %s", name, raw, default)
        return default
    return value


def _routing(seq: int, models: list[dict] | None, gateway_name: str) -> _Routing:
    return _Routing(seq, models, Version(0, models).apps(gateway_name) if models is not None else {})


def _outcome(entry: _Entry, state: str, reason: str) -> dict:
    return {"id": entry.request.id, "state": state, "reason": reason, "models": {}, "version": None}


def _kill(handle) -> None:
    with contextlib.suppress(Exception):
        ray.kill(handle)


def _reconstructed() -> bool:
    try:
        context = ray.get_runtime_context()
        if context.get_actor_id() is None:
            return False
        # Ray's own implementation reads a deprecated property
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return context.was_current_actor_reconstructed
    except Exception:
        return False


@ray.remote(num_cpus=0)
class DeployCoordinator:
    """Cluster-wide deploy bookkeeping: per-gateway deploy queues and routing versions, per-node
    deploy leases, replica-death counts and fatal errors. A lease unrenewed for `LEASE_SECONDS`
    is freed, so a holder that died doesn't hold its node shut."""

    def __init__(self, startup_window: bool = True):
        # nothing else configures logging in this process
        configure_logging()
        self._store = get_state_store()
        self._fatal_errors: dict[str, str] = {}
        self._deaths: dict[str, int] = {}
        # deployment -> (death limit, last reason)
        self._last_deaths: dict[str, tuple[int, str]] = {}
        self._leases: dict[str, _Lease] = {}
        # holders of a crashed predecessor stop within one lease period
        window = startup_window or _reconstructed()
        self._grants_from = time.monotonic() + (LEASE_SECONDS if window else 0.0)
        self._reaper = asyncio.create_task(self._reap_forever())
        self._switch_timeout = _seconds_env(_SWITCH_TIMEOUT_ENV, _DEFAULT_SWITCH_TIMEOUT_S)
        self._cancel_grace = _seconds_env(_CANCEL_GRACE_ENV, _DEFAULT_CANCEL_GRACE_S)
        # queued and running requests by id, until their outcome is collected
        self._requests: dict[str, _Entry] = {}
        self._queues: dict[str, deque[_Entry]] = {}
        self._running: dict[str, _Entry] = {}
        self._drainers: dict[str, asyncio.Task] = {}
        self._routing: dict[str, _Routing] = {}
        # per gateway: held while a commit writes the store, so a rollback reads the version it lands
        self._version_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        # nanosecond clock, so a restarted deploy coordinator never repeats a routing seq
        self._seq = time.time_ns()
        self._recovery: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()

    async def acquire(self, key: str, holder: str) -> str | None:
        """None when granted, else what holds `key` up."""
        now = time.monotonic()
        if now < self._grants_from:
            return "lease service starting"
        lease = self._leases.get(key)
        if lease is not None:
            return f"held by {lease.holder}"
        self._leases[key] = _Lease(holder, now + LEASE_SECONDS)
        return None

    async def renew(self, key: str, holder: str) -> bool:
        lease = self._leases.get(key)
        if lease is None or lease.holder != holder:
            return False
        self._leases[key] = lease._replace(expires_at=time.monotonic() + LEASE_SECONDS)
        return True

    async def release(self, key: str, holder: str) -> None:
        lease = self._leases.get(key)
        if lease is not None and lease.holder == holder:
            del self._leases[key]

    async def _reap_forever(self) -> None:
        while True:
            await asyncio.sleep(_REAP_INTERVAL_SECONDS)
            self._reap(time.monotonic())

    def _reap(self, now: float) -> None:
        for key, lease in list(self._leases.items()):
            if lease.expires_at <= now:
                logger.warning("Deploy lease %s expired without release (holder %s)", key, lease.holder)
                del self._leases[key]

    async def submit(self, request: DeployRequest) -> dict:
        """Queues *request* on its gateway. Returns its id, the request it waits behind, and a ref to its outcome."""
        request.id = random_uuid()[:12]
        entry = _Entry(request)
        self._requests[request.id] = entry
        queue = self._queues.setdefault(request.gateway, deque())
        running = self._running.get(request.gateway)
        behind = queue[-1].request.id if queue else (running.request.id if running else None)
        queue.append(entry)
        if request.gateway not in self._drainers:
            self._drainers[request.gateway] = asyncio.create_task(self._drain(request.gateway))
        logger.info(
            "Deploy %s queued on gateway %s (%s, %s)", request.id, request.gateway, request.mode, request.strategy
        )
        return {"id": request.id, "behind": behind, "outcome": self._self().wait.remote(request.id)}

    async def wait(self, request_id: str) -> dict:
        """The request's outcome, once it has finished."""
        entry = self._requests[request_id]
        try:
            return await asyncio.shield(entry.done)
        finally:
            self._requests.pop(request_id, None)

    async def cancel(self, request_id: str) -> dict:
        """Drops a queued request, or rolls back a running one that hasn't committed. Returns whether it did,
        and a message saying what happened."""
        entry = self._requests.get(request_id)
        if entry is None or entry.done.done():
            return {"cancelled": False, "message": f"no queued or running deploy {request_id}"}
        if entry.state == "queued":
            self._queues[entry.request.gateway].remove(entry)
            entry.done.set_result(_outcome(entry, "cancelled", "cancelled before it started"))
            logger.info("Deploy %s cancelled before it started", request_id)
            return {"cancelled": True, "message": f"deploy {request_id} cancelled"}
        if entry.state in ("committing", "retiring"):
            return {"cancelled": False, "message": f"deploy {request_id} is already committed and can't be cancelled"}
        if not entry.cancelled:
            entry.cancelled = True
            logger.info("Cancelling deploy %s", request_id)
            self._spawn(self._kill_unless_seen(entry))
        return {"cancelled": True, "message": f"deploy {request_id} is being cancelled and rolled back"}

    async def is_cancelled(self, request_id: str) -> bool:
        entry = self._requests.get(request_id)
        if entry is None:
            return True
        if entry.cancelled:
            entry.cancel_seen = True
        return entry.cancelled

    async def switch(self, request_id: str, gateway_name: str, models: list[dict]) -> int:
        """Routes the gateway by *models* while the request confirms its replicas follow; returns the routing seq."""
        entry = self._running_entry(request_id, gateway_name)
        entry.state = "switching"
        self._routing[gateway_name] = _routing(self._next_seq(), models, gateway_name)
        logger.info("Deploy %s: switching gateway %s to its new models", request_id, gateway_name)
        return self._routing[gateway_name].seq

    async def reset_routing(self, gateway_name: str) -> tuple[int, list[dict] | None]:
        """Routes the gateway by its committed version again; returns the routing seq and that version's models."""
        async with self._version_locks[gateway_name]:
            committed = await self._committed(gateway_name)
        current = self._routing.get(gateway_name)
        if current is None or current.models != committed:
            self._routing[gateway_name] = _routing(self._next_seq(), committed, gateway_name)
        return self._routing[gateway_name].seq, committed

    async def commit(self, request_id: str, gateway_name: str, models: list[dict]) -> int | None:
        """Writes *models* as the gateway's next version; None, writing nothing, for a cancelled request."""
        entry = self._running_entry(request_id, gateway_name)
        if entry.cancelled:
            return None
        entry.state = "committing"
        async with self._version_locks[gateway_name]:
            version = await asyncio.to_thread(commit_version, self._store, gateway_name, models)
        entry.state = "retiring"
        logger.info("Deploy %s committed version %d of gateway %s", request_id, version.number, gateway_name)
        return version.number

    async def routing_versions(self, gateway_names: list[str]) -> dict[str, dict]:
        """Each gateway's routing seq and model -> app table; the table is empty without a committed version."""
        self._start_recovery()
        versions = {}
        for name in gateway_names:
            if name not in self._routing:
                committed = await self._committed(name)
                self._routing.setdefault(name, _routing(self._next_seq(), committed, name))
            routing = self._routing[name]
            versions[name] = {"seq": routing.seq, "apps": routing.apps}
        return versions

    def _running_entry(self, request_id: str, gateway_name: str) -> _Entry:
        entry = self._running.get(gateway_name)
        if entry is None or entry.request.id != request_id:
            raise ValueError(f"deploy {request_id} is not running on gateway {gateway_name}")
        if entry.state == "rolling_back":
            raise ValueError(f"deploy {request_id} is being rolled back")
        return entry

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _self(self):
        return ray.get_runtime_context().current_actor

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _committed(self, gateway_name: str) -> list[dict] | None:
        committed, _ = await asyncio.to_thread(read_versions, self._store, gateway_name)
        return committed.models if committed is not None else None

    async def _drain(self, gateway_name: str) -> None:
        """Runs the gateway's queued requests one at a time."""
        queue = self._queues[gateway_name]
        try:
            self._start_recovery()
            assert self._recovery is not None
            await asyncio.shield(self._recovery)
            while queue:
                entry = queue.popleft()
                self._running[gateway_name] = entry
                entry.state = "applying"
                try:
                    outcome = await self._apply(entry)
                except Exception as e:
                    logger.exception("Deploy %s failed", entry.request.id)
                    outcome = _outcome(entry, "failed", f"{type(e).__name__}: {e}")
                finally:
                    del self._running[gateway_name]
                entry.done.set_result(outcome)
        finally:
            del self._drainers[gateway_name]

    async def _apply(self, entry: _Entry) -> dict:
        from modelship.deploy.worker import create_worker

        gateway = entry.request.gateway
        committed = await self._committed(gateway)
        entry.worker = create_worker(self._self())
        try:
            return await entry.worker.run.remote(entry.request, committed, self._switch_timeout)
        except RayActorError as e:
            logger.warning(
                "Deploy %s's worker stopped while %s; rolling back gateway %s", entry.request.id, entry.state, gateway
            )
            switching = entry.state in ("switching", "committing")
            if entry.state not in ("committing", "retiring"):
                # refuses a switch or commit the dead worker sent before it died
                entry.state = "rolling_back"
            await self._roll_back(gateway, switching=switching)
            # after the rollback, which waits out a commit in flight
            if entry.state == "retiring":
                return _outcome(entry, "succeeded", "")
            if entry.cancelled:
                return _outcome(entry, "cancelled", "cancelled")
            return _outcome(entry, "failed", f"the deploy worker died: {e}")
        finally:
            _kill(entry.worker)

    async def _roll_back(self, gateway_name: str, switching: bool) -> None:
        """Routes the gateway back to its committed version and deletes the apps that version doesn't name."""
        from modelship.deploy.worker import create_worker

        worker = create_worker(self._self())
        try:
            await worker.roll_back.remote(gateway_name, switching, self._switch_timeout)
        except Exception:
            logger.exception("Could not roll back gateway %s; the next deploy to it deletes what's left", gateway_name)
        finally:
            _kill(worker)

    async def _kill_unless_seen(self, entry: _Entry) -> None:
        await asyncio.sleep(self._cancel_grace)
        if not entry.cancel_seen and not entry.done.done() and entry.worker is not None:
            logger.warning(
                "Deploy %s did not stop within %.0f s; killing its worker", entry.request.id, self._cancel_grace
            )
            _kill(entry.worker)

    def _start_recovery(self) -> None:
        if self._recovery is None:
            self._recovery = asyncio.create_task(self._recover())

    async def _recover(self) -> None:
        """Rolls every gateway with model apps back to its committed version."""
        try:
            apps = await asyncio.to_thread(lambda: list(serve.status().applications))
            gateways = sorted({parsed[0] for name in apps if (parsed := parse_deployment_name(name)) is not None})
            await asyncio.gather(*(self._roll_back(gateway, switching=True) for gateway in gateways))
        except Exception:
            logger.exception("Could not check for leftover deployments")

    async def report_replica_death(self, deployment_name: str, replica_ceiling: int, reason: str) -> None:
        """Count one backend death against `deployment_name`; past `_DEATHS_PER_REPLICA * replica_ceiling`
        a deploy bringing it up fails. Only a deploy resets the count. The reporting replica exits whatever
        this returns."""
        deaths = self._deaths.get(deployment_name, 0) + 1
        self._deaths[deployment_name] = deaths
        limit = _DEATHS_PER_REPLICA * max(replica_ceiling, 1)
        self._last_deaths[deployment_name] = (limit, reason)
        if deaths < limit:
            logger.warning("Replica death %d of %d for %s: %s", deaths, limit, deployment_name, reason)
        else:
            logger.error("%s's backend has died %d time(s); last: %s", deployment_name, deaths, reason)

    async def crash_looping(self, deployment_names: list[str]) -> dict[str, str]:
        """Each of *deployment_names* past its death limit, with its last death's reason."""
        return {
            name: self._last_deaths[name][1]
            for name in deployment_names
            if name in self._last_deaths and self._deaths.get(name, 0) >= self._last_deaths[name][0]
        }

    async def forget_deaths(self, deployment_names: list[str]) -> None:
        for name in deployment_names:
            self._deaths.pop(name, None)
            self._last_deaths.pop(name, None)

    def report_fatal_error(self, deployment_name: str, reason: str) -> None:
        self._fatal_errors[deployment_name] = reason

    def pop_fatal_error(self, deployment_name: str) -> str | None:
        return self._fatal_errors.pop(deployment_name, None)


def find_coordinator():
    """The deploy coordinator's handle, or None when nothing has created it yet."""
    try:
        return ray.get_actor(COORDINATOR_ACTOR_NAME, namespace=COORDINATOR_NAMESPACE)
    except ValueError:
        return None


def get_or_create_coordinator(startup_window: bool = True):
    """Return the cluster-wide deploy coordinator handle, creating it on the head node if absent.
    *startup_window* applies only when this call creates it; a restarted one always waits it out."""
    return DeployCoordinator.options(
        name=COORDINATOR_ACTOR_NAME,
        namespace=COORDINATOR_NAMESPACE,
        get_if_exists=True,
        lifetime="detached",
        num_cpus=0,
        max_restarts=-1,
        runtime_env={"env_vars": build_env_vars(DEPLOY_COORDINATOR_ENV_VARS) | state_store_env_var()},
        **head_node_options(),
    ).remote(startup_window)
