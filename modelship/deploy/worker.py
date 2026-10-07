"""One deploy request's Serve work, in its own actor: the steps of the diff the deploy coordinator builds for
it, then switch and commit, or roll back.

The deploy coordinator runs one request per gateway at a time, so each app of a gateway that its committed
version doesn't name belongs to the running request, or was left behind by one.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

import ray
from ray import serve
from ray.serve.schema import ApplicationStatus, ApplicationStatusOverview

from modelship.deploy.actor_options import build_cache_env_vars, build_deployment_options, total_gpu_reservation
from modelship.deploy.config import resolve_all_model_sources
from modelship.deploy.diff import Diff, Rescale
from modelship.deploy.ledger import DeployRequest, to_config
from modelship.deploy.strategy import live_app, rescale_app, submit_app
from modelship.infer.infer_config import ModelshipConfig, ModelshipModelConfig
from modelship.logging import configure_logging, get_logger, serve_logging_config
from modelship.metrics import DEPLOY_DURATION_SECONDS, DEPLOY_MODELS_CHANGED_TOTAL
from modelship.state import state_store_env_var
from modelship.utils import head_node_options
from modelship.utils.runtime_env import cluster_env_vars

logger = get_logger("deploy")

POLL_SECONDS = 2.0
# a request fails once Serve's status has been unreadable this long; a rollback keeps reading
_UNREADABLE_LIMIT_S = 30.0
_MAX_TRANSIENT_FAILURES = 3
_RETRY_SLEEP_S = 2.0
_PENDING_LOG_EVERY_N_POLLS = 30
_RPC_TIMEOUT_S = 10.0
# a model's result while its action runs
_STARTS_AS = {
    "add": "coming up",
    "replace": "coming up",
    "redeploy": "coming up",
    "rescale": "rescaling",
    "keep": "unchanged",
}


class RequestFailedError(Exception):
    """Ends a request, which is then rolled back; the message is the reason."""


class RequestCancelledError(Exception):
    pass


def read_until_readable() -> dict[str, ApplicationStatusOverview]:
    """Serve's apps, retrying without limit."""
    failing = False
    while True:
        try:
            return dict(serve.status().applications)
        except Exception:
            if not failing:
                logger.warning("Could not read Serve's status; retrying", exc_info=True)
            failing = True
            time.sleep(POLL_SECONDS)


def delete_apps(names: list[str], read: Callable[[], dict[str, ApplicationStatusOverview]]) -> None:
    """Deletes *names* without blocking on Serve, then polls *read* until all are gone."""
    left = set(names)
    while True:
        left &= set(apps := read())
        if not left:
            return
        for name in sorted(left):
            if apps[name].status != ApplicationStatus.DELETING:
                try:
                    serve.delete(name, _blocking=False)
                except Exception:
                    logger.warning("Could not delete %s; retrying", name, exc_info=True)
        time.sleep(POLL_SECONDS)


class DeployLedger:
    """The deploy coordinator calls a worker makes."""

    def __init__(self, handle):
        self._handle = handle

    def _call(self, method: str, *args) -> Any:
        # no timeout: a call can outlast a store write, and the deploy coordinator's death fails it
        return ray.get(getattr(self._handle, method).remote(*args))

    def cancelled(self, request_id: str) -> bool:
        return self._call("is_cancelled", request_id)

    def rolling_back(self, request_id: str, gateway_name: str) -> None:
        self._call("rolling_back", request_id, gateway_name)

    def crash_looping(self, apps: list[str]) -> dict[str, str]:
        return self._call("crash_looping", apps)

    def pop_fatal_error(self, app: str) -> str | None:
        return self._call("pop_fatal_error", app)

    def forget_deaths(self, apps: list[str]) -> None:
        self._call("forget_deaths", apps)

    def diff(self, request_id: str, gateway_name: str) -> Diff | None:
        return self._call("diff", request_id, gateway_name)

    def rollback_diff(self, gateway_name: str) -> Diff | None:
        return self._call("rollback_diff", gateway_name)

    def switch(self, request_id: str, gateway_name: str, models: list[dict]) -> int:
        return self._call("switch", request_id, gateway_name, models)

    def reset_routing(self, gateway_name: str) -> int:
        return self._call("reset_routing", gateway_name)

    def commit(self, request_id: str, gateway_name: str, models: list[dict]) -> int | None:
        return self._call("commit", request_id, gateway_name, models)


class GatewayReplicas:
    """Asks the gateway coordinator whether a gateway's replicas hold a routing version."""

    def wait_switched(self, gateway_name: str, routing: int, timeout: float) -> bool:
        from modelship.infer.gateway_coordinator import get_or_create_gateway_coordinator

        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            try:
                coordinator: Any = get_or_create_gateway_coordinator()
                return ray.get(
                    coordinator.wait_switched.remote(gateway_name, routing, left), timeout=left + _RPC_TIMEOUT_S
                )
            except Exception:
                logger.warning("Could not ask the gateway coordinator about gateway %s; retrying", gateway_name)
                time.sleep(1.0)
        return False


def roll_back(
    ledger: DeployLedger,
    replicas: GatewayReplicas,
    gateway_name: str,
    switching: bool,
    switch_timeout: float,
) -> list[str]:
    """Routes the gateway back to its committed version, deletes each of its apps that version doesn't name,
    then gives the apps it names that version's replica-count fields."""
    routing = ledger.reset_routing(gateway_name)
    if switching and not replicas.wait_switched(gateway_name, routing, switch_timeout):
        logger.warning(
            "Gateway %s did not switch back within %.0f s; deleting its new apps", gateway_name, switch_timeout
        )
    failing = False
    while (diff := ledger.rollback_diff(gateway_name)) is None:
        if not failing:
            logger.warning("Could not read Serve's status; retrying")
        failing = True
        time.sleep(POLL_SECONDS)
    if diff.leftovers:
        logger.info("Rolling back gateway %s: deleting %s", gateway_name, ", ".join(diff.leftovers))
    delete_apps(diff.leftovers, read_until_readable)
    restore_scaling(gateway_name, diff.scale_downs + diff.scale_ups)
    return diff.leftovers


def restore_scaling(gateway_name: str, rescales: list[Rescale]) -> None:
    """Re-submits each of *rescales*' apps with its replica-count fields; an app that keeps failing is given
    up on after `_UNREADABLE_LIMIT_S`."""
    left = {rescale.app: rescale for rescale in rescales}
    deadline = time.monotonic() + _UNREADABLE_LIMIT_S
    while left:
        for name, rescale in list(left.items()):
            try:
                live = live_app(name)
                if live is not None:
                    logger.info("Rolling back gateway %s: rescaling %s", gateway_name, name)
                    rescale_app(name, live, rescale.scaling, serve_logging_config())
            except Exception:
                if time.monotonic() < deadline:
                    continue
                logger.exception("Could not give %s its committed replica-count fields", name)
            del left[name]
        if left:
            time.sleep(POLL_SECONDS)


@dataclass
class _Add:
    config: ModelshipModelConfig
    failures: int = 0
    last_error: str = ""
    # set while waiting out the backoff before being submitted again
    retry_at: float | None = None
    # set once Serve reports the app, not being deleted
    seen: bool = False


class Run:
    """Applies one request; a failure or a cancel rolls it back."""

    def __init__(
        self,
        request: DeployRequest,
        ledger: DeployLedger,
        replicas: GatewayReplicas,
        switch_timeout: float,
    ):
        self._request = request
        self._ledger = ledger
        self._replicas = replicas
        self._switch_timeout = switch_timeout
        self._switching = False
        self._unreadable_since: float | None = None
        self._results: dict[str, str] = {}
        self._lines: list[str] = []
        self._version: int | None = None

    def execute(self) -> dict:
        request = self._request
        started = time.monotonic()
        tags = {"gateway": request.gateway}
        try:
            diff = self._apply()
        except RequestCancelledError:
            logger.warning("Deploy %s cancelled; rolling back", request.id)
            return self._rolled_back("cancelled", "cancelled")
        except RequestFailedError as e:
            logger.error("Deploy %s failed: %s; rolling back", request.id, e)
            DEPLOY_MODELS_CHANGED_TOTAL.inc(1, tags={**tags, "action": "fail"})
            return self._rolled_back("failed", str(e))
        except Exception as e:
            logger.exception("Deploy %s failed; rolling back", request.id)
            DEPLOY_MODELS_CHANGED_TOTAL.inc(1, tags={**tags, "action": "fail"})
            return self._rolled_back("failed", f"{type(e).__name__}: {e}")

        if request.strategy == "blue_green":
            delete_apps(diff.retired, read_until_readable)
        rescaled = sum(action == "rescale" for action in diff.actions.values())
        for action, count in (("add", len(diff.deploys)), ("remove", len(diff.retired)), ("rescale", rescaled)):
            if count:
                DEPLOY_MODELS_CHANGED_TOTAL.inc(count, tags={**tags, "action": action})
        DEPLOY_DURATION_SECONDS.observe(time.monotonic() - started, tags=tags)
        logger.info("Deploy %s succeeded", request.id)
        return self._outcome("succeeded", "")

    def _apply(self) -> Diff:
        request = self._request
        diff = self._diff()
        self._lines = diff.lines()
        for model, action in diff.actions.items():
            if action != "remove":
                self._results[model] = _STARTS_AS[action]
        self._log(diff)

        if diff.deploys:
            try:
                resolve_all_model_sources(ModelshipConfig.model_construct(models=diff.deploys))
            except Exception as e:
                raise RequestFailedError(str(e)) from e
        delete_apps(diff.leftovers, self._read)
        self._rescale(diff.scale_downs)
        if request.strategy == "stop_start":
            delete_apps(diff.retired, self._read)
        self._bring_up(diff.deploys)
        self._rescale(diff.scale_ups)
        if diff.commits:
            assert diff.models is not None
            self._switch_and_commit(diff.models)
        for model, action in diff.actions.items():
            if action == "remove":
                self._results[model] = "removed"
        return diff

    def _diff(self) -> Diff:
        """The request's diff, asked of the deploy coordinator until it can read Serve."""
        request = self._request
        while True:
            if self._ledger.cancelled(request.id):
                raise RequestCancelledError()
            diff = self._ledger.diff(request.id, request.gateway)
            if diff is not None:
                self._unreadable_since = None
                return diff
            self._unreadable()
            time.sleep(POLL_SECONDS)

    def _log(self, diff: Diff) -> None:
        request = self._request
        logger.info("Deploy %s on gateway %s (%s, %s):", request.id, request.gateway, request.mode, request.strategy)
        for line in self._lines or ["nothing to do"]:
            logger.info("  %s", line)
        if not diff.models:
            return
        # Log-only and optimistic: fractions can sum under the GPU total yet not pack.
        demand = sum(
            (m.autoscaling_config.max_replicas if m.autoscaling_config else m.num_replicas)
            * total_gpu_reservation(build_deployment_options(m, request.env))
            for m in to_config(diff.models).models
        )
        cluster_gpus = ray.cluster_resources().get("GPU", 0)
        if demand > cluster_gpus:
            logger.warning(
                "Gateway %s's models need at least %.2f GPU(s) at full scale; cluster has %.2f total. "
                "Deploys exceeding available capacity pend until more nodes join.",
                request.gateway,
                demand,
                cluster_gpus,
            )

    def _bring_up(self, adds: list[ModelshipModelConfig]) -> None:
        """Submits *adds* and polls until every one is RUNNING, retrying a failed deploy."""
        gateway = self._request.gateway
        pending = {c.deployment_name(gateway): _Add(c) for c in adds}
        if not pending:
            return
        self._ledger.forget_deaths(list(pending))
        for name, item in pending.items():
            self._submit(name, item)

        polls = 0
        statuses: dict[str, ApplicationStatusOverview] = {}
        while pending:
            time.sleep(POLL_SECONDS)
            polls += 1
            now = time.monotonic()
            for name, item in pending.items():
                if item.retry_at is not None and now >= item.retry_at:
                    item.retry_at = None
                    self._submit(name, item)
            if (read := self._try_read()) is None:
                continue
            statuses = read
            crashing = self._ledger.crash_looping(list(pending))
            for name, item in list(pending.items()):
                if name in crashing:
                    self._fail(item.config.name, f"its backend keeps dying: {crashing[name]}")
                if item.retry_at is not None:
                    continue
                app = statuses.get(name)
                if app is None or app.status == ApplicationStatus.DELETING:
                    if item.seen:
                        self._fail(item.config.name, f"deployment {name} was deleted")
                    continue
                item.seen = True
                if app.status == ApplicationStatus.RUNNING:
                    logger.info("Model ready: %s (deployment: %s)", item.config.name, name)
                    self._results[item.config.name] = "up"
                    del pending[name]
                # UNHEALTHY follows RUNNING while Serve replaces a replica, so it stays pending
                elif app.status == ApplicationStatus.DEPLOY_FAILED:
                    self._failed_attempt(name, item, app.message)
            if pending and (polls == 1 or polls % _PENDING_LOG_EVERY_N_POLLS == 0):
                _log_pending({name: item.config.name for name, item in pending.items()}, statuses)

    def _rescale(self, rescales: list[Rescale]) -> None:
        """Re-submits each live app with its new replica-count fields and polls until every one is RUNNING."""
        pending = {rescale.app: rescale for rescale in rescales}
        if not pending:
            return
        self._ledger.forget_deaths(list(pending))
        for name, rescale in pending.items():
            # one left by a replica that failed before this request
            self._ledger.pop_fatal_error(name)
            try:
                live = live_app(name)
                if live is not None:
                    rescale_app(name, live, rescale.scaling, serve_logging_config())
            except Exception as e:
                self._fail(rescale.model, f"{type(e).__name__}: {e}")
            if live is None:
                self._fail(rescale.model, f"deployment {name} was deleted")

        polls = 0
        while pending:
            time.sleep(POLL_SECONDS)
            polls += 1
            if (statuses := self._try_read()) is None:
                continue
            crashing = self._ledger.crash_looping(list(pending))
            for name, rescale in list(pending.items()):
                if name in crashing:
                    self._fail(rescale.model, f"its backend keeps dying: {crashing[name]}")
                # Serve retries a new replica without limit and never reports DEPLOY_FAILED
                if (fatal := self._ledger.pop_fatal_error(name)) is not None:
                    self._fail(rescale.model, fatal)
                app = statuses.get(name)
                if app is None or app.status == ApplicationStatus.DELETING:
                    self._fail(rescale.model, f"deployment {name} was deleted")
                if app.status == ApplicationStatus.RUNNING:
                    logger.info("Model rescaled: %s (deployment: %s)", rescale.model, name)
                    # a replaced app's rescale leaves its model coming up
                    if self._results.get(rescale.model) == "rescaling":
                        self._results[rescale.model] = "rescaled"
                    del pending[name]
            if pending and (polls == 1 or polls % _PENDING_LOG_EVERY_N_POLLS == 0):
                _log_pending({name: rescale.model for name, rescale in pending.items()}, statuses)

    def _submit(self, name: str, item: _Add) -> None:
        request = self._request
        try:
            submit_app(item.config, request.gateway, serve_logging_config(), request.env)
        except Exception as e:
            self._failed_attempt(name, item, f"{type(e).__name__}: {e}")

    def _failed_attempt(self, name: str, item: _Add, message: str) -> None:
        """Schedules a resubmit after a backoff, or fails the request on a fatal error or the last attempt."""
        fatal = self._ledger.pop_fatal_error(name)
        if fatal is not None:
            self._fail(item.config.name, fatal)
        item.failures += 1
        item.last_error = message
        if item.failures >= _MAX_TRANSIENT_FAILURES:
            self._fail(item.config.name, f"{item.failures} failed attempts; last: {message}")
        logger.warning("Deploy failed for %s (deployment=%s); will retry: %s", item.config.name, name, message)
        item.retry_at = time.monotonic() + _RETRY_SLEEP_S * 2**item.failures

    def _fail(self, model: str, reason: str) -> NoReturn:
        self._results[model] = f"failed: {reason}"
        raise RequestFailedError(f"model {model!r}: {reason}")

    def _switch_and_commit(self, proposed: list[dict]) -> None:
        request = self._request
        routing = self._ledger.switch(request.id, request.gateway, proposed)
        self._switching = True
        if not self._replicas.wait_switched(request.gateway, routing, self._switch_timeout):
            raise RequestFailedError(
                f"gateway {request.gateway!r} did not switch to the new models within {self._switch_timeout:.0f} s"
            )
        version = self._ledger.commit(request.id, request.gateway, proposed)
        if version is None:
            raise RequestCancelledError()
        self._switching = False
        self._version = version
        logger.info("Gateway %s is on version %d", request.gateway, version)

    def _read(self) -> dict[str, ApplicationStatusOverview]:
        while (apps := self._try_read()) is None:
            time.sleep(POLL_SECONDS)
        return apps

    def _try_read(self) -> dict[str, ApplicationStatusOverview] | None:
        """Serve's apps; None while unreadable, failing the request after `_UNREADABLE_LIMIT_S`."""
        if self._ledger.cancelled(self._request.id):
            raise RequestCancelledError()
        try:
            apps = dict(serve.status().applications)
        except Exception:
            self._unreadable(exc_info=True)
            return None
        self._unreadable_since = None
        return apps

    def _unreadable(self, exc_info: bool = False) -> None:
        """Notes a failed read of Serve; fails the request once they have lasted `_UNREADABLE_LIMIT_S`."""
        now = time.monotonic()
        if self._unreadable_since is None:
            self._unreadable_since = now
            logger.warning("Could not read Serve's status; retrying", exc_info=exc_info)
        if now - self._unreadable_since >= _UNREADABLE_LIMIT_S:
            raise RequestFailedError(f"Serve's status could not be read for {_UNREADABLE_LIMIT_S:.0f} s") from None

    def _rolled_back(self, state: str, reason: str) -> dict:
        request = self._request
        self._ledger.rolling_back(request.id, request.gateway)
        roll_back(self._ledger, self._replicas, request.gateway, self._switching, self._switch_timeout)
        logger.info("Deploy %s rolled back", request.id)
        for model, result in self._results.items():
            if result in ("up", "coming up", "rescaled", "rescaling"):
                self._results[model] = "rolled back"
        return self._outcome(state, reason)

    def _outcome(self, state: str, reason: str) -> dict:
        return {
            "id": self._request.id,
            "state": state,
            "reason": reason,
            "models": dict(self._results),
            "diff": list(self._lines),
            "version": self._version,
        }


def _log_pending(pending: dict[str, str], statuses: dict[str, ApplicationStatusOverview]) -> None:
    """Logs the models of the *pending* apps, each with what Serve says it waits on."""
    waiting = []
    for name, model in pending.items():
        app = statuses.get(name)
        reason = next((d.message for d in app.deployments.values() if d.message), app.message) if app else ""
        waiting.append(f"{model} ({reason})" if reason else model)
    logger.info("Waiting on %d model(s): %s", len(pending), ", ".join(waiting))


@ray.remote(num_cpus=0)
class DeployWorker:
    """Runs one deploy request, or one rollback, for the deploy coordinator."""

    def __init__(self, coordinator):
        # nothing else configures logging in this process
        configure_logging()
        self._ledger = DeployLedger(coordinator)
        self._replicas = GatewayReplicas()

    def run(self, request: DeployRequest, switch_timeout: float) -> dict:
        return Run(request, self._ledger, self._replicas, switch_timeout).execute()

    def roll_back(self, gateway_name: str, switching: bool, switch_timeout: float) -> list[str]:
        return roll_back(self._ledger, self._replicas, gateway_name, switching, switch_timeout)


def create_worker(coordinator):
    """A deploy worker on the head node; source checks need the replicas' cache paths, and the replicas get its
    cluster-wide settings and state store."""
    return DeployWorker.options(
        runtime_env={"env_vars": build_cache_env_vars() | cluster_env_vars() | state_store_env_var()},
        **head_node_options(),
    ).remote(coordinator)
