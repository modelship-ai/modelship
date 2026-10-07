"""The deploy worker's request run and rollback, against a fake Serve, deploy coordinator and gateway replicas,
and the submits it uses."""

import inspect
import logging
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from ray import serve
from ray.serve.schema import (
    ApplicationStatus,
    ApplicationStatusOverview,
    DeploymentStatus,
    DeploymentStatusOverview,
    DeploymentStatusTrigger,
    LoggingConfig,
)

from modelship.deploy import strategy, worker
from modelship.deploy.diff import build_diff
from modelship.deploy.ledger import DeployRequest, to_config
from modelship.deploy.strategy import LiveApp, ServeApp, rescale_app, serve_apps, serve_scaling, submit_app
from modelship.deploy.worker import DeployLedger, Run, roll_back
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.infer.sources import LocalSource


def _raw(name: str, **overrides) -> dict:
    return {"name": name, "model": f"org/{name}", "usecase": "generate", "loader": "llama_server", **overrides}


def _app_name(raw: dict, gateway: str = "gw") -> str:
    return ModelshipModelConfig.model_validate(raw).deployment_name(gateway)


def _app(status: ApplicationStatus, message: str = "") -> ApplicationStatusOverview:
    deployment = DeploymentStatusOverview(
        status=DeploymentStatus.HEALTHY,
        status_trigger=DeploymentStatusTrigger.CONFIG_UPDATE_COMPLETED,
        replica_states={},
        message="",
    )
    return ApplicationStatusOverview(
        status=status, message=message, last_deployed_time_s=0.0, deployments={"d": deployment}
    )


RUNNING = _app(ApplicationStatus.RUNNING)
DEPLOYING = _app(ApplicationStatus.DEPLOYING)
FAILED = _app(ApplicationStatus.DEPLOY_FAILED, "engine died")
UNHEALTHY = _app(ApplicationStatus.UNHEALTHY)
DELETING = _app(ApplicationStatus.DELETING)
A, A2, B = _raw("a", num_cpus=1), _raw("a", num_cpus=2), _raw("b")
C = _raw("c", llama_server_config={"n_ctx": 4096})
C2 = {**C, "num_replicas": 2}


class _Serve:
    """Serve's apps as a worker reads them. A submitted app follows its model's script of statuses, one entry
    per read across submits, the last repeating; a deleted or DELETING app is gone at the next read."""

    def __init__(self):
        self.apps: dict[str, ApplicationStatusOverview] = {"gw": RUNNING}
        self.scripts: dict[str, list[ApplicationStatusOverview]] = {}
        self.running: dict[str, list[ApplicationStatusOverview]] = {}
        self.deleted: set[str] = set()
        self.unreadable = 0
        self.events: list[tuple[str, str]] = []
        self.logging_configs: list = []
        self.configs: dict[str, ModelshipModelConfig] = {}

    def status(self):
        if self.unreadable:
            self.unreadable -= 1
            raise RuntimeError("serve controller unreachable")
        self.deleted |= {name for name, app in self.apps.items() if app.status == ApplicationStatus.DELETING}
        for name in self.deleted:
            self.apps.pop(name, None)
            self.running.pop(name, None)
        self.deleted.clear()
        for name, script in self.running.items():
            if name in self.apps:
                self.apps[name] = script.pop(0) if len(script) > 1 else script[0]
        return SimpleNamespace(applications=dict(self.apps))

    def delete(self, name: str, _blocking: bool = True) -> None:
        assert not _blocking
        self.events.append(("delete", name))
        self.deleted.add(name)

    def submit(self, config, gateway_name, serve_logging_config, env) -> None:
        name = config.deployment_name(gateway_name)
        self.events.append(("submit", name))
        self.logging_configs.append(serve_logging_config)
        self.configs[name] = config
        self.running[name] = self.scripts.setdefault(config.name, [RUNNING])
        self.apps[name] = DEPLOYING

    def read(self) -> dict[str, ServeApp]:
        return {
            name: ServeApp(app.status, serve_scaling(self.configs[name]) if name in self.configs else None)
            for name, app in self.status().applications.items()
        }

    def live(self, app_name: str) -> LiveApp | None:
        if app_name not in self.apps or app_name not in self.configs:
            return None
        return LiveApp(self.configs[app_name], {}, "v1")

    def rescale(self, app_name: str, live: LiveApp, scaling: dict, serve_logging_config) -> None:
        self.events.append(("rescale", app_name))
        self.configs[app_name] = live.config.model_copy(update=scaling)
        self.running[app_name] = self.scripts.setdefault(live.config.name, [RUNNING])
        self.apps[app_name] = DEPLOYING


class _Ledger:
    def __init__(self, serve: _Serve):
        self.serve = serve
        self.cancel = False
        self.cancel_on_commit = False
        # the commit's store write lands, then the call raises
        self.commit_error: Exception | None = None
        self.committed: list[dict] | None = None
        self.crashing: dict[str, str] = {}
        self.fatal: dict[str, str] = {}
        self.routing = 0
        self.commits: list[list[dict]] = []
        self.request: DeployRequest | None = None

    def cancelled(self, request_id):
        return self.cancel

    def diff(self, request_id, gateway_name):
        return self._diff(self.request)

    def rollback_diff(self, gateway_name):
        return self._diff(DeployRequest(gateway_name, "bare", "blue_green", None, {}))

    def _diff(self, request):
        try:
            return build_diff(request, self.committed, self.serve.read())
        except RuntimeError:
            return None

    def rolling_back(self, request_id, gateway_name):
        self.serve.events.append(("rolling back", request_id))

    def crash_looping(self, apps):
        return {app: reason for app, reason in self.crashing.items() if app in apps}

    def pop_fatal_error(self, app):
        return self.fatal.pop(app, None)

    def forget_deaths(self, apps):
        pass

    def switch(self, request_id, gateway_name, models):
        self.routing += 1
        self.serve.events.append(("switch", str(self.routing)))
        return self.routing

    def reset_routing(self, gateway_name):
        self.serve.events.append(("reset", str(self.routing)))
        return self.routing

    def commit(self, request_id, gateway_name, models):
        if self.cancel_on_commit:
            return None
        self.committed = models
        if self.commit_error is not None:
            raise self.commit_error
        self.commits.append(models)
        self.serve.events.append(("commit", str(len(self.commits))))
        return len(self.commits)


class _Replicas:
    def __init__(self):
        self.switches = True
        self.waits: list[int] = []

    def wait_switched(self, gateway_name, routing, timeout):
        self.waits.append(routing)
        return self.switches


class _Clock:
    def __init__(self):
        self.now = 0.0
        # called with the new time after each sleep
        self.hook = lambda now: None

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.hook(self.now)

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def cluster(monkeypatch):
    serve, clock = _Serve(), _Clock()
    monkeypatch.setattr(worker.serve, "status", serve.status)
    monkeypatch.setattr(worker.serve, "delete", serve.delete)
    monkeypatch.setattr(worker, "submit_app", serve.submit)
    monkeypatch.setattr(worker, "live_app", serve.live)
    monkeypatch.setattr(worker, "rescale_app", serve.rescale)
    monkeypatch.setattr(worker.time, "sleep", clock.sleep)
    monkeypatch.setattr(worker.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(worker.ray, "cluster_resources", lambda: {})
    serve.pinned = []
    monkeypatch.setattr(worker, "resolve_all_model_sources", lambda conf: serve.pinned.extend(conf.models))
    return SimpleNamespace(serve=serve, ledger=_Ledger(serve), replicas=_Replicas(), clock=clock)


def _run(cluster, models, committed=None, mode="additive", strategy="blue_green") -> dict:
    request = DeployRequest("gw", mode, strategy, models, {}, id="r1")
    cluster.ledger.committed = committed
    cluster.ledger.request = request
    return Run(request, cluster.ledger, cluster.replicas, 60.0).execute()


def _existing(cluster, *raws: dict) -> None:
    for raw in raws:
        cluster.serve.apps[_app_name(raw)] = RUNNING
        cluster.serve.configs[_app_name(raw)] = ModelshipModelConfig.model_validate(raw)


def _configs(*raws: dict) -> list[ModelshipModelConfig]:
    return list(to_config(list(raws)).models)


class TestSucceeds:
    def test_a_new_model_comes_up_then_the_gateway_switches_then_it_commits(self, cluster):
        outcome = _run(cluster, [A])
        assert outcome == {
            "id": "r1",
            "state": "succeeded",
            "reason": "",
            "models": {"a": "up"},
            "diff": ["4. add a", "6. switch the gateway and commit"],
            "version": 1,
        }
        assert cluster.serve.events == [("submit", _app_name(A)), ("switch", "1"), ("commit", "1")]
        assert cluster.ledger.commits == [[A]]
        assert cluster.serve.pinned[0].name == "a"

    def test_apps_are_submitted_with_the_workers_serve_logging(self, cluster, monkeypatch):
        monkeypatch.setattr(worker, "serve_logging_config", lambda: "worker-logging")
        _run(cluster, [A])
        assert cluster.serve.logging_configs == ["worker-logging"]

    def test_blue_green_deletes_the_replaced_app_only_after_the_commit(self, cluster):
        _existing(cluster, A)
        _run(cluster, [A2], committed=[A])
        assert cluster.serve.events == [
            ("submit", _app_name(A2)),
            ("switch", "1"),
            ("commit", "1"),
            ("delete", _app_name(A)),
        ]

    def test_stop_start_deletes_the_replaced_app_before_submitting(self, cluster):
        _existing(cluster, A)
        _run(cluster, [A2], committed=[A], strategy="stop_start")
        assert cluster.serve.events[:2] == [("delete", _app_name(A)), ("submit", _app_name(A2))]

    def test_reconcile_removes_a_dropped_model_after_the_commit(self, cluster):
        _existing(cluster, A, B)
        outcome = _run(cluster, [A], committed=[A, B], mode="reconcile")
        assert cluster.serve.events == [("switch", "1"), ("commit", "1"), ("delete", _app_name(B))]
        assert outcome["models"] == {"a": "unchanged", "b": "removed"}

    def test_a_replica_count_change_rescales_the_live_app_then_commits(self, cluster):
        _existing(cluster, C)
        outcome = _run(cluster, [C2], committed=[C])
        assert outcome["models"] == {"c": "rescaled"}
        assert outcome["diff"] == ["5. rescale c: num_replicas: 1 -> 2", "6. switch the gateway and commit"]
        assert cluster.serve.events == [("rescale", _app_name(C)), ("switch", "1"), ("commit", "1")]
        assert cluster.ledger.commits == [[C2]]
        assert cluster.serve.configs[_app_name(C)].num_replicas == 2
        assert cluster.serve.pinned == []

    def test_a_new_model_comes_up_before_a_scale_up(self, cluster):
        _existing(cluster, C)
        _run(cluster, [C2, A], committed=[C])
        assert cluster.serve.events[:2] == [("submit", _app_name(A)), ("rescale", _app_name(C))]

    def test_a_scale_down_runs_before_a_new_model_comes_up(self, cluster):
        _existing(cluster, C2)
        _run(cluster, [A, C], committed=[C2])
        assert cluster.serve.events[:2] == [("rescale", _app_name(C)), ("submit", _app_name(A))]
        assert cluster.serve.configs[_app_name(C)].num_replicas == 1

    def test_stop_start_scales_down_then_deletes_then_submits(self, cluster):
        _existing(cluster, A, C2)
        _run(cluster, [A2, C], committed=[A, C2], strategy="stop_start")
        assert cluster.serve.events[:3] == [
            ("rescale", _app_name(C)),
            ("delete", _app_name(A)),
            ("submit", _app_name(A2)),
        ]

    def test_a_replaced_app_is_scaled_down_before_its_replacement_comes_up(self, cluster):
        changed = {**C, "num_cpus": 2}
        _existing(cluster, C2)
        outcome = _run(cluster, [changed], committed=[C2])
        assert outcome["models"] == {"c": "up"}
        assert cluster.serve.events == [
            ("rescale", _app_name(C2)),
            ("submit", _app_name(changed)),
            ("switch", "1"),
            ("commit", "1"),
            ("delete", _app_name(C2)),
        ]

    def test_a_replica_count_serve_holds_against_the_committed_version_is_put_back(self, cluster):
        _existing(cluster, C2)
        outcome = _run(cluster, [C], committed=[C])
        assert outcome["models"] == {"c": "rescaled"}
        assert outcome["version"] is None
        assert cluster.serve.events == [("rescale", _app_name(C))]
        assert cluster.serve.configs[_app_name(C)].num_replicas == 1

    def test_a_bare_request_puts_a_committed_replica_count_back(self, cluster):
        _existing(cluster, C2)
        _run(cluster, None, committed=[C], mode="bare")
        assert cluster.serve.events == [("rescale", _app_name(C))]

    def test_a_rescale_is_pending_until_the_app_is_running_again(self, cluster):
        _existing(cluster, C)
        cluster.serve.scripts["c"] = [DEPLOYING] * 50 + [RUNNING]
        assert _run(cluster, [C2], committed=[C])["state"] == "succeeded"
        assert cluster.serve.events.count(("rescale", _app_name(C))) == 1

    def test_a_fatal_error_reported_before_the_request_does_not_fail_a_rescale(self, cluster):
        _existing(cluster, C)
        cluster.ledger.fatal[_app_name(C)] = "an older failure"
        assert _run(cluster, [C2], committed=[C])["state"] == "succeeded"

    def test_a_bare_request_resubmits_a_missing_committed_app_without_switching(self, cluster):
        _existing(cluster, A)
        outcome = _run(cluster, None, committed=[A, B], mode="bare")
        assert cluster.serve.events == [("submit", _app_name(B))]
        assert outcome["models"] == {"a": "unchanged", "b": "up"}
        assert outcome["version"] is None

    def test_a_request_that_changes_nothing_submits_nothing(self, cluster):
        _existing(cluster, A)
        assert _run(cluster, [A], committed=[A])["state"] == "succeeded"
        assert cluster.serve.events == []

    def test_leftovers_are_deleted_before_anything_else(self, cluster):
        _existing(cluster, A, B)
        _run(cluster, [A2], committed=[A])
        assert cluster.serve.events[0] == ("delete", _app_name(B))

    def test_a_transient_deploy_failure_is_retried(self, cluster):
        cluster.serve.scripts["a"] = [FAILED, RUNNING]
        assert _run(cluster, [A])["state"] == "succeeded"
        assert cluster.serve.events.count(("submit", _app_name(A))) == 2

    def test_unhealthy_is_still_pending(self, cluster):
        cluster.serve.scripts["a"] = [UNHEALTHY, UNHEALTHY, RUNNING]
        assert _run(cluster, [A])["state"] == "succeeded"

    def test_waiting_for_capacity_never_uses_an_attempt(self, cluster):
        cluster.serve.scripts["a"] = [DEPLOYING] * 50 + [RUNNING]
        assert _run(cluster, [A])["state"] == "succeeded"
        assert cluster.serve.events.count(("submit", _app_name(A))) == 1

    def test_a_submit_that_raises_is_retried(self, cluster, monkeypatch):
        attempts = []

        def flaky(config, gateway_name, serve_logging_config, env):
            attempts.append(config.name)
            if len(attempts) == 1:
                raise RuntimeError("controller busy")
            cluster.serve.submit(config, gateway_name, serve_logging_config, env)

        monkeypatch.setattr(worker, "submit_app", flaky)
        assert _run(cluster, [A])["state"] == "succeeded"
        assert attempts == ["a", "a"]

    def test_the_first_poll_logs_what_is_outstanding(self, cluster, caplog):
        caplog.set_level(logging.INFO, logger="modelship")
        cluster.serve.scripts["a"] = [DEPLOYING, RUNNING]
        _run(cluster, [A])
        assert "Waiting on 1 model(s): a" in caplog.messages


class TestRollsBack:
    def test_a_fatal_failure_deletes_the_new_app_and_commits_nothing(self, cluster):
        _existing(cluster, A)
        cluster.serve.scripts["a"] = [FAILED]
        cluster.ledger.fatal[_app_name(A2)] = "bad weights"
        outcome = _run(cluster, [A2], committed=[A])
        assert outcome["state"] == "failed"
        assert outcome["reason"] == "model 'a': bad weights"
        assert outcome["models"] == {"a": "failed: bad weights"}
        assert cluster.ledger.commits == []
        assert ("delete", _app_name(A2)) in cluster.serve.events
        assert ("delete", _app_name(A)) not in cluster.serve.events

    def test_a_model_still_failing_after_its_retries_fails_the_request(self, cluster):
        cluster.serve.scripts["a"] = [FAILED]
        outcome = _run(cluster, [A])
        assert outcome["state"] == "failed"
        assert cluster.serve.events.count(("submit", _app_name(A))) == 3

    def test_a_models_other_models_are_rolled_back_too(self, cluster):
        cluster.serve.scripts["b"] = [DEPLOYING, FAILED]
        cluster.ledger.fatal[_app_name(B)] = "boom"
        outcome = _run(cluster, [A, B])
        assert outcome["models"] == {"a": "rolled back", "b": "failed: boom"}
        assert {("delete", _app_name(A)), ("delete", _app_name(B))} <= set(cluster.serve.events)

    def test_a_gateway_that_does_not_switch_is_routed_back_before_the_new_app_goes(self, cluster):
        cluster.replicas.switches = False
        outcome = _run(cluster, [A])
        assert outcome["reason"] == "gateway 'gw' did not switch to the new models within 60 s"
        assert cluster.replicas.waits == [1, 1]
        assert cluster.serve.events[-2:] == [("reset", "1"), ("delete", _app_name(A))]
        assert cluster.ledger.commits == []

    def test_a_cancel_while_coming_up(self, cluster):
        cluster.serve.scripts["a"] = [DEPLOYING]
        cluster.clock.hook = lambda now: setattr(cluster.ledger, "cancel", now > 10)
        outcome = _run(cluster, [A])
        assert outcome["state"] == "cancelled"
        assert outcome["models"] == {"a": "rolled back"}
        assert ("delete", _app_name(A)) in cluster.serve.events

    def test_a_request_marks_itself_rolling_back_before_routing_back(self, cluster):
        cluster.serve.scripts["a"] = [FAILED]
        _run(cluster, [A])
        rolling_back = cluster.serve.events.index(("rolling back", "r1"))
        assert cluster.serve.events[rolling_back + 1] == ("reset", "0")

    def test_a_cancel_that_lands_while_switching_is_refused_at_commit(self, cluster):
        cluster.ledger.cancel_on_commit = True
        assert _run(cluster, [A])["state"] == "cancelled"
        assert ("delete", _app_name(A)) in cluster.serve.events

    def test_a_commit_that_lands_despite_an_error_keeps_its_apps(self, cluster):
        _existing(cluster, A)
        cluster.ledger.commit_error = RuntimeError("store connection reset")
        outcome = _run(cluster, [A2], committed=[A])
        assert outcome["state"] == "failed"
        assert _app_name(A2) in cluster.serve.apps
        assert ("delete", _app_name(A)) in cluster.serve.events

    def test_serve_unreadable_for_30s_fails_the_request(self, cluster):
        cluster.serve.scripts["a"] = [DEPLOYING]
        cluster.serve.unreadable = 1000
        cluster.clock.hook = lambda now: now > 100 and setattr(cluster.serve, "unreadable", 0)
        outcome = _run(cluster, [A])
        assert outcome["reason"] == "Serve's status could not be read for 30 s"

    def test_an_app_deleted_after_serve_reported_it(self, cluster):
        cluster.serve.scripts["a"] = [DEPLOYING, DEPLOYING, DELETING]
        assert _run(cluster, [A])["reason"] == f"model 'a': deployment {_app_name(A)} was deleted"

    def test_a_backend_that_keeps_dying(self, cluster):
        cluster.serve.scripts["a"] = [DEPLOYING]
        cluster.ledger.crashing[_app_name(A)] = "llama-server exited"
        assert _run(cluster, [A])["reason"] == "model 'a': its backend keeps dying: llama-server exited"

    def test_a_fatal_error_while_rescaling_gives_the_app_its_committed_replica_count(self, cluster):
        _existing(cluster, C)
        cluster.serve.scripts["c"] = [DEPLOYING] * 50 + [RUNNING]
        cluster.clock.hook = lambda now: cluster.ledger.fatal.setdefault(_app_name(C), "no memory left")
        outcome = _run(cluster, [C2], committed=[C])
        assert outcome["reason"] == "model 'c': no memory left"
        assert outcome["models"] == {"c": "failed: no memory left"}
        assert cluster.serve.events.count(("rescale", _app_name(C))) == 2
        assert cluster.serve.configs[_app_name(C)].num_replicas == 1
        assert cluster.ledger.commits == []

    def test_a_cancel_while_rescaling_gives_the_app_its_committed_replica_count(self, cluster):
        _existing(cluster, C)
        cluster.serve.scripts["c"] = [DEPLOYING]
        cluster.clock.hook = lambda now: setattr(cluster.ledger, "cancel", now > 10)
        outcome = _run(cluster, [C2], committed=[C])
        assert outcome["state"] == "cancelled"
        assert outcome["models"] == {"c": "rolled back"}
        assert cluster.serve.configs[_app_name(C)].num_replicas == 1
        assert _app_name(C) in cluster.serve.apps

    def test_a_rescale_of_an_app_serve_no_longer_has_fails_the_request(self, cluster):
        cluster.serve.apps[_app_name(C)] = RUNNING
        outcome = _run(cluster, [C2], committed=[C])
        assert outcome["reason"] == f"model 'c': deployment {_app_name(C)} was deleted"

    def test_a_rescale_that_raises_fails_the_request(self, cluster, monkeypatch):
        def broken(app_name, live, scaling, serve_logging_config):
            raise RuntimeError("controller busy")

        _existing(cluster, C)
        monkeypatch.setattr(worker, "rescale_app", broken)
        assert _run(cluster, [C2], committed=[C])["reason"] == "model 'c': RuntimeError: controller busy"

    def test_a_source_that_cannot_be_checked_fails_before_any_submit(self, cluster, monkeypatch):
        def missing(conf):
            raise FileNotFoundError("repo org/a not found")

        monkeypatch.setattr(worker, "resolve_all_model_sources", missing)
        outcome = _run(cluster, [A])
        assert outcome["reason"] == "repo org/a not found"
        assert not any(event == "submit" for event, _ in cluster.serve.events)


class TestRollBack:
    def test_deletes_only_what_the_committed_version_does_not_name(self, cluster):
        _existing(cluster, A, A2, B)
        cluster.ledger.committed = [A, B]
        assert roll_back(cluster.ledger, cluster.replicas, "gw", False, 60.0) == [_app_name(A2)]
        assert cluster.replicas.waits == []

    def test_waits_for_the_switch_back_only_when_switching(self, cluster):
        roll_back(cluster.ledger, cluster.replicas, "gw", True, 60.0)
        assert cluster.replicas.waits == [0]

    def test_gives_a_live_app_its_committed_replica_count(self, cluster):
        _existing(cluster, C2, B)
        cluster.ledger.committed = [C, B]
        roll_back(cluster.ledger, cluster.replicas, "gw", False, 60.0)
        assert [event for event in cluster.serve.events if event[0] == "rescale"] == [("rescale", _app_name(C))]
        assert cluster.serve.configs[_app_name(C)].num_replicas == 1

    def test_does_not_bring_back_a_committed_app_serve_no_longer_has(self, cluster):
        cluster.ledger.committed = [C]
        roll_back(cluster.ledger, cluster.replicas, "gw", False, 60.0)
        assert cluster.serve.events == [("reset", "0")]

    def test_gives_up_on_an_app_whose_deployment_cannot_be_read(self, cluster, monkeypatch, caplog):
        def unreadable(app_name):
            raise RuntimeError("controller busy")

        _existing(cluster, C2)
        cluster.ledger.committed = [C]
        monkeypatch.setattr(worker, "live_app", unreadable)
        roll_back(cluster.ledger, cluster.replicas, "gw", False, 60.0)
        assert cluster.clock.now >= 30
        assert f"Could not give {_app_name(C)} its committed replica-count fields" in caplog.messages

    def test_without_a_committed_version_every_model_app_goes(self, cluster):
        _existing(cluster, A, B)
        assert roll_back(cluster.ledger, cluster.replicas, "gw", False, 60.0) == sorted([_app_name(A), _app_name(B)])
        assert "gw" in cluster.serve.apps


def test_deploy_coordinator_calls_have_no_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(worker.ray, "get", lambda ref, **kwargs: calls.append(kwargs))
    handle = SimpleNamespace(commit=SimpleNamespace(remote=lambda *args: None))
    DeployLedger(handle).commit("r1", "gw", [A])
    assert calls == [{}]


class TestServeApps:
    def test_each_app_with_its_status_and_the_replica_fields_its_deployment_is_set_to(self, monkeypatch):
        tunables = {"target_ongoing_requests": 2.0, "upscale_delay_s": 30.0, "downscale_delay_s": 600.0}
        autoscaling = {"min_replicas": 1, "max_replicas": 3, "initial_replicas": None, **tunables}
        details = {
            "applications": {
                "gw.a-1": {
                    "status": ApplicationStatus.RUNNING,
                    "deployments": {"gw.a-1": {"deployment_config": {"num_replicas": 2, "autoscaling_config": None}}},
                },
                "gw.b-1": {
                    "status": "DEPLOYING",
                    "deployments": {
                        "gw.b-1": {"deployment_config": {"autoscaling_config": {**autoscaling, "policy": {}}}}
                    },
                },
                "gw": {"status": ApplicationStatus.RUNNING, "deployments": {"ModelshipAPI": {}}},
            }
        }
        from ray.serve.context import _get_global_client

        client = SimpleNamespace(get_serve_details=lambda: details)
        monkeypatch.setattr(
            "ray.serve.context._get_global_client", create_autospec(_get_global_client, return_value=client)
        )
        assert serve_apps() == {
            "gw.a-1": ServeApp(ApplicationStatus.RUNNING, {"num_replicas": 2}),
            "gw.b-1": ServeApp(ApplicationStatus.DEPLOYING, autoscaling),
            "gw": ServeApp(ApplicationStatus.RUNNING, None),
        }

    def test_a_configs_replica_fields_as_serve_holds_them(self):
        autoscaled = {**C, "autoscaling_config": {"min_replicas": 1, "max_replicas": 2}}
        assert serve_scaling(ModelshipModelConfig.model_validate(C2)) == {"num_replicas": 2}
        assert serve_scaling(ModelshipModelConfig.model_validate(autoscaled)) == {
            "min_replicas": 1,
            "max_replicas": 2,
            "initial_replicas": None,
            "target_ongoing_requests": 2,
            "upscale_delay_s": 30.0,
            "downscale_delay_s": 600.0,
        }

    def test_serve_still_reports_what_is_read(self):
        from ray.serve.config import AutoscalingConfig
        from ray.serve.schema import ApplicationDetails, DeploymentDetails, DeploymentSchema

        assert {"status", "deployments"} <= set(ApplicationDetails.model_fields)
        assert "deployment_config" in DeploymentDetails.model_fields
        assert {"num_replicas", "autoscaling_config"} <= set(DeploymentSchema.model_fields)
        assert set(strategy._AUTOSCALING_FIELDS) <= set(AutoscalingConfig.model_fields)


class TestSubmitApp:
    def test_hands_off_without_waiting(self, monkeypatch):
        calls = []
        monkeypatch.setattr(strategy.serve, "run_many", lambda targets, **kwargs: calls.append(kwargs))
        submit_app(_configs(A)[0], "gw", LoggingConfig(), {"MSHIP_PREFLIGHT": "false"})
        assert calls == [{"wait_for_applications_running": False}]

    def test_run_many_still_takes_wait_for_applications_running(self):
        # @DeveloperAPI: the only public way to deploy without blocking on RUNNING.
        assert "wait_for_applications_running" in inspect.signature(serve.run_many).parameters

    def test_leaves_the_version_to_serve(self, monkeypatch):
        targets = []
        monkeypatch.setattr(strategy.serve, "run_many", lambda submitted, **kwargs: targets.extend(submitted))
        submit_app(_configs(A)[0], "gw", LoggingConfig(), {})
        assert targets[0].target._bound_deployment._version is None


class TestRescaleApp:
    def test_resubmits_the_live_config_under_its_version_and_runtime_env(self, monkeypatch):
        targets = []
        monkeypatch.setattr(strategy.serve, "run_many", lambda submitted, **kwargs: targets.extend(submitted))
        config = _configs(C)[0]
        config._pinned_source = LocalSource("/models/c.gguf")
        live = LiveApp(config, {"env_vars": {"MSHIP_PREFLIGHT": "false"}}, "v7")

        rescale_app(_app_name(C), live, _configs(C2)[0].scaling(), LoggingConfig())

        deployment = targets[0].target._bound_deployment
        assert targets[0].name == deployment.name == _app_name(C)
        assert deployment._version == "v7"
        assert deployment.num_replicas == 2
        assert deployment.ray_actor_options["runtime_env"] == {"env_vars": {"MSHIP_PREFLIGHT": "false"}}
        assert deployment.init_args[0].num_replicas == 2
        assert deployment.init_args[0]._pinned_source == LocalSource("/models/c.gguf")

    def test_serve_still_has_what_a_rescale_reads_and_sets(self):
        from ray.serve._private.client import ServeControllerClient
        from ray.serve._private.config import ReplicaConfig
        from ray.serve._private.deployment_info import DeploymentInfo

        assert "deployment._version or get_random_string()" in inspect.getsource(
            ServeControllerClient.deploy_applications
        )
        assert list(inspect.signature(ServeControllerClient.get_deployment_info).parameters)[1:] == ["name", "app_name"]
        assert "version" in inspect.signature(DeploymentInfo.__init__).parameters
        assert isinstance(inspect.getattr_static(ReplicaConfig, "init_args"), property)
        assert "ray_actor_options" in inspect.signature(ReplicaConfig.__init__).parameters
