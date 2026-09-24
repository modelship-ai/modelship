"""`run_deploy_loop` must give up on a model that keeps failing to deploy, while
leaving one that is only short of capacity to come up on its own."""

import inspect
import threading
from unittest.mock import MagicMock

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

from modelship.deploy import strategy
from modelship.infer.infer_config import ModelshipModelConfig


def _model(name: str) -> ModelshipModelConfig:
    return ModelshipModelConfig.model_validate(
        {"name": name, "model": f"org/{name}", "usecase": "generate", "loader": "vllm", "num_gpus": 1}
    )


def _app(status: ApplicationStatus, message: str = "", deployment_message: str = "") -> ApplicationStatusOverview:
    deployment = DeploymentStatusOverview(
        status=DeploymentStatus.UPDATING,
        status_trigger=DeploymentStatusTrigger.CONFIG_UPDATE_STARTED,
        replica_states={},
        message=deployment_message,
    )
    return ApplicationStatusOverview(
        status=status, message=message, last_deployed_time_s=0.0, deployments={"d": deployment}
    )


DEPLOYING = _app(ApplicationStatus.DEPLOYING, deployment_message="no room yet")
STARTING = _app(ApplicationStatus.DEPLOYING)
RUNNING = _app(ApplicationStatus.RUNNING)
FAILED = _app(ApplicationStatus.DEPLOY_FAILED, "engine died")
UNHEALTHY = _app(ApplicationStatus.UNHEALTHY, "replica failed its health check")


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def loop(monkeypatch):
    """Drives run_deploy_loop off a per-model script of Serve statuses, one entry
    consumed per poll; the last entry repeats. Returns what the loop did."""
    clock = _Clock()
    monkeypatch.setattr(strategy.time, "sleep", clock.sleep)
    monkeypatch.setattr(strategy.time, "monotonic", clock.monotonic)
    removed: list[str] = []
    monkeypatch.setattr(strategy, "remove_apps", lambda names, replica_coord, gateway: removed.extend(names))
    monkeypatch.setattr(strategy.ray, "get", lambda ref, **kwargs: ref)

    def run(scripts: dict[str, list[ApplicationStatusOverview]], fatal: dict[str, str] | None = None, timeout="30"):
        monkeypatch.setenv("MSHIP_DEPLOY_TIMEOUT_S", timeout)
        fatal = fatal or {}
        names = {name: _model(name).deployment_name("g") for name in scripts}
        submitted: list[str] = []
        polls = {"n": -1}

        def submit(config, ctx):
            submitted.append(config.name)
            ctx.deployed_this_run[config.deployment_name(ctx.gateway_name)] = config.name

        monkeypatch.setattr(strategy, "submit_deploy", submit)

        def status():
            polls["n"] += 1
            apps = {}
            for name, script in scripts.items():
                apps[names[name]] = script[min(polls["n"], len(script) - 1)]
            return MagicMock(applications=apps)

        monkeypatch.setattr(strategy.serve, "status", status)

        coordinator = MagicMock()
        coordinator.pop_fatal_error.remote.side_effect = lambda name: fatal.get(name)
        ctx = strategy.DeployContext(
            coordinator=coordinator,
            replica_coordinator=MagicMock(),
            gateway_name="g",
            serve_logging_config=LoggingConfig(),
            deployed_this_run={},
        )
        outcome = strategy.run_deploy_loop([_model(name) for name in scripts], ctx)
        return {
            "ready": [c.name for c in outcome.ready],
            "pending": {c.name: reason for c, reason in outcome.still_pending},
            "failed": {c.name: detail for c, detail in outcome.fatally_failed},
            "submitted": submitted,
            "removed": removed,
            "deployed_this_run": ctx.deployed_this_run,
        }

    return run


class TestTransientCap:
    def test_a_model_that_always_fails_is_given_up_on(self, loop):
        r = loop({"a": [FAILED]})
        assert r["submitted"].count("a") == strategy._MAX_TRANSIENT_FAILURES
        assert r["failed"] == {"a": "engine died"}

    def test_giving_up_removes_the_failed_app(self, loop):
        r = loop({"a": [FAILED]})
        assert r["removed"] == [_model("a").deployment_name("g")]

    def test_a_fatal_report_removes_the_app(self, loop):
        r = loop({"a": [FAILED]}, fatal={_model("a").deployment_name("g"): "bad config"})
        assert r["removed"] == [_model("a").deployment_name("g")]

    def test_a_retry_resubmits_over_the_failed_app(self, loop):
        r = loop({"a": [FAILED] + [DEPLOYING] * 3 + [RUNNING]})
        assert r["submitted"] == ["a", "a"]
        assert r["removed"] == []
        assert r["failed"] == {}
        assert r["pending"] == {}

    def test_removal_runs_off_the_polling_thread(self, loop, monkeypatch):
        threads = []
        monkeypatch.setattr(strategy, "remove_apps", lambda *args: threads.append(threading.current_thread()))
        loop({"a": [FAILED]})
        assert len(threads) == 1
        assert threads[0] is not threading.current_thread()

    def test_a_fatal_report_wins_immediately(self, loop):
        r = loop({"a": [FAILED]}, fatal={_model("a").deployment_name("g"): "bad config"})
        assert r["submitted"] == ["a"]
        assert r["failed"] == {"a": "bad config"}

    def test_one_failing_model_does_not_strand_a_healthy_one(self, loop):
        r = loop({"a": [FAILED], "b": [RUNNING]})
        assert r["failed"] == {"a": "engine died"}
        assert r["pending"] == {}
        assert r["submitted"].count("b") == 1


class TestPendingIsNotFailure:
    def test_a_model_short_of_capacity_is_reported_pending(self, loop):
        r = loop({"a": [DEPLOYING]}, timeout="10")
        assert r["failed"] == {}
        assert r["pending"] == {"a": "no room yet"}

    def test_waiting_never_consumes_an_attempt(self, loop):
        r = loop({"a": [DEPLOYING] * 20 + [RUNNING]})
        assert r["submitted"] == ["a"]
        assert r["failed"] == {}

    def test_a_pending_model_does_not_hold_up_another(self, loop):
        r = loop({"a": [DEPLOYING], "b": [RUNNING]}, timeout="10")
        assert r["ready"] == ["b"]
        assert r["pending"] == {"a": "no room yet"}
        assert r["failed"] == {}

    def test_an_unhealthy_app_stays_pending_without_a_resubmit(self, loop):
        r = loop({"a": [UNHEALTHY]}, timeout="10")
        assert r["submitted"] == ["a"]
        assert r["failed"] == {}
        assert r["pending"] == {"a": "replica failed its health check"}

    def test_the_first_poll_logs_what_is_outstanding(self, loop, caplog):
        with caplog.at_level("INFO"):
            loop({"a": [DEPLOYING, RUNNING]})
        assert "Waiting on 1 model(s): a (no room yet)" in caplog.text

    def test_a_model_serve_gives_no_reason_for_is_pending_with_an_empty_reason(self, loop):
        r = loop({"a": [STARTING]}, timeout="10")
        assert r["pending"] == {"a": ""}

    def test_a_model_without_a_reason_is_logged_by_name_alone(self, loop, caplog):
        with caplog.at_level("INFO"):
            loop({"a": [STARTING, RUNNING]})
        assert "Waiting on 1 model(s): a" in caplog.messages

    def test_a_model_waiting_to_retry_at_the_deadline_is_failed(self, loop):
        r = loop({"a": [FAILED]}, timeout="5")
        assert r["pending"] == {}
        assert r["failed"] == {"a": "engine died"}
        assert r["removed"] == [_model("a").deployment_name("g")]


class TestServeApiCanary:
    def test_run_many_still_takes_wait_for_applications_running(self):
        # @DeveloperAPI: the only public way to deploy without blocking on RUNNING.
        assert "wait_for_applications_running" in inspect.signature(serve.run_many).parameters


class TestSubmit:
    def test_declares_then_hands_off_without_waiting(self, monkeypatch):
        calls = []
        replica_coordinator = MagicMock()
        replica_coordinator.declare_deployment.remote.side_effect = lambda *args: calls.append(("declare", *args))
        monkeypatch.setattr(strategy.ray, "get", lambda ref, **kwargs: ref)
        monkeypatch.setattr(strategy.serve, "run_many", lambda targets, **kwargs: calls.append(("run_many", kwargs)))
        ctx = strategy.DeployContext(
            coordinator=MagicMock(),
            replica_coordinator=replica_coordinator,
            gateway_name="g",
            serve_logging_config=LoggingConfig(),
            deployed_this_run={},
        )
        strategy.submit_deploy(_model("a"), ctx)
        name = _model("a").deployment_name("g")
        assert calls == [("declare", "g", name, "a"), ("run_many", {"wait_for_applications_running": False})]
        assert ctx.deployed_this_run == {name: "a"}


class TestThisRunsDeployments:
    def test_a_running_model_stays_recorded(self, loop):
        r = loop({"a": [DEPLOYING, RUNNING]})
        assert r["deployed_this_run"] == {_model("a").deployment_name("g"): "a"}

    def test_a_failed_model_is_dropped(self, loop):
        r = loop({"a": [FAILED]}, fatal={_model("a").deployment_name("g"): "bad config"})
        assert r["deployed_this_run"] == {}
