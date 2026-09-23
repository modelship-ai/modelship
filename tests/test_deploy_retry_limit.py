"""`run_deploy_loop` must give up on a model that keeps failing to deploy, while
leaving one that is only short of capacity to come up on its own."""

import inspect
from unittest.mock import MagicMock

import pytest
from ray import serve
from ray.serve.schema import ApplicationStatus, ApplicationStatusOverview, LoggingConfig

from modelship.deploy import strategy
from modelship.infer.infer_config import ModelshipModelConfig


def _model(name: str) -> ModelshipModelConfig:
    return ModelshipModelConfig.model_validate(
        {"name": name, "model": f"org/{name}", "usecase": "generate", "loader": "vllm", "num_gpus": 1}
    )


def _app(status: ApplicationStatus, message: str = "") -> ApplicationStatusOverview:
    return ApplicationStatusOverview(status=status, message=message, last_deployed_time_s=0.0, deployments={})


DEPLOYING = _app(ApplicationStatus.DEPLOYING, "no room yet")
RUNNING = _app(ApplicationStatus.RUNNING)
FAILED = _app(ApplicationStatus.DEPLOY_FAILED, "engine died")


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
    deleted: list[str] = []
    monkeypatch.setattr(strategy, "delete_apps_quietly", lambda names: deleted.extend(names))
    monkeypatch.setattr(strategy.ray, "get", lambda ref, **kwargs: ref)

    def run(scripts: dict[str, list[ApplicationStatusOverview]], fatal: dict[str, str] | None = None, timeout="30"):
        monkeypatch.setenv("MSHIP_DEPLOY_TIMEOUT_S", timeout)
        fatal = fatal or {}
        names = {name: _model(name).deployment_name("g") for name in scripts}
        submitted: list[str] = []
        polls = {"n": -1}

        monkeypatch.setattr(strategy, "submit_deploy", lambda config, ctx: submitted.append(config.name))

        def status():
            polls["n"] += 1
            apps = {}
            for name, script in scripts.items():
                apps[names[name]] = script[min(polls["n"], len(script) - 1)]
            return MagicMock(applications=apps)

        monkeypatch.setattr(strategy.serve, "status", status)

        coordinator = MagicMock()
        coordinator.pop_fatal_error.remote.side_effect = lambda name: fatal.get(name)
        replica_coordinator = MagicMock()
        ctx = strategy.DeployContext(
            coordinator=coordinator,
            replica_coordinator=replica_coordinator,
            gateway_name="g",
            serve_logging_config=LoggingConfig(),
            deployed_this_run={},
        )
        pending, failed = strategy.run_deploy_loop([_model(name) for name in scripts], ctx)
        return {
            "pending": {c.name: reason for c, reason in pending},
            "failed": {c.name: detail for c, detail in failed},
            "submitted": submitted,
            "deleted": deleted,
            "registered": [call.args[1] for call in replica_coordinator.register_deployment.remote.call_args_list],
            "deployed_this_run": ctx.deployed_this_run,
        }

    return run


class TestTransientCap:
    def test_a_model_that_always_fails_is_given_up_on(self, loop):
        r = loop({"a": [FAILED]})
        assert r["submitted"].count("a") == strategy._MAX_TRANSIENT_FAILURES
        assert r["failed"] == {"a": "engine died"}

    def test_giving_up_deletes_the_failed_app(self, loop):
        r = loop({"a": [FAILED]})
        assert _model("a").deployment_name("g") in r["deleted"]

    def test_a_recovering_model_is_not_given_up_on(self, loop):
        r = loop({"a": [FAILED] + [DEPLOYING] * 3 + [RUNNING]})
        assert r["failed"] == {}
        assert r["pending"] == {}

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
        assert r["registered"] == [_model("b").deployment_name("g")]
        assert r["pending"] == {"a": "no room yet"}


class TestServeApiCanary:
    def test_run_many_still_takes_wait_for_applications_running(self):
        # @DeveloperAPI: the only public way to deploy without blocking on RUNNING.
        assert "wait_for_applications_running" in inspect.signature(serve.run_many).parameters


class TestRegistration:
    def test_a_running_model_is_registered_once(self, loop):
        r = loop({"a": [DEPLOYING, RUNNING]})
        assert r["registered"] == [_model("a").deployment_name("g")]

    def test_a_failed_model_is_dropped_from_this_runs_deployments(self, loop):
        r = loop({"a": [FAILED]}, fatal={_model("a").deployment_name("g"): "bad config"})
        assert r["deployed_this_run"] == {}
