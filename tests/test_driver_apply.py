"""_apply: what the driver writes, deploys and deletes around the deploy loop."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from ray.serve.schema import ApplicationStatus

from modelship import driver
from modelship.deploy.effective_config import read_effective, write_effective
from modelship.deploy.strategy import DeployOutcome
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.state import MemoryStoreActor

_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class


def _raw(name: str, **overrides) -> dict:
    return {"name": name, "model": f"org/{name}", "usecase": "generate", "loader": "llama_server", **overrides}


def _config(raw: dict) -> ModelshipModelConfig:
    return ModelshipModelConfig.model_validate(raw)


def _app(raw: dict) -> str:
    return _config(raw).deployment_name("g")


@pytest.fixture
def apply():
    def run(prev, desired, statuses, outcome=None, replace_strategy="blue_green", sources_error=None):
        store = _MemoryStore()
        write_effective(store, "g", prev)
        events: list = []
        effective_at_submit: list = []

        def deploy_loop(models, ctx):
            effective_at_submit.append(read_effective(store, "g"))
            events.append("deploy")
            return outcome or DeployOutcome([], [], [])

        run_deploy_loop = MagicMock(side_effect=deploy_loop)
        delete = MagicMock(side_effect=lambda names: events.append(("delete", list(names))))
        leases = MagicMock()
        changed = MagicMock()
        args = SimpleNamespace(reconcile=True, config="models.yaml", model=None, replace_strategy=replace_strategy)
        with ExitStack() as stack:
            for target, value in {
                "modelship.deploy.serve_utils.get_app_statuses": MagicMock(return_value=statuses),
                "modelship.deploy.config.resolve_input_models": MagicMock(return_value=desired),
                "modelship.deploy.config.resolve_all_model_sources": MagicMock(side_effect=sources_error),
                "modelship.state.get_state_store": MagicMock(return_value=store),
                "modelship.openai.compaction_crypto.ensure_key_seeded": MagicMock(),
                "modelship.infer.deploy_coordinator.get_or_create_coordinator": MagicMock(),
                "modelship.infer.replica_coordinator.get_or_create_replica_coordinator": MagicMock(),
                "modelship.infer.deploy_leases.get_or_create_leases": leases,
                "modelship.deploy.strategy.run_deploy_loop": run_deploy_loop,
                "modelship.deploy.removal.delete_apps_quietly": delete,
                "modelship.metrics.DEPLOY_DURATION_SECONDS": MagicMock(),
                "modelship.metrics.DEPLOY_MODELS_CHANGED_TOTAL": changed,
                "ray.cluster_resources": MagicMock(return_value={}),
            }.items():
                stack.enter_context(patch(target, value))
            try:
                failed, error = driver._apply(args, "g", MagicMock(), {}), None
            except RuntimeError as e:
                failed, error = None, e
        return SimpleNamespace(
            failed=failed,
            error=error,
            submitted=run_deploy_loop.call_args.args[0] if run_deploy_loop.called else None,
            events=events,
            effective=read_effective(store, "g"),
            effective_at_submit=effective_at_submit[0] if effective_at_submit else None,
            leases=leases,
            changed={call.kwargs["tags"]["action"]: call.args[0] for call in changed.inc.call_args_list},
        )

    return run


class TestEffectiveConfig:
    def test_is_written_before_anything_is_submitted(self, apply):
        a = _raw("a")
        r = apply([], [a], {})
        assert r.effective_at_submit == [a]

    def test_keeps_a_model_that_failed(self, apply):
        a = _raw("a")
        r = apply([], [a], {}, outcome=DeployOutcome([], [], [(_config(a), "engine died")]))
        assert r.effective == [a]

    def test_is_left_alone_when_a_model_source_fails(self, apply):
        a, b = _raw("a"), _raw("b")
        r = apply([a], [a, b], {}, sources_error=RuntimeError("repo not found"))
        assert r.error is not None
        assert r.effective == [a]
        assert r.submitted is None


class TestReplacement:
    def test_blue_green_leaves_the_old_app_to_the_replica_coordinator(self, apply):
        old, new = _raw("a", num_cpus=1), _raw("a", num_cpus=2)
        r = apply([old], [new], {_app(old): ApplicationStatus.RUNNING})
        assert r.events == ["deploy"]

    def test_stop_start_deletes_stale_apps_before_deploying(self, apply):
        old, new, dropped = _raw("a", num_cpus=1), _raw("a", num_cpus=2), _raw("b")
        statuses = {_app(old): ApplicationStatus.RUNNING, _app(dropped): ApplicationStatus.RUNNING}
        r = apply([old, dropped], [new], statuses, replace_strategy="stop_start")
        assert r.events == [("delete", sorted([_app(old), _app(dropped)])), "deploy"]


class TestLiveApps:
    def test_a_failed_app_is_deployed_again(self, apply):
        a = _raw("a")
        r = apply([a], [a], {_app(a): ApplicationStatus.DEPLOY_FAILED})
        assert [c.name for c in r.submitted] == ["a"]

    def test_a_live_app_is_not_deployed_again(self, apply):
        a = _raw("a")
        r = apply([a], [a], {_app(a): ApplicationStatus.RUNNING})
        assert r.submitted is None


class TestLeaseStartupWindow:
    def test_skipped_when_this_gateway_is_the_only_app(self, apply):
        r = apply([], [_raw("a")], {"g": ApplicationStatus.RUNNING})
        r.leases.assert_called_once_with(startup_window=False)

    def test_kept_when_a_model_app_exists(self, apply):
        a = _raw("a")
        r = apply([a], [a], {"g": ApplicationStatus.RUNNING, _app(a): ApplicationStatus.RUNNING})
        r.leases.assert_called_once_with(startup_window=True)

    def test_kept_when_serve_status_is_unreadable(self, apply):
        r = apply([], [_raw("a")], {})
        r.leases.assert_called_once_with(startup_window=True)


class TestReporting:
    def test_add_counts_only_models_that_came_up(self, apply):
        a, b = _raw("a"), _raw("b")
        r = apply([], [a, b], {}, outcome=DeployOutcome([_config(a)], [(_config(b), "no room yet")], []))
        assert r.changed == {"add": 1}

    def test_remove_counts_the_stale_apps(self, apply):
        a, b = _raw("a"), _raw("b")
        r = apply([a, b], [a], {_app(a): ApplicationStatus.RUNNING, _app(b): ApplicationStatus.RUNNING})
        assert r.changed == {"remove": 1}

    def test_a_pending_model_without_a_reason_is_logged_by_name_alone(self, apply, caplog):
        a = _raw("a")
        with caplog.at_level("WARNING"):
            apply([], [a], {}, outcome=DeployOutcome([], [(_config(a), "")], []))
        assert "Model 'a' is still coming up and will land on its own" in caplog.messages

    def test_returns_the_models_that_failed(self, apply):
        a = _raw("a")
        r = apply([], [a], {}, outcome=DeployOutcome([], [], [(_config(a), "engine died")]))
        assert [(c.name, reason) for c, reason in r.failed] == [("a", "engine died")]
        assert r.changed == {"fail": 1}
