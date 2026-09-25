"""_apply: what the driver writes, deploys and deletes around the deploy loop."""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from ray.serve.schema import ApplicationStatus

from modelship import driver
from modelship.deploy.effective_config import read_effective, write_effective
from modelship.deploy.strategy import DeployOutcome
from modelship.infer.deploy_leases import DeployLeaseError
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
    def run(
        prev,
        desired,
        statuses,
        outcome=None,
        replace_strategy="blue_green",
        sources_error=None,
        reconcile=True,
        on_hold=None,
        statuses_under_lease=None,
        refuse_write=False,
    ):
        store = _MemoryStore()
        write_effective(store, "g", prev)
        events: list = []
        effective_at_submit: list = []
        held = {"now": False}

        def write(raw_models):
            if refuse_write:
                raise DeployLeaseError("lost the deploy lease of gateway 'g' before writing its effective config")
            events.append("write")
            write_effective(store, "g", raw_models)

        @contextmanager
        def gateway_lease(gateway_name):
            events.append("hold")
            held["now"] = True
            if on_hold is not None:
                on_hold(store)
            try:
                yield SimpleNamespace(write_effective=write)
            finally:
                held["now"] = False
                events.append("release")

        def app_statuses():
            return statuses_under_lease if held["now"] and statuses_under_lease is not None else statuses

        def deploy_loop(models, ctx):
            effective_at_submit.append(read_effective(store, "g"))
            events.append("deploy")
            return outcome or DeployOutcome([], [], [])

        run_deploy_loop = MagicMock(side_effect=deploy_loop)
        delete = MagicMock(side_effect=lambda names: events.append(("delete", list(names))))
        coordinator = MagicMock()
        changed = MagicMock()
        args = SimpleNamespace(reconcile=reconcile, config="models.yaml", model=None, replace_strategy=replace_strategy)
        with ExitStack() as stack:
            for target, value in {
                "modelship.deploy.serve_utils.get_app_statuses": app_statuses,
                "modelship.infer.deploy_leases.gateway_lease": gateway_lease,
                "modelship.deploy.config.resolve_input_models": MagicMock(return_value=desired),
                "modelship.deploy.config.resolve_all_model_sources": MagicMock(side_effect=sources_error),
                "modelship.state.get_state_store": MagicMock(return_value=store),
                "modelship.openai.compaction_crypto.ensure_key_seeded": MagicMock(),
                "modelship.infer.deploy_coordinator.get_or_create_coordinator": coordinator,
                "modelship.infer.gateway_coordinator.get_or_create_gateway_coordinator": MagicMock(),
                "modelship.deploy.strategy.run_deploy_loop": run_deploy_loop,
                "modelship.deploy.removal.delete_apps_quietly": delete,
                "modelship.metrics.DEPLOY_DURATION_SECONDS": MagicMock(),
                "modelship.metrics.DEPLOY_MODELS_CHANGED_TOTAL": changed,
                "ray.cluster_resources": MagicMock(return_value={}),
            }.items():
                stack.enter_context(patch(target, value))
            try:
                failed, error = driver._apply(args, "g", MagicMock(), {}), None
            except Exception as e:
                failed, error = None, e
        return SimpleNamespace(
            failed=failed,
            error=error,
            submitted=run_deploy_loop.call_args.args[0] if run_deploy_loop.called else None,
            events=events,
            effective=read_effective(store, "g"),
            effective_at_submit=effective_at_submit[0] if effective_at_submit else None,
            coordinator=coordinator,
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


class TestGatewayLease:
    def test_a_write_made_before_the_lease_was_held_is_merged_onto(self, apply):
        a, b, c = _raw("a"), _raw("b"), _raw("c")
        r = apply([a], [c], {}, reconcile=False, on_hold=lambda store: write_effective(store, "g", [a, b]))
        assert r.effective == [a, b, c]

    def test_plans_from_the_statuses_read_under_the_lease(self, apply):
        a = _raw("a")
        r = apply([], [a], {}, statuses_under_lease={_app(a): ApplicationStatus.RUNNING})
        assert r.submitted is None

    def test_a_lost_lease_writes_and_submits_nothing(self, apply):
        a, b = _raw("a"), _raw("b")
        r = apply([a], [a, b], {}, refuse_write=True)
        assert isinstance(r.error, DeployLeaseError)
        assert r.effective == [a]
        assert r.submitted is None


class TestReplacement:
    def test_blue_green_leaves_the_old_app_to_the_gateway_coordinator(self, apply):
        old, new = _raw("a", num_cpus=1), _raw("a", num_cpus=2)
        r = apply([old], [new], {_app(old): ApplicationStatus.RUNNING})
        assert r.events == ["hold", "write", "release", "deploy"]

    def test_stop_start_deletes_stale_apps_before_deploying(self, apply):
        old, new, dropped = _raw("a", num_cpus=1), _raw("a", num_cpus=2), _raw("b")
        statuses = {_app(old): ApplicationStatus.RUNNING, _app(dropped): ApplicationStatus.RUNNING}
        r = apply([old, dropped], [new], statuses, replace_strategy="stop_start")
        assert r.events == ["hold", "write", ("delete", sorted([_app(old), _app(dropped)])), "release", "deploy"]


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
        r.coordinator.assert_called_once_with(startup_window=False)

    def test_kept_when_a_model_app_exists(self, apply):
        a = _raw("a")
        r = apply([a], [a], {"g": ApplicationStatus.RUNNING, _app(a): ApplicationStatus.RUNNING})
        r.coordinator.assert_called_once_with(startup_window=True)

    def test_kept_when_serve_status_is_unreadable(self, apply):
        r = apply([], [_raw("a")], {})
        r.coordinator.assert_called_once_with(startup_window=True)


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

    def test_the_summary_counts_models_removed_elsewhere(self, apply, caplog):
        a = _raw("a")
        with caplog.at_level("INFO"):
            apply([], [a], {}, outcome=DeployOutcome([], [], [], [_config(a)]))
        assert "Deploy complete: 0 model(s) up, 0 still coming up, 0 failed, 1 removed before coming up." in (
            caplog.messages
        )

    def test_returns_the_models_that_failed(self, apply):
        a = _raw("a")
        r = apply([], [a], {}, outcome=DeployOutcome([], [], [(_config(a), "engine died")]))
        assert [(c.name, reason) for c, reason in r.failed] == [("a", "engine died")]
        assert r.changed == {"fail": 1}
