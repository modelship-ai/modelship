"""_apply: what the driver deploys, routes and removes around the deploy loop."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from ray.serve.schema import ApplicationStatus

from modelship import driver
from modelship.deploy.effective_config import write_effective
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
    def run(prev, desired, statuses, routed=None, outcome=None, replace_strategy="blue_green"):
        store = _MemoryStore()
        write_effective(store, "g", prev)
        replica_coord = MagicMock()
        replica_coord.get_routing.remote.return_value = {"models": routed or {}}
        run_deploy_loop = MagicMock(return_value=outcome or DeployOutcome([], [], []))
        remove_apps = MagicMock()
        leases = MagicMock()
        changed = MagicMock()
        args = SimpleNamespace(reconcile=True, config="models.yaml", model=None, replace_strategy=replace_strategy)
        with ExitStack() as stack:
            for target, value in {
                "modelship.deploy.serve_utils.get_app_statuses": MagicMock(return_value=statuses),
                "modelship.deploy.serve_utils.seed_expected_models": MagicMock(),
                "modelship.deploy.config.resolve_input_models": MagicMock(return_value=desired),
                "modelship.deploy.config.resolve_all_model_sources": MagicMock(),
                "modelship.state.get_state_store": MagicMock(return_value=store),
                "modelship.openai.compaction_crypto.ensure_key_seeded": MagicMock(),
                "modelship.infer.deploy_coordinator.get_or_create_coordinator": MagicMock(),
                "modelship.infer.replica_coordinator.get_or_create_replica_coordinator": MagicMock(
                    return_value=replica_coord
                ),
                "modelship.infer.deploy_leases.get_or_create_leases": leases,
                "modelship.deploy.strategy.run_deploy_loop": run_deploy_loop,
                "modelship.deploy.removal.remove_apps": remove_apps,
                "modelship.metrics.DEPLOY_DURATION_SECONDS": MagicMock(),
                "modelship.metrics.DEPLOY_MODELS_CHANGED_TOTAL": changed,
                "ray.cluster_resources": MagicMock(return_value={}),
                "ray.get": MagicMock(side_effect=lambda ref, **kwargs: ref),
            }.items():
                stack.enter_context(patch(target, value))
            failed = driver._apply(args, "g", MagicMock(), {})
        return SimpleNamespace(
            failed=failed,
            submitted=run_deploy_loop.call_args.args[0] if run_deploy_loop.called else None,
            removed=[name for call in remove_apps.call_args_list for name in call.args[0]],
            replica_coord=replica_coord,
            leases=leases,
            changed={call.kwargs["tags"]["action"]: call.args[0] for call in changed.inc.call_args_list},
        )

    return run


class TestBlueGreen:
    def test_an_old_app_serving_a_pending_replacement_is_kept(self, apply):
        old, new = _raw("a", num_cpus=1), _raw("a", num_cpus=2)
        r = apply(
            [old],
            [new],
            {_app(old): ApplicationStatus.RUNNING},
            routed={_app(old): "a"},
            outcome=DeployOutcome([], [(_config(new), "no room yet")], []),
        )
        assert r.removed == []

    def test_an_old_app_is_removed_once_its_replacement_is_up(self, apply):
        old, new = _raw("a", num_cpus=1), _raw("a", num_cpus=2)
        r = apply(
            [old],
            [new],
            {_app(old): ApplicationStatus.RUNNING},
            routed={_app(old): "a"},
            outcome=DeployOutcome([_config(new)], [], []),
        )
        assert r.removed == [_app(old)]

    def test_stop_start_removes_the_old_app_before_deploying(self, apply):
        old, new = _raw("a", num_cpus=1), _raw("a", num_cpus=2)
        r = apply(
            [old],
            [new],
            {_app(old): ApplicationStatus.RUNNING},
            routed={_app(old): "a"},
            outcome=DeployOutcome([], [(_config(new), "no room yet")], []),
            replace_strategy="stop_start",
        )
        assert r.removed == [_app(old)]


class TestLiveApps:
    def test_a_failed_app_is_deployed_again(self, apply):
        a = _raw("a")
        r = apply([a], [a], {_app(a): ApplicationStatus.DEPLOY_FAILED})
        assert [c.name for c in r.submitted] == ["a"]

    def test_a_live_unrouted_app_is_routed_without_redeploying(self, apply):
        a = _raw("a")
        r = apply([a], [a], {_app(a): ApplicationStatus.RUNNING})
        assert r.submitted is None
        r.replica_coord.register_deployment.remote.assert_called_once_with("g", _app(a), "a")


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

    def test_returns_the_models_that_failed(self, apply):
        a = _raw("a")
        r = apply([], [a], {}, outcome=DeployOutcome([], [], [(_config(a), "engine died")]))
        assert [(c.name, reason) for c, reason in r.failed] == [("a", "engine died")]
        assert r.changed == {"fail": 1}
