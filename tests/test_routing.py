"""compute_routing: which app serves each model, what the gateway waits for, and what is unused."""

import pytest
from ray.serve._private.common import ReplicaState
from ray.serve.schema import (
    ApplicationStatus,
    ApplicationStatusOverview,
    DeploymentStatus,
    DeploymentStatusOverview,
    DeploymentStatusTrigger,
)

from modelship.deploy.routing import can_serve, compute_routing

OLD, NEW, NEWER = "gw.m-aaaaaaaaaa", "gw.m-bbbbbbbbbb", "gw.m-cccccccccc"
OTHER = "gw.x-dddddddddd"


def _app(status=ApplicationStatus.RUNNING, running=1, deployed_at=0.0, starting=0, key=str):
    states = {key("RUNNING"): running} if running else {}
    if starting:
        states[key("STARTING")] = starting
    deployment = DeploymentStatusOverview(
        status=DeploymentStatus.HEALTHY,
        status_trigger=DeploymentStatusTrigger.CONFIG_UPDATE_COMPLETED,
        replica_states=states,
        message="",
    )
    return ApplicationStatusOverview(
        status=status, message="", last_deployed_time_s=deployed_at, deployments={"d": deployment}
    )


def _route(apps, targets):
    return compute_routing("gw", targets, {"gw": _app()} | apps)


class TestCanServe:
    @pytest.mark.parametrize(
        ("app", "expected"),
        [
            (_app(ApplicationStatus.RUNNING), True),
            (_app(ApplicationStatus.DEPLOYING, running=1, starting=1), True),
            (_app(ApplicationStatus.DEPLOYING, running=0, starting=1), False),
            (_app(ApplicationStatus.UNHEALTHY, running=2), True),
            (_app(ApplicationStatus.UNHEALTHY, running=0, starting=1), False),
            (_app(ApplicationStatus.DEPLOY_FAILED, running=0), False),
            (_app(ApplicationStatus.DELETING, running=1), False),
            (_app(ApplicationStatus.RUNNING, key=ReplicaState), True),
        ],
    )
    def test_needs_a_running_replica_and_no_delete(self, app, expected):
        assert can_serve(app) is expected


class TestModelTable:
    def test_a_target_that_can_serve_is_routed(self):
        assert _route({NEW: _app()}, {"m": NEW}).models == {NEW: "m"}

    def test_a_loading_target_leaves_the_older_app_serving(self):
        apps = {OLD: _app(), NEW: _app(ApplicationStatus.DEPLOYING, running=0, starting=1, deployed_at=1)}
        assert _route(apps, {"m": NEW}).models == {OLD: "m"}

    def test_the_target_takes_over_at_its_first_running_replica(self):
        apps = {OLD: _app(), NEW: _app(ApplicationStatus.DEPLOYING, running=1, starting=1, deployed_at=1)}
        assert _route(apps, {"m": NEW}).models == {NEW: "m"}

    def test_a_target_with_no_running_replica_falls_back_to_an_older_app(self):
        apps = {OLD: _app(), NEW: _app(ApplicationStatus.UNHEALTHY, running=0, starting=1, deployed_at=1)}
        assert _route(apps, {"m": NEW}).models == {OLD: "m"}

    def test_the_newest_older_app_that_can_serve_wins(self):
        apps = {
            OLD: _app(deployed_at=1),
            NEWER: _app(deployed_at=2),
            NEW: _app(ApplicationStatus.DEPLOYING, running=0, deployed_at=3),
        }
        assert _route(apps, {"m": NEW}).models == {NEWER: "m"}

    def test_a_model_with_nothing_serving_is_left_out(self):
        apps = {NEW: _app(ApplicationStatus.DEPLOY_FAILED, running=0)}
        assert _route(apps, {"m": NEW}).models == {}

    def test_an_app_being_deleted_is_not_routed(self):
        assert _route({OLD: _app(ApplicationStatus.DELETING)}, {"m": NEW}).models == {}

    def test_other_gateways_and_non_modelship_apps_are_ignored(self):
        apps = {"edge.m-aaaaaaaaaa": _app(), "dashboard": _app(), "m-aaaaaaaaaa": _app()}
        r = _route(apps, {"m": NEW})
        assert r.models == {}
        assert r.unused == set()

    def test_nothing_is_computed_without_the_gateways_own_app(self):
        assert compute_routing("gw", {"m": NEW}, {NEW: _app()}) is None


class TestUnused:
    def test_the_older_app_is_unused_once_the_target_serves(self):
        assert _route({OLD: _app(), NEW: _app(deployed_at=1)}, {"m": NEW}).unused == {OLD}

    def test_an_older_app_still_serving_is_kept(self):
        apps = {OLD: _app(), NEW: _app(ApplicationStatus.DEPLOYING, running=0)}
        assert _route(apps, {"m": NEW}).unused == set()

    def test_a_target_is_never_unused(self):
        assert _route({NEW: _app(ApplicationStatus.DEPLOY_FAILED, running=0)}, {"m": NEW}).unused == set()

    def test_a_dropped_models_app_is_unused(self):
        assert _route({NEW: _app(), OTHER: _app()}, {"m": NEW}).unused == {OTHER}

    def test_an_app_already_being_deleted_is_not_unused(self):
        assert _route({OTHER: _app(ApplicationStatus.DELETING)}, {"m": NEW}).unused == set()


class TestRetiring:
    def test_an_unused_app_is_retiring(self):
        assert _route({OLD: _app(), NEW: _app(deployed_at=1)}, {"m": NEW}).retiring == {OLD}

    def test_an_app_being_deleted_is_retiring(self):
        assert _route({NEW: _app(), OTHER: _app(ApplicationStatus.DELETING)}, {"m": NEW}).retiring == {OTHER}

    def test_an_older_app_still_serving_is_not_retiring(self):
        apps = {OLD: _app(), NEW: _app(ApplicationStatus.DEPLOYING, running=0)}
        assert _route(apps, {"m": NEW}).retiring == set()

    def test_another_gateways_app_being_deleted_is_not_retiring(self):
        assert _route({"edge.m-aaaaaaaaaa": _app(ApplicationStatus.DELETING)}, {}).retiring == set()


class TestExpected:
    def test_a_model_whose_target_exists_is_expected(self):
        apps = {NEW: _app(ApplicationStatus.DEPLOYING, running=0)}
        assert _route(apps, {"m": NEW}).expected == ["m"]

    def test_a_model_served_only_by_an_older_app_is_expected(self):
        assert _route({OLD: _app()}, {"m": NEW}).expected == ["m"]

    def test_a_model_with_no_app_is_expected(self):
        assert _route({}, {"m": NEW}).expected == ["m"]

    def test_a_model_whose_target_is_being_deleted_is_expected(self):
        assert _route({NEW: _app(ApplicationStatus.DELETING, running=0)}, {"m": NEW}).expected == ["m"]

    def test_a_model_whose_old_app_is_being_deleted_before_its_target_exists_is_expected(self):
        assert _route({OLD: _app(ApplicationStatus.DELETING)}, {"m": NEW}).expected == ["m"]


class TestUnknownTargets:
    def test_each_model_is_served_by_its_newest_app(self):
        apps = {OLD: _app(deployed_at=1), NEW: _app(deployed_at=2), OTHER: _app()}
        r = _route(apps, None)
        assert r.models == {NEW: "m", OTHER: "x"}
        assert r.expected == ["m", "x"]

    def test_nothing_is_unused(self):
        assert _route({OLD: _app(deployed_at=1), NEW: _app(deployed_at=2)}, None).unused == set()

    def test_only_apps_being_deleted_are_retiring(self):
        apps = {OLD: _app(ApplicationStatus.DELETING), NEW: _app(deployed_at=2)}
        assert _route(apps, None).retiring == {OLD}
