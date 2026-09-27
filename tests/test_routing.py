"""compute_routing: which app serves each model and what the gateway waits for."""

import pytest
from ray.serve._private.common import ReplicaState
from ray.serve.schema import (
    ApplicationStatus,
    ApplicationStatusOverview,
    DeploymentStatus,
    DeploymentStatusOverview,
    DeploymentStatusTrigger,
)

from modelship.deploy.routing import can_serve, compute_routing, running_replicas

OLD, NEW = "gw.m-aaaaaaaaaa", "gw.m-bbbbbbbbbb"
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
            (_app(ApplicationStatus.RUNNING, running=0), True),
            (_app(ApplicationStatus.RUNNING, running=0, starting=1), True),
            (_app(ApplicationStatus.DEPLOYING, running=1, starting=1), True),
            (_app(ApplicationStatus.DEPLOYING, running=0, starting=1), False),
            (_app(ApplicationStatus.UNHEALTHY, running=2), True),
            (_app(ApplicationStatus.UNHEALTHY, running=0, starting=1), False),
            (_app(ApplicationStatus.DEPLOY_FAILED, running=0), False),
            (_app(ApplicationStatus.DELETING, running=1), False),
            (_app(ApplicationStatus.UNHEALTHY, key=ReplicaState), True),
        ],
    )
    def test_needs_a_running_app_or_replica_and_no_delete(self, app, expected):
        assert can_serve(app) is expected


class TestModelTable:
    def test_a_target_that_can_serve_is_routed(self):
        assert _route({NEW: _app()}, {"m": NEW}).models == {NEW: "m"}

    def test_a_target_scaled_to_zero_is_routed(self):
        assert _route({NEW: _app(running=0)}, {"m": NEW}).models == {NEW: "m"}

    def test_a_target_takes_over_at_its_first_running_replica(self):
        apps = {OLD: _app(), NEW: _app(ApplicationStatus.DEPLOYING, running=1, starting=1, deployed_at=1)}
        assert _route(apps, {"m": NEW}).models == {NEW: "m"}

    def test_a_target_that_cannot_serve_leaves_the_model_out_even_with_another_app(self):
        apps = {OLD: _app(), NEW: _app(ApplicationStatus.UNHEALTHY, running=0, starting=1, deployed_at=1)}
        assert _route(apps, {"m": NEW}).models == {}

    def test_a_missing_target_leaves_the_model_out(self):
        assert _route({OLD: _app()}, {"m": NEW}).models == {}

    def test_an_app_being_deleted_is_not_routed(self):
        assert _route({NEW: _app(ApplicationStatus.DELETING)}, {"m": NEW}).models == {}

    def test_other_gateways_and_non_modelship_apps_are_ignored(self):
        apps = {"edge.m-aaaaaaaaaa": _app(), "dashboard": _app(), "m-aaaaaaaaaa": _app()}
        assert _route(apps, {"m": NEW}).models == {}

    def test_nothing_is_computed_without_the_gateways_own_app(self):
        assert compute_routing("gw", {"m": NEW}, {NEW: _app()}) is None


class TestExpected:
    def test_every_targeted_model_is_expected(self):
        assert _route({}, {"m": NEW, "x": OTHER}).expected == ["m", "x"]

    def test_a_model_whose_target_cannot_serve_is_expected(self):
        assert _route({NEW: _app(ApplicationStatus.DEPLOYING, running=0)}, {"m": NEW}).expected == ["m"]


class TestRunningReplicas:
    def test_counts_running_replicas_across_deployments(self):
        assert running_replicas(_app(running=2, starting=1)) == 2

    def test_none_running(self):
        assert running_replicas(_app(running=0, starting=1)) == 0
