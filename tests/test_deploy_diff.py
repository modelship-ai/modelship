import pytest
from ray.serve.schema import ApplicationStatus

from modelship.deploy.diff import Rescale, build_diff
from modelship.deploy.ledger import DeployRequest
from modelship.deploy.strategy import ServeApp, serve_scaling
from modelship.infer.infer_config import ModelshipModelConfig


def _raw(name: str, **overrides) -> dict:
    return {"name": name, "model": f"org/{name}", "usecase": "generate", "loader": "llama_server", **overrides}


def _config(raw: dict) -> ModelshipModelConfig:
    return ModelshipModelConfig.model_validate(raw)


def _app_name(raw: dict, gateway: str = "gw") -> str:
    return _config(raw).deployment_name(gateway)


def _held(*raws: dict, status: ApplicationStatus = ApplicationStatus.RUNNING, gateway: str = "gw") -> dict:
    return {_app_name(raw, gateway): ServeApp(status, serve_scaling(_config(raw))) for raw in raws}


def _diff(models, committed=None, apps=None, mode="additive", strategy="blue_green"):
    return build_diff(DeployRequest("gw", mode, strategy, models, {}), committed, apps or {})


A, A2, B = _raw("a", num_cpus=1), _raw("a", num_cpus=2), _raw("b")
C = _raw("c", llama_server_config={"n_ctx": 4096})
C2 = {**C, "num_replicas": 2}
C_AUTO = {**C, "autoscaling_config": {"min_replicas": 1, "max_replicas": 2}}
C_CHANGED = {**C, "num_cpus": 2}


class TestActions:
    def test_a_model_the_gateway_does_not_have_is_added(self):
        diff = _diff([A])
        assert diff.actions == {"a": "add"}
        assert diff.deploys == [_config(A)]

    def test_a_fingerprint_change_replaces_the_app(self):
        diff = _diff([A2], [A], _held(A))
        assert diff.actions == {"a": "replace"}
        assert diff.deploys == [_config(A2)]
        assert diff.retired == [_app_name(A)]

    def test_a_live_app_with_the_wanted_replica_fields_is_kept(self):
        diff = _diff([C2], [C2], _held(C2))
        assert diff.actions == {"c": "keep"}
        assert (diff.deploys, diff.scale_downs, diff.scale_ups, diff.retired) == ([], [], [], [])

    @pytest.mark.parametrize("status", [None, ApplicationStatus.DEPLOY_FAILED, ApplicationStatus.DELETING])
    def test_an_app_serve_is_not_running_is_redeployed(self, status):
        apps = _held(C, status=status) if status else {}
        diff = _diff([C2], [C], apps)
        assert diff.actions == {"c": "redeploy"}
        assert diff.deploys == [_config(C2)]
        assert (diff.scale_downs, diff.scale_ups) == ([], [])

    def test_a_fingerprint_change_replaces_whatever_the_replica_count(self):
        changed = {**C_CHANGED, "num_replicas": 2}
        diff = _diff([changed], [C], _held(C))
        assert diff.actions == {"c": "replace"}
        assert (diff.scale_downs, diff.scale_ups) == ([], [])

    def test_reconcile_removes_the_models_the_request_leaves_out(self):
        diff = _diff([A2], [A, B], _held(A, B), mode="reconcile")
        assert diff.actions == {"a": "replace", "b": "remove"}
        assert diff.retired == sorted([_app_name(A), _app_name(B)])

    def test_additive_leaves_the_models_the_request_leaves_out(self):
        diff = _diff([A2], [A, B], _held(A, B))
        assert diff.actions == {"a": "replace"}
        assert diff.retired == [_app_name(A)]
        assert "keep, not in the request: b" in diff.lines()

    def test_a_bare_request_redeploys_and_rescales_towards_the_committed_version(self):
        diff = _diff(None, [A, B, C], _held(A, C2), mode="bare")
        assert diff.actions == {"a": "keep", "b": "redeploy", "c": "rescale"}
        assert diff.retired == []


class TestRescale:
    @pytest.mark.parametrize("wanted", [C2, C_AUTO])
    def test_a_raised_replica_limit_is_a_scale_up(self, wanted):
        diff = _diff([wanted], [C], _held(C))
        assert diff.actions == {"c": "rescale"}
        assert diff.scale_ups == [Rescale("c", _app_name(C), _config(wanted).scaling())]
        assert diff.scale_downs == []

    @pytest.mark.parametrize("wanted", [C, C_AUTO])
    def test_no_raised_replica_limit_is_a_scale_down(self, wanted):
        diff = _diff([wanted], [C2], _held(C2))
        assert diff.scale_downs == [Rescale("c", _app_name(C), _config(wanted).scaling())]
        assert diff.scale_ups == []

    def test_an_autoscaling_tunable_alone_is_a_rescale(self):
        tuned = {**C, "autoscaling_config": {"min_replicas": 1, "max_replicas": 2, "upscale_delay_s": 5}}
        diff = _diff([tuned], [C_AUTO], _held(C_AUTO))
        assert [rescale.app for rescale in diff.scale_downs] == [_app_name(C)]
        assert diff.lines()[0] == "2. rescale c: autoscaling_config.upscale_delay_s: 30.0 -> 5.0"

    def test_the_replica_fields_are_compared_with_serve_not_the_committed_version(self):
        diff = _diff([C], [C], _held(C2))
        assert diff.scale_downs == [Rescale("c", _app_name(C), _config(C).scaling())]
        assert diff.commits is False
        assert diff.lines() == ["2. rescale c: num_replicas: 2 -> 1 (Serve differs from the committed version)"]

    def test_a_live_app_without_a_deployment_in_serve_is_rescaled_last(self):
        apps = {_app_name(C): ServeApp(ApplicationStatus.RUNNING, None)}
        assert [rescale.app for rescale in _diff([C2], [C], apps).scale_ups] == [_app_name(C)]


class TestShrink:
    def test_a_replaced_app_gets_a_lower_replica_count_before_its_replacement(self):
        diff = _diff([C_CHANGED], [C2], _held(C2))
        assert diff.actions == {"c": "replace"}
        assert diff.scale_downs == [Rescale("c", _app_name(C2), _config(C_CHANGED).scaling())]
        assert diff.lines()[0] == "2. rescale c's replaced app: num_replicas: 2 -> 1"

    def test_stop_start_deletes_the_replaced_app_instead(self):
        assert _diff([C_CHANGED], [C2], _held(C2), strategy="stop_start").scale_downs == []

    @pytest.mark.parametrize("wanted", [{**C_CHANGED, "num_replicas": 2}, {**C_CHANGED, "num_replicas": 3}])
    def test_a_replica_count_that_does_not_drop_leaves_the_replaced_app(self, wanted):
        assert _diff([wanted], [C2], _held(C2)).scale_downs == []

    def test_a_failed_replaced_app_is_left(self):
        apps = _held(C2, status=ApplicationStatus.DEPLOY_FAILED)
        assert _diff([C_CHANGED], [C2], apps).scale_downs == []


class TestLeftovers:
    def test_the_gateways_apps_the_committed_version_does_not_name(self):
        apps = _held(A, A2) | _held(B, gateway="edge") | {"gw": ServeApp(ApplicationStatus.RUNNING, None)}
        assert _diff([A], [A], apps).leftovers == [_app_name(A2)]

    def test_every_model_app_without_a_committed_version(self):
        assert _diff(None, None, _held(A, B), mode="bare").leftovers == sorted([_app_name(A), _app_name(B)])

    def test_a_leftover_of_an_added_models_name_is_deleted_and_the_model_added(self):
        diff = _diff([A], None, _held(A))
        assert diff.leftovers == [_app_name(A)]
        assert diff.actions == {"a": "add"}


class TestCommits:
    def test_reconcile_commits_the_request(self):
        diff = _diff([B], [A], mode="reconcile")
        assert (diff.commits, diff.models) == (True, [B])

    def test_additive_commits_the_request_merged_into_the_committed_version(self):
        diff = _diff([A2], [A, B])
        assert (diff.commits, diff.models) == (True, [B, A2])

    def test_a_first_request_commits_its_models(self):
        diff = _diff([A])
        assert (diff.commits, diff.models) == (True, [A])

    def test_a_request_that_changes_no_app_commits_nothing(self):
        diff = _diff([A], [A, B], _held(A, B))
        assert (diff.commits, diff.models) == (False, [A, B])

    @pytest.mark.parametrize("wanted", [C2, C_AUTO])
    def test_a_replica_field_change_is_committed(self, wanted):
        diff = _diff([wanted], [C, B], _held(C, B))
        assert (diff.commits, diff.models) == (True, [wanted, B])

    def test_a_bare_request_commits_nothing(self):
        diff = _diff(None, [A], mode="bare")
        assert (diff.commits, diff.models) == (False, [A])


class TestLines:
    def test_the_actions_in_the_order_they_run(self):
        apps = _held(A, B, C2) | _held(_raw("stray"))
        diff = _diff([A2, C, _raw("d"), _raw("e")], [A, B, C2, _raw("e")], apps | _held(_raw("e")), mode="reconcile")
        assert diff.lines() == [
            f"1. delete leftover apps: {_app_name(_raw('stray'))}",
            "2. rescale c: num_replicas: 2 -> 1",
            "4. replace a: num_cpus: 1.0 -> 2.0",
            "4. add d",
            "6. switch the gateway and commit",
            "7. remove b",
            "keep: e",
        ]

    def test_stop_start_removes_before_the_new_apps(self):
        diff = _diff([A2], [A, B], _held(A, B), mode="reconcile", strategy="stop_start")
        assert diff.lines() == ["3. remove b", "4. replace a: num_cpus: 1.0 -> 2.0", "6. switch the gateway and commit"]

    def test_a_redeploy_says_what_serve_has(self):
        failed = _held(A, status=ApplicationStatus.DEPLOY_FAILED)
        assert _diff(None, [A, B], failed, mode="bare").lines() == [
            "4. redeploy a (DEPLOY_FAILED in Serve)",
            "4. redeploy b (missing from Serve)",
        ]

    def test_a_change_between_a_fixed_count_and_autoscaling(self):
        assert _diff([C_AUTO], [C2], _held(C2)).lines()[0] == "2. rescale c: num_replicas 2 -> autoscaling_config 1..2"
