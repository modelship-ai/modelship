import inspect
import json
import logging
import os
import urllib.error
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

from modelship import remote

URL = "http://head:8265"


def _info(status, exit_code=None, message=""):
    return SimpleNamespace(status=status, driver_exit_code=exit_code, message=message)


class _FakeClient:
    """Plays back (info, logs) per poll, per submitted job; an exception entry is raised instead."""

    def __init__(self, *jobs):
        self.jobs = [list(polls) for polls in jobs]
        self.submitted: list[dict] = []
        self._current: list = []

    def submit_job(self, **kwargs):
        self.submitted.append(kwargs)
        self._current = self.jobs[len(self.submitted) - 1]
        return f"raysubmit_{len(self.submitted)}"

    def get_job_info(self, job_id):
        entry = self._current.pop(0)
        if isinstance(entry, Exception):
            raise entry
        self._last_logs = entry[1]
        return entry[0]

    def get_job_logs(self, job_id):
        return self._last_logs


def _run(client, argv=("--wait",), config_path=None, env=None):
    with (
        patch.dict(os.environ, env or {}),
        patch("ray.job_submission.JobSubmissionClient", return_value=client) as make_client,
        patch.object(remote, "_check_dashboard"),
        patch.object(remote, "configure_logging"),
        patch.object(remote.time, "sleep"),
        patch.object(remote.signal, "signal"),
        pytest.raises(SystemExit) as exc,
    ):
        if env is None or "MSHIP_RAY_AUTH_TOKEN" not in env:
            os.environ.pop("MSHIP_RAY_AUTH_TOKEN", None)
        remote.run(URL, list(argv), config_path)
    return exc.value.code, make_client


class TestHeadArgv:
    @pytest.mark.parametrize(
        "argv",
        [
            ["--ray-dashboard-url", URL, "--config", "m.yaml", "--wait"],
            [f"--ray-dashboard-url={URL}", "--config=m.yaml", "--wait"],
        ],
    )
    def test_drops_the_client_flags(self, argv):
        assert remote.head_argv(argv, ships_config=False) == ["--wait"]

    def test_a_shipped_config_is_read_from_the_job(self):
        assert remote.head_argv(["--config", "m.yaml", "--reconcile"], ships_config=True) == [
            "--reconcile",
            "--config-from-job",
        ]

    def test_entrypoint_runs_the_heads_engine_and_quotes_the_args(self):
        entry = remote.entrypoint(["--model", "org/repo:*Q4_K_M.gguf", "--wait"], ships_config=False)
        assert entry == (
            'MSHIP_RAY_DASHBOARD_URL= "${MSHIP_ENGINE_PYTHON:?this head was not started by mship start}" '
            "-m modelship.launcher deploy --model 'org/repo:*Q4_K_M.gguf' --wait"
        )


class TestJobModelsYaml:
    def test_reads_the_jobs_metadata(self):
        config = {"metadata": {remote.MODELS_YAML_KEY: "models: []\n"}}
        with patch.dict(os.environ, {remote.RAY_JOB_CONFIG_ENV_VAR: json.dumps(config)}):
            assert remote.job_models_yaml() == "models: []\n"

    def test_outside_a_job_is_an_error(self):
        with patch.dict(os.environ):
            os.environ.pop(remote.RAY_JOB_CONFIG_ENV_VAR, None)
            with pytest.raises(ValueError, match="not a Ray job"):
                remote.job_models_yaml()

    def test_ray_passes_metadata_to_the_driver_under_this_name(self):
        from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR
        from ray.dashboard.modules.job import job_supervisor

        assert remote.RAY_JOB_CONFIG_ENV_VAR == RAY_JOB_CONFIG_JSON_ENV_VAR
        assert '"metadata": self._metadata' in inspect.getsource(job_supervisor.JobSupervisor._get_driver_env_vars)


class TestRun:
    def test_a_successful_deploy_exits_0_printing_the_log_once(self, capsys):
        client = _FakeClient(
            [(_info("PENDING"), ""), (_info("RUNNING"), "a\n"), (_info("SUCCEEDED", 0), "a\nb\n")],
        )
        code, _ = _run(client)
        assert code == 0
        assert capsys.readouterr().out == "a\nb\n"
        (submitted,) = client.submitted
        assert submitted["entrypoint_resources"] == {"node:__internal_head__": 0.001}
        assert submitted["metadata"] is None

    def test_the_config_rides_the_jobs_metadata(self, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text("models: []\n")
        client = _FakeClient([(_info("SUCCEEDED", 0), "")])
        _run(client, argv=["--config", str(config)], config_path=str(config))
        (submitted,) = client.submitted
        assert submitted["metadata"] == {remote.MODELS_YAML_KEY: "models: []\n"}
        assert submitted["entrypoint"].endswith("deploy --config-from-job")

    def test_a_failed_deploy_exits_with_its_code(self):
        client = _FakeClient([(_info("FAILED", 1), "boom\n")])
        code, _ = _run(client)
        assert code == 1
        assert len(client.submitted) == 1

    def test_a_job_that_died_with_the_head_is_submitted_again(self, caplog):
        caplog.set_level(logging.INFO, logger="modelship")
        client = _FakeClient(
            [(_info("RUNNING"), ""), requests.ConnectionError(), (_info("FAILED"), "")],
            [(_info("SUCCEEDED", 0), "")],
        )
        code, _ = _run(client)
        assert code == 0
        assert len(client.submitted) == 2
        assert any("submitting it again" in m for m in caplog.messages)

    def test_a_job_the_restarted_head_no_longer_knows_is_submitted_again(self):
        gone = RuntimeError("Request failed with status code 404: Job raysubmit_1 does not exist.")
        client = _FakeClient(
            [(_info("RUNNING"), ""), requests.ConnectionError(), gone],
            [(_info("SUCCEEDED", 0), "")],
        )
        code, _ = _run(client)
        assert code == 0
        assert len(client.submitted) == 2

    def test_a_head_still_starting_is_submitted_again(self):
        client = _FakeClient(
            [(_info("FAILED", remote.EXIT_NO_DEPLOY_COORDINATOR), "")],
            [(_info("SUCCEEDED", 0), "")],
        )
        code, _ = _run(client)
        assert code == 0
        assert len(client.submitted) == 2

    def test_a_head_that_never_finishes_starting_exits_with_its_code(self):
        starting = [(_info("FAILED", remote.EXIT_NO_DEPLOY_COORDINATOR), "")]
        client = _FakeClient(starting, list(starting), list(starting))
        with patch.object(remote.time, "monotonic", side_effect=[100.0, 130.0, 161.0]):
            code, _ = _run(client)
        assert code == remote.EXIT_NO_DEPLOY_COORDINATOR
        assert len(client.submitted) == 3

    def test_a_job_that_never_started_is_not_submitted_again(self):
        client = _FakeClient([(_info("FAILED", None, "Argument list too long"), "")])
        code, _ = _run(client)
        assert code == "error: Ray job raysubmit_1 failed: Argument list too long"
        assert len(client.submitted) == 1

    def test_the_token_is_an_explicit_header_and_ray_auth_mode_is_cleared(self):
        client = _FakeClient([(_info("SUCCEEDED", 0), "")])
        env = {"MSHIP_RAY_AUTH_TOKEN": "t0k", "RAY_AUTH_MODE": "token"}
        with patch.dict(os.environ):
            _, make_client = _run(client, env=env)
            assert "RAY_AUTH_MODE" not in os.environ
        assert make_client.call_args.kwargs["headers"] == {"Authorization": "Bearer t0k"}

    def test_no_token_sends_no_header(self):
        client = _FakeClient([(_info("SUCCEEDED", 0), "")])
        _, make_client = _run(client)
        assert make_client.call_args.kwargs["headers"] == {}

    def test_a_rejected_token_names_the_env_var(self):
        rejected = requests.HTTPError(response=MagicMock(status_code=403))
        with (
            patch.dict(os.environ, {"MSHIP_RAY_AUTH_TOKEN": "wrong"}),
            patch("ray.job_submission.JobSubmissionClient", side_effect=rejected),
            patch.object(remote, "_check_dashboard"),
            patch.object(remote, "configure_logging"),
            pytest.raises(SystemExit, match="rejected the Ray auth token: check MSHIP_RAY_AUTH_TOKEN"),
        ):
            remote.run(URL, [], None)

    def test_a_config_over_the_cap_is_refused(self, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text("#" * (remote.MAX_MODELS_YAML_BYTES + 1))
        code, _ = _run(_FakeClient(), config_path=str(config))
        assert "over 96 KiB" in code


class TestCheckDashboard:
    def _check(self, token, body=None, error=None):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(body or {}).encode()
        with patch.object(remote.urllib.request, "urlopen", side_effect=error, return_value=response):
            remote._check_dashboard(URL, token)

    def test_a_token_cluster_without_a_token_is_refused(self):
        with pytest.raises(SystemExit, match="requires its Ray auth token: set MSHIP_RAY_AUTH_TOKEN"):
            self._check(None, {"authentication_mode": "token"})

    def test_a_token_cluster_with_a_token_passes(self):
        self._check("t0k", {"authentication_mode": "token"})

    def test_a_cluster_without_auth_passes(self):
        self._check(None, {"authentication_mode": "disabled"})

    def test_no_dashboard_exits_3_naming_the_gcs_mixup(self, capsys):
        with pytest.raises(SystemExit) as exc:
            self._check(None, error=urllib.error.URLError("Connection refused"))
        assert exc.value.code == remote.EXIT_UNREACHABLE
        assert "not the GCS address `mship join` takes" in capsys.readouterr().err
