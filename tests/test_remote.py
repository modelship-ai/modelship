import http.client
import inspect
import json
import logging
import os
import time
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

    def __init__(self, *jobs, submit_errors=()):
        self.jobs = [list(polls) for polls in jobs]
        self.submit_errors = list(submit_errors)
        self.submitted: list[dict] = []
        self._current: list = []

    def submit_job(self, **kwargs):
        if self.submit_errors:
            raise self.submit_errors.pop(0)
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


def _run(client, argv=("--wait",), config_path=None, env=None, connect_errors=()):
    errors = list(connect_errors)

    def connect(*args, **kwargs):
        if errors:
            raise errors.pop(0)
        return client

    with (
        patch.dict(os.environ, env or {}),
        patch("ray.job_submission.JobSubmissionClient", side_effect=connect) as make_client,
        patch.object(remote, "_wait_for_dashboard"),
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

    def test_no_deploy_coordinator_exits_4_without_submitting_again(self):
        client = _FakeClient([(_info("FAILED", remote.EXIT_NO_DEPLOY_COORDINATOR), "")])
        code, _ = _run(client)
        assert code == remote.EXIT_NO_DEPLOY_COORDINATOR
        assert len(client.submitted) == 1

    def test_losing_the_dashboard_exits_3_without_submitting_again(self, capsys):
        client = _FakeClient([(_info("RUNNING"), ""), requests.ConnectionError()])
        code, _ = _run(client)
        assert code == remote.EXIT_UNREACHABLE
        assert len(client.submitted) == 1
        assert "lost the Ray dashboard at http://head:8265; Ray job raysubmit_1 may still be running" in (
            capsys.readouterr().err
        )

    def test_a_job_that_never_ran_exits_with_rays_message(self):
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
            patch.object(remote, "_wait_for_dashboard"),
            patch.object(remote, "configure_logging"),
            pytest.raises(SystemExit, match="rejected the Ray auth token: check MSHIP_RAY_AUTH_TOKEN"),
        ):
            remote.run(URL, [], None)

    def test_a_5xx_on_submit_is_retried_until_the_dashboard_takes_the_job(self, caplog):
        caplog.set_level(logging.INFO, logger="modelship")
        no_agent = RuntimeError("Request failed with status code 500: No available agent to submit job.")
        unavailable = requests.HTTPError(response=MagicMock(status_code=503))
        client = _FakeClient(
            [(_info("SUCCEEDED", 0), "")], submit_errors=[no_agent, unavailable, requests.ConnectionError()]
        )
        code, _ = _run(client)
        assert code == 0
        assert len(client.submitted) == 1
        assert [m for m in caplog.messages if m.startswith("Waiting for the Ray dashboard")] == [
            f"Waiting for the Ray dashboard at {URL} to take the job ({no_agent})."
        ]

    # Ray's version check, run by the client's constructor and by submit_job, raises the builtin ConnectionError.
    def test_a_dashboard_down_at_connect_or_submit_is_waited_on(self, caplog):
        caplog.set_level(logging.INFO, logger="modelship")
        refused = ConnectionError(f"Failed to connect to Ray at address: {URL}.")
        client = _FakeClient([(_info("SUCCEEDED", 0), "")], submit_errors=[refused])
        code, make_client = _run(client, connect_errors=[refused])
        assert code == 0
        assert make_client.call_count == 3
        assert len(client.submitted) == 1
        assert [m for m in caplog.messages if m.startswith("Waiting for the Ray dashboard")] == [
            f"Waiting for the Ray dashboard at {URL} to take the job (connection failed)."
        ]

    def test_a_dashboard_still_down_at_the_deadline_exits_3(self, capsys):
        refused = ConnectionError(f"Failed to connect to Ray at address: {URL}.")
        with patch.object(remote, "HEAD_WAIT_S", 0):
            code, _ = _run(_FakeClient(), connect_errors=[refused])
        assert code == remote.EXIT_UNREACHABLE
        assert "didn't take the job within 0 min: connection failed" in capsys.readouterr().err

    def test_a_4xx_on_submit_is_not_retried(self):
        client = _FakeClient(submit_errors=[RuntimeError("Request failed with status code 400: bad entrypoint")])
        code, _ = _run(client)
        assert (
            code
            == f"error: the Ray dashboard at {URL} failed the request: Request failed with status code 400: bad entrypoint"
        )

    def test_a_4xx_on_the_version_check_is_not_retried(self):
        client = _FakeClient(
            submit_errors=[requests.HTTPError("400 Client Error", response=MagicMock(status_code=400))]
        )
        code, _ = _run(client)
        assert code == f"error: the Ray dashboard at {URL} failed the request: 400 Client Error"
        assert client.submitted == []

    def test_submit_gives_up_at_the_deadline(self, capsys):
        client = _FakeClient(submit_errors=[RuntimeError("Request failed with status code 500: No available agent.")])
        with patch.object(remote, "HEAD_WAIT_S", 0):
            code, _ = _run(client)
        assert code == remote.EXIT_UNREACHABLE
        assert f"the Ray dashboard at {URL} didn't take the job within 0 min" in capsys.readouterr().err

    def test_a_config_over_the_cap_is_refused(self, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text("#" * remote.MAX_MODELS_YAML_JSON_BYTES)
        code, _ = _run(_FakeClient(), config_path=str(config))
        assert "over 96 KiB once JSON-escaped" in code

    def test_the_cap_counts_escaped_bytes(self, tmp_path):
        config = tmp_path / "models.yaml"
        # 48 KiB raw, 144 KiB escaped.
        config.write_text("# " + "é" * (24 * 1024 - 1))
        code, _ = _run(_FakeClient(), config_path=str(config))
        assert "over 96 KiB once JSON-escaped" in code

    def test_a_config_at_the_cap_is_sent(self, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text("#" * (remote.MAX_MODELS_YAML_JSON_BYTES - 2))
        client = _FakeClient([(_info("SUCCEEDED"), "")])
        code, _ = _run(client, config_path=str(config))
        assert code == 0
        assert len(client.submitted[0]["metadata"][remote.MODELS_YAML_KEY]) == remote.MAX_MODELS_YAML_JSON_BYTES - 2


class TestDashboardDown:
    def _down(self, token=None, body=None, error=None):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(body or {}).encode()
        with patch.object(remote.urllib.request, "urlopen", side_effect=error, return_value=response):
            return remote._dashboard_down(URL, token)

    def test_a_token_cluster_without_a_token_is_refused(self):
        with pytest.raises(SystemExit, match="requires its Ray auth token: set MSHIP_RAY_AUTH_TOKEN"):
            self._down(None, {"authentication_mode": "token"})

    def test_a_token_cluster_with_a_token_is_up(self):
        assert self._down("t0k", {"authentication_mode": "token"}) is None

    def test_a_cluster_without_auth_is_up(self):
        assert self._down(None, {"authentication_mode": "disabled"}) is None

    @pytest.mark.parametrize(
        ("error", "reason"),
        [
            (urllib.error.URLError("Connection refused"), "Connection refused"),
            (urllib.error.HTTPError(URL, 503, "unavailable", {}, None), "HTTP 503"),
            (http.client.RemoteDisconnected(), "connection closed"),
        ],
        ids=["refused", "5xx", "closed"],
    )
    def test_what_may_pass_is_waited_on(self, error, reason):
        assert self._down(error=error) == reason

    @pytest.mark.parametrize(
        ("error", "reason"),
        [
            (http.client.BadStatusLine("\x00\x00"), "not an HTTP reply"),
            (urllib.error.HTTPError(URL, 404, "not found", {}, None), "HTTP 404"),
        ],
        ids=["gcs_port", "4xx"],
    )
    def test_what_wont_pass_exits_3_at_once(self, error, reason, capsys):
        with pytest.raises(SystemExit) as exc:
            self._down(error=error)
        assert exc.value.code == remote.EXIT_UNREACHABLE
        assert f"({reason})" in capsys.readouterr().err


class TestWaitForDashboard:
    def _wait(self, downs, deadline):
        with (
            patch.object(remote, "_dashboard_down", side_effect=downs),
            patch.object(remote.time, "sleep") as sleep,
            patch.object(remote.signal, "signal"),
        ):
            remote._wait_for_dashboard(URL, None, deadline)
        return sleep

    def test_waits_until_the_dashboard_answers(self, caplog):
        caplog.set_level(logging.INFO, logger="modelship")
        sleep = self._wait(["Connection refused", "Connection refused", None], time.monotonic() + 60)
        assert sleep.call_count == 2
        assert caplog.messages == [f"Waiting for the Ray dashboard at {URL} (Connection refused)."]

    def test_gives_up_at_the_deadline_naming_the_gcs_mixup(self, capsys):
        with pytest.raises(SystemExit) as exc:
            self._wait(["Connection refused"], time.monotonic())
        assert exc.value.code == remote.EXIT_UNREACHABLE
        err = capsys.readouterr().err
        assert "(Connection refused; waited 5 min)" in err
        assert "not the GCS address `mship join` takes" in err
