"""Tests for the start/join/deploy/stop CLI parsing and driver helpers."""

import json
import logging
import os
import signal
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from ray.exceptions import AuthenticationError, RayActorError

from modelship.deploy.actor_options import (
    build_cache_env_vars,
    build_deployment_options,
    total_gpu_reservation,
)
from modelship.deploy.gateway_sizing import GATEWAY_SIZING_ENV_VARS, gateway_sizing
from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase, VllmEngineConfig
from modelship.utils import parse_memory_bytes, rand_suffix
from modelship.utils.cli import apply_args_to_env, parse_args
from modelship.utils.runtime_env import CLUSTER_ENV_DEFAULTS, NODE_ENV_DEFAULTS


class TestParseMemoryBytes:
    def test_bare_bytes(self):
        assert parse_memory_bytes("1024") == 1024

    def test_ki_suffix(self):
        assert parse_memory_bytes("4Ki") == 4 * 1024

    def test_mi_suffix(self):
        assert parse_memory_bytes("512Mi") == 512 * 1024**2

    def test_gi_suffix(self):
        assert parse_memory_bytes("8Gi") == 8 * 1024**3

    def test_ti_suffix(self):
        assert parse_memory_bytes("2Ti") == 2 * 1024**4

    def test_case_insensitive(self):
        assert parse_memory_bytes("8gi") == 8 * 1024**3
        assert parse_memory_bytes("8GI") == 8 * 1024**3

    def test_whitespace_tolerant(self):
        assert parse_memory_bytes(" 8Gi ") == 8 * 1024**3
        assert parse_memory_bytes("8 Gi") == 8 * 1024**3

    def test_rejects_decimal_units(self):
        with pytest.raises(ValueError, match="Invalid memory size"):
            parse_memory_bytes("8GB")

    def test_rejects_garbage(self):
        with pytest.raises(ValueError, match="Invalid memory size"):
            parse_memory_bytes("not-a-size")

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="Invalid memory size"):
            parse_memory_bytes("-8Gi")


class TestParseArgs:
    @pytest.mark.parametrize("command", ["start", "deploy"])
    def test_defaults(self, command):
        args = parse_args(command, [])
        assert args.config is None
        assert args.reconcile is False
        assert args.gateway_name is None

    def test_reconcile_flag(self):
        args = parse_args("deploy", ["--reconcile"])
        assert args.reconcile is True
        assert args.replace_strategy is None

    def test_reconcile_with_stop_start_strategy(self):
        args = parse_args("deploy", ["--reconcile", "--replace-strategy", "stop_start"])
        assert args.reconcile is True
        assert args.replace_strategy == "stop_start"

    @pytest.mark.parametrize(
        ("command", "argv", "attr", "expected"),
        [
            ("start", ["--config", "/some/path/models.yaml"], "config", "/some/path/models.yaml"),
            ("start", ["--gateway-min-replicas", "2"], "gateway_min_replicas", 2),
            ("start", ["--gateway-max-replicas", "8"], "gateway_max_replicas", 8),
            ("start", ["--gateway-target-ongoing-requests", "32"], "gateway_target_ongoing_requests", 32.0),
            ("start", ["--gateway-max-ongoing-requests", "256"], "gateway_max_ongoing_requests", 256),
            ("deploy", ["--gateway-name", "my-gateway"], "gateway_name", "my-gateway"),
            ("start", ["--enable-ray-auth"], "enable_ray_auth", True),
            ("start", ["--gcs-port", "6380"], "gcs_port", 6380),
            ("start", ["--ray-dashboard-port", "8266"], "ray_dashboard_port", 8266),
            ("start", ["--ray-dashboard-host", "0.0.0.0"], "ray_dashboard_host", "0.0.0.0"),
            ("start", ["--metrics-port", "9090"], "metrics_port", 9090),
            ("join", ["--gcs-address", "h:1", "--metrics-port", "9090"], "metrics_port", 9090),
            ("join", ["--gcs-address", "mship-head:6380"], "gcs_address", "mship-head:6380"),
            ("start", ["--node-num-cpus", "4"], "node_num_cpus", 4),
            ("join", ["--gcs-address", "h:1", "--node-num-gpus", "2"], "node_num_gpus", 2),
            ("join", ["--gcs-address", "h:1", "--node-memory", "8Gi"], "node_memory", 8 * 1024**3),
            ("deploy", ["--responses-ttl-s", "60"], "responses_ttl_s", 60.0),
            ("start", ["--state-sweep-interval-s", "30"], "state_sweep_interval_s", 30.0),
            ("start", ["--log-format", "json"], "log_format", "json"),
            ("start", ["--otel-endpoint", "http://c:4317"], "otel_endpoint", "http://c:4317"),
            ("join", ["--gcs-address", "h:1", "--log-level", "debug"], "log_level", "DEBUG"),
            ("join", ["--gcs-address", "h:1", "--log-target", "syslog://h:514"], "log_target", "syslog://h:514"),
            ("deploy", ["--cancel", "abc123"], "cancel", "abc123"),
        ],
    )
    def test_flag_parses(self, command, argv, attr, expected):
        assert getattr(parse_args(command, argv), attr) == expected

    @pytest.mark.parametrize(
        ("command", "argv"),
        [
            ("deploy", ["--gcs-port", "6380"]),
            ("deploy", ["--gateway-max-replicas", "8"]),
            ("join", ["--gcs-address", "h:1", "--gateway-min-replicas", "2"]),
            ("deploy", ["--node-num-cpus", "4"]),
            ("deploy", ["--prune-ray-sessions", "false"]),
            ("deploy", ["--gcs-address", "h:1"]),
            ("start", ["--gcs-address", "h:1"]),
            ("start", ["--token", "secret"]),
            ("start", ["--replace-strategy", "stop_start"]),
            ("join", ["--gcs-address", "h:1", "--config", "models.yaml"]),
            ("join", ["--gcs-address", "h:1", "--model", "org/repo"]),
            ("join", ["--gcs-address", "h:1", "--gcs-port", "6380"]),
            ("join", ["--gcs-address", "h:1", "--ray-dashboard-host", "0.0.0.0"]),
            ("deploy", ["--metrics-port", "9090"]),
            ("join", ["--gcs-address", "h:1", "--state-store", "redis://h:6379/0"]),
            ("deploy", ["--state-store", "redis://h:6379/0"]),
            ("deploy", ["--state-sweep-interval-s", "30"]),
            ("deploy", ["--no-metrics"]),
            ("deploy", ["--log-format", "json"]),
            ("join", ["--gcs-address", "h:1", "--log-format", "json"]),
            ("deploy", ["--log-level", "DEBUG"]),
            ("deploy", ["--deploy-id", "a"]),
            ("start", ["--cancel", "a"]),
            ("deploy", ["--enable-ray-auth"]),
            ("join", ["--gcs-address", "h:1", "--enable-ray-auth"]),
        ],
    )
    def test_flag_owned_by_another_command_is_rejected(self, command, argv):
        with pytest.raises(SystemExit):
            parse_args(command, argv)

    @pytest.mark.parametrize("command", ["start", "join", "deploy"])
    @pytest.mark.parametrize(
        "flag",
        [
            "--use-existing-ray-cluster",
            "--address=h:1",
            "--deploy-timeout=5",
            "--gateway-replicas=2",
            "--api-keys=k1",
            "--cluster=h:1",
            "--ray-port=6380",
            "--ray-auth=token",
            "--token=secret",
        ],
    )
    def test_removed_flags_are_rejected(self, command, flag):
        with pytest.raises(SystemExit):
            parse_args(command, [flag])

    @pytest.mark.parametrize(
        "argv",
        [
            ["--config", "models.yaml"],
            ["--model", "org/repo"],
            ["--name", "qwen"],
            ["--reconcile"],
            ["--replace-strategy", "stop_start"],
            ["--no-preflight"],
            ["--llama-server-config.n-ctx", "8192"],
        ],
    )
    def test_cancel_takes_no_model_options(self, argv, capsys):
        with pytest.raises(SystemExit):
            parse_args("deploy", ["--cancel", "r1", *argv])
        assert f"--cancel takes no model options: {argv[0]}." in capsys.readouterr().err

    def test_ray_dashboard_url_takes_a_url(self, capsys):
        assert parse_args("deploy", ["--ray-dashboard-url", "https://ray.internal"]).ray_dashboard_url == (
            "https://ray.internal"
        )
        with pytest.raises(SystemExit):
            parse_args("deploy", ["--ray-dashboard-url", "head:8265"])
        assert "--ray-dashboard-url takes a URL, e.g. http://head:8265" in capsys.readouterr().err

    def test_ray_dashboard_url_env_takes_a_url(self, capsys):
        with patch.dict(os.environ, {"MSHIP_RAY_DASHBOARD_URL": "head:8265"}), pytest.raises(SystemExit):
            parse_args("deploy", [])
        assert "--ray-dashboard-url takes a URL, e.g. http://head:8265" in capsys.readouterr().err

    def test_flags_are_never_abbreviated(self):
        with pytest.raises(SystemExit):
            parse_args("deploy", ["--conf", "models.yaml"])

    @pytest.mark.parametrize("argv", [["--config", "m.yaml"], ["--model", "org/repo"]])
    def test_config_from_job_takes_no_other_config(self, argv):
        with pytest.raises(SystemExit):
            parse_args("deploy", ["--config-from-job", *argv])

    def test_join_requires_a_gcs_address(self, monkeypatch):
        monkeypatch.delenv("MSHIP_GCS_ADDRESS", raising=False)
        with pytest.raises(SystemExit):
            parse_args("join", [])

    def test_join_takes_the_gcs_address_from_env(self, monkeypatch):
        monkeypatch.setenv("MSHIP_GCS_ADDRESS", "mship-head:6380")
        assert parse_args("join", []).gcs_address is None


class TestApplyArgsToEnv:
    @pytest.mark.parametrize(
        ("command", "argv", "env_var", "expected"),
        [
            ("start", ["--state-store", "redis://cache:6379/0"], "MSHIP_STATE_STORE", "redis://cache:6379/0"),
            ("start", ["--gateway-min-replicas", "2"], "MSHIP_GATEWAY_MIN_REPLICAS", "2"),
            ("start", ["--gateway-max-replicas", "8"], "MSHIP_GATEWAY_MAX_REPLICAS", "8"),
            ("start", ["--gateway-target-ongoing-requests", "32"], "MSHIP_GATEWAY_TARGET_ONGOING_REQUESTS", "32.0"),
            ("start", ["--gateway-max-ongoing-requests", "256"], "MSHIP_GATEWAY_MAX_ONGOING_REQUESTS", "256"),
            ("start", ["--enable-ray-auth"], "MSHIP_RAY_AUTH", "true"),
            ("start", ["--gcs-port", "6380"], "MSHIP_GCS_PORT", "6380"),
            ("start", ["--ray-dashboard-port", "8266"], "MSHIP_RAY_DASHBOARD_PORT", "8266"),
            ("start", ["--ray-dashboard-host", "0.0.0.0"], "MSHIP_RAY_DASHBOARD_HOST", "0.0.0.0"),
            ("join", ["--gcs-address", "h:1", "--metrics-port", "9090"], "MSHIP_METRICS_PORT", "9090"),
            ("join", ["--gcs-address", "mship-head:6380"], "MSHIP_GCS_ADDRESS", "mship-head:6380"),
            ("join", ["--gcs-address", "h:1", "--node-num-cpus", "4"], "MSHIP_NODE_NUM_CPUS", "4"),
            ("start", ["--node-num-gpus", "2"], "MSHIP_NODE_NUM_GPUS", "2"),
            ("start", ["--node-memory", "8Gi"], "MSHIP_NODE_MEMORY", str(8 * 1024**3)),
            ("join", ["--gcs-address", "h:1", "--prune-ray-sessions", "false"], "MSHIP_PRUNE_RAY_SESSIONS", "false"),
            ("start", ["--no-preflight"], "MSHIP_PREFLIGHT", "false"),
            ("start", ["--no-metrics"], "MSHIP_METRICS", "false"),
            ("join", ["--gcs-address", "h:1", "--log-level", "TRACE"], "MSHIP_LOG_LEVEL", "TRACE"),
            ("deploy", ["--responses-ttl-s", "60"], "MSHIP_RESPONSES_TTL_S", "60.0"),
            ("start", ["--state-sweep-interval-s", "30"], "MSHIP_STATE_SWEEP_INTERVAL_S", "30.0"),
        ],
    )
    def test_flag_sets_env(self, command, argv, env_var, expected):
        # patch.dict reverts the write; monkeypatch.delenv on an absent var registers no cleanup.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(env_var, None)
            apply_args_to_env(parse_args(command, argv))
            assert os.environ[env_var] == expected

    def test_the_log_level_flag_sets_the_library_levels(self):
        from modelship.logging import propagate_lib_log_env

        with patch.dict(os.environ, {}, clear=False):
            for name in ("MSHIP_LOG_LEVEL", "VLLM_LOGGING_LEVEL", "TRANSFORMERS_VERBOSITY"):
                os.environ.pop(name, None)
            apply_args_to_env(parse_args("join", ["--gcs-address", "h:1", "--log-level", "debug"]))
            propagate_lib_log_env()
            assert (os.environ["VLLM_LOGGING_LEVEL"], os.environ["TRANSFORMERS_VERBOSITY"]) == ("DEBUG", "debug")

    def test_flag_overrides_preset_env(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_STORE", "redis://from-env:6379/0")
        apply_args_to_env(parse_args("start", ["--state-store", "redis://from-flag:6379/0"]))
        assert os.environ["MSHIP_STATE_STORE"] == "redis://from-flag:6379/0"

    @pytest.mark.parametrize("command", ["start", "deploy"])
    def test_absent_flags_leave_env_untouched(self, command):
        names = ["MSHIP_STATE_STORE", "MSHIP_RAY_AUTH", "MSHIP_PREFLIGHT", "MSHIP_METRICS"]
        with patch.dict(os.environ, {}, clear=False):
            for name in names:
                os.environ.pop(name, None)
            apply_args_to_env(parse_args(command, []))
            assert not any(name in os.environ for name in names)


class TestDriverCacheEnv:
    @staticmethod
    def _env_before_ray(env: dict[str, str], argv: list[str]) -> dict[str, str]:
        class _StopError(Exception):
            pass

        seen: dict[str, str] = {}

        def _capture_env():
            seen.update(os.environ)
            raise _StopError

        from modelship.driver import run

        with (
            patch.dict(os.environ, env, clear=True),
            patch("modelship.driver.resolve_cache_root", return_value="/tmp/mship-test-cache"),
            patch("modelship.driver.resolve_ray_auth_env", side_effect=_capture_env),
            pytest.raises(_StopError),
        ):
            run("start", argv)
        return seen

    def test_cache_dir_flags_reach_the_import_latched_vars(self):
        seen = self._env_before_ray({}, ["--cache-dir", "/custom/shared", "--node-cache-dir", "/custom/node"])
        assert seen["HF_HOME"] == "/custom/shared/huggingface"
        assert seen["VLLM_CACHE_ROOT"] == "/custom/node/vllm"
        assert seen["FLASHINFER_WORKSPACE_BASE"] == "/custom/node/flashinfer"

    @pytest.mark.parametrize(
        ("env", "argv", "expected"),
        [
            ({"MSHIP_HOME": "/opt/mship"}, [], "/opt/mship/node-cache"),
            ({"MSHIP_NODE_CACHE_DIR": "/from/env"}, [], "/from/env"),
            ({"MSHIP_NODE_CACHE_DIR": "/from/env"}, ["--node-cache-dir", "/from/flag"], "/from/flag"),
        ],
    )
    def test_roots_are_exported_before_ray_starts(self, env, argv, expected):
        seen = self._env_before_ray(env, argv)
        assert seen["MSHIP_CACHE_DIR"] == "/tmp/mship-test-cache"
        assert seen["MSHIP_NODE_CACHE_DIR"] == expected


_HEAD_SETTINGS = {
    "env": {**CLUSTER_ENV_DEFAULTS, "MSHIP_METRICS": "false", "MSHIP_STATE_STORE": "redis://head:6379/0"},
    "serve_logging_config": "head-logging",
    "gateway_sizing": "head-sizing",
}


class TestDriverVerbs:
    @pytest.fixture(autouse=True)
    def _isolate(self):
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("modelship.driver.signal.signal"),
            patch("modelship.remote.HEAD_WAIT_S", 0),
        ):
            for key in ("MSHIP_GATEWAY_NAME", "MSHIP_STATE_STORE", *CLUSTER_ENV_DEFAULTS, *GATEWAY_SIZING_ENV_VARS):
                os.environ.pop(key, None)
            yield

    def test_start_refuses_when_a_cluster_runs_here(self):
        from modelship import driver
        from modelship.deploy import serve_utils

        with (
            patch.object(serve_utils, "local_ray_clusters", return_value={"10.0.0.1:6380"}),
            patch.object(serve_utils, "start_head") as mock_start_head,
            pytest.raises(SystemExit, match="already running on this machine"),
        ):
            driver._start(parse_args("start", []))
        mock_start_head.assert_not_called()

    def _start(self):
        from modelship import driver
        from modelship.deploy import serve_utils

        with (
            patch.object(serve_utils, "local_ray_clusters", return_value=set()),
            patch.object(serve_utils, "start_head"),
            patch.object(serve_utils, "start_serve"),
            patch.object(serve_utils, "start_gateway") as gateway,
            patch.object(driver, "_log_cluster"),
            patch.object(driver, "_log_join_hint"),
            patch.object(driver, "_log_gpus"),
            patch.object(driver, "_prepare_state_store") as prepare_state_store,
            patch(
                "modelship.infer.deploy_coordinator.get_or_create_coordinator", return_value="deploy-coordinator"
            ) as deploy_coordinator,
            patch("modelship.infer.gateway_coordinator.get_or_create_gateway_coordinator") as gateway_coordinator,
            patch.object(driver, "_send") as send,
            patch.object(driver.signal, "pause"),
        ):
            calls = MagicMock()
            for name, mock in [
                ("gateway_coordinator", gateway_coordinator),
                ("deploy_coordinator", deploy_coordinator),
                ("gateway", gateway),
                ("send", send),
            ]:
                calls.attach_mock(mock, name)
            driver._start(parse_args("start", []))
        return SimpleNamespace(
            gateway=gateway,
            prepare_state_store=prepare_state_store,
            deploy_coordinator=deploy_coordinator,
            gateway_coordinator=gateway_coordinator,
            send=send,
            calls=calls,
        )

    def test_start_exports_its_engine_for_job_shells(self):
        self._start()
        assert os.environ["MSHIP_ENGINE_PYTHON"] == sys.executable

    @pytest.mark.parametrize(
        ("env", "warns"),
        [
            ({"MSHIP_RAY_DASHBOARD_HOST": "0.0.0.0"}, True),
            ({"MSHIP_RAY_DASHBOARD_HOST": "10.0.0.5"}, True),
            ({"MSHIP_RAY_DASHBOARD_HOST": "0.0.0.0", "MSHIP_RAY_AUTH": "true"}, False),
            ({"MSHIP_RAY_DASHBOARD_HOST": "127.0.0.1"}, False),
            ({"MSHIP_RAY_DASHBOARD_HOST": "localhost"}, False),
            ({}, False),
        ],
    )
    def test_start_warns_about_a_dashboard_beyond_loopback_without_auth(self, env, warns, caplog):
        os.environ.pop("MSHIP_RAY_DASHBOARD_HOST", None)
        os.environ.pop("MSHIP_RAY_AUTH", None)
        os.environ.update(env)
        self._start()
        assert any("without token auth" in m for m in caplog.messages) is warns

    def test_start_creates_both_coordinators_and_sends_to_the_deploy_coordinator(self):
        started = self._start()
        started.deploy_coordinator.assert_called_once_with()
        started.gateway_coordinator.assert_called_once_with()
        assert started.send.call_args.args[2] == "deploy-coordinator"

    def test_start_creates_the_gateway_after_both_coordinators_and_before_its_deploy(self):
        calls = self._start().calls.mock_calls
        assert [c[0] for c in calls] == ["gateway_coordinator", "deploy_coordinator", "gateway", "send"]

    def test_start_prepares_its_state_store_and_gives_the_gateway_its_settings(self):
        os.environ["MSHIP_STATE_STORE"] = "redis://head:6379/0"
        os.environ["MSHIP_METRICS"] = "false"
        started = self._start()
        started.prepare_state_store.assert_called_once_with()
        assert started.gateway.call_args.args[3] == {
            **CLUSTER_ENV_DEFAULTS,
            "MSHIP_METRICS": "false",
            "MSHIP_STATE_STORE": "redis://head:6379/0",
        }

    def test_start_gives_the_gateway_the_defaults_explicitly(self):
        assert self._start().gateway.call_args.args[3] == {**CLUSTER_ENV_DEFAULTS, "MSHIP_STATE_STORE": "memory://"}

    def test_start_sizes_the_gateway_from_its_env(self):
        os.environ["MSHIP_GATEWAY_MAX_REPLICAS"] = "8"
        assert self._start().gateway.call_args.args[4]["autoscaling_config"]["max_replicas"] == 8

    def test_start_refuses_invalid_gateway_sizing_before_starting_its_head(self):
        from modelship import driver
        from modelship.deploy import serve_utils

        os.environ["MSHIP_GATEWAY_MIN_REPLICAS"] = "0"
        with (
            patch.object(serve_utils, "start_head") as mock_start_head,
            pytest.raises(ValueError, match="MSHIP_GATEWAY_MIN_REPLICAS"),
        ):
            driver._start(parse_args("start", []))
        mock_start_head.assert_not_called()

    def test_deploy_refuses_without_a_local_cluster(self):
        from modelship import driver
        from modelship.deploy import serve_utils

        with (
            patch.object(serve_utils, "local_ray_clusters", return_value=set()),
            patch.object(serve_utils, "attach_cluster") as mock_attach,
            pytest.raises(SystemExit, match="mship start"),
        ):
            driver._deploy(parse_args("deploy", []))
        mock_attach.assert_not_called()

    def _deploy(self, argv, existing_apps, outcome=None, waiting=None, found=True):
        from modelship import driver
        from modelship.deploy import serve_utils

        outcome = outcome or {"id": "r1", "state": "succeeded", "reason": "", "models": {"a": "up"}, "version": 1}
        receipt = {"id": "r1", "behind": None, "outcome": "outcome-ref"}
        coordinator = MagicMock()
        coordinator.cluster_settings.remote.return_value = "settings-ref"

        def get(ref):
            return _HEAD_SETTINGS if ref == "settings-ref" else (waiting or (lambda _ref: outcome))(ref)

        with (
            patch.object(serve_utils, "local_ray_clusters", return_value={"10.0.0.1:6380"}),
            patch.object(serve_utils, "attach_cluster"),
            patch.object(serve_utils, "start_serve") as mock_serve,
            patch.object(serve_utils, "get_existing_apps", return_value=existing_apps),
            patch.object(serve_utils, "start_gateway") as mock_gateway,
            patch("modelship.infer.deploy_coordinator.find_coordinator", return_value=coordinator if found else None),
            patch.object(driver, "_log_cluster"),
            patch.object(driver, "_send", return_value=receipt) as mock_send,
            patch("ray.get", side_effect=get) as mock_get,
        ):
            args = parse_args("deploy", argv)
            apply_args_to_env(args)
            driver._deploy(args)
        return SimpleNamespace(
            gateway=mock_gateway, send=mock_send, get=mock_get, serve=mock_serve, coordinator=coordinator
        )

    def test_deploy_refuses_a_missing_default_gateway(self):
        with pytest.raises(SystemExit, match="no gateway 'modelship'"):
            self._deploy([], existing_apps=set())

    def test_deploy_without_a_deploy_coordinator_exits_4(self, capsys):
        with pytest.raises(SystemExit) as exc:
            self._deploy([], existing_apps={"modelship"}, found=False)
        assert exc.value.code == 4
        assert "no deploy coordinator on this cluster after 0 min; `mship start` creates it" in capsys.readouterr().err

    def test_waits_for_a_local_node_then_the_deploy_coordinator(self, caplog):
        from modelship import driver, remote
        from modelship.deploy import serve_utils

        caplog.set_level(logging.INFO, logger="modelship")
        coordinator = MagicMock()
        with (
            patch.object(remote, "HEAD_WAIT_S", 60),
            patch.object(driver.time, "sleep") as sleep,
            patch.object(serve_utils, "local_ray_clusters", side_effect=[set(), set(), {"10.0.0.1:6380"}]),
            patch.object(serve_utils, "attach_cluster") as attach,
            patch("modelship.infer.deploy_coordinator.find_coordinator", side_effect=[None, coordinator]),
            patch.object(driver, "_serve_logging", return_value=(logging.INFO, None)),
        ):
            assert driver._attach_when_ready() is coordinator
        attach.assert_called_once()
        assert sleep.call_count == 3
        assert caplog.messages == [
            "Waiting for a Ray node on this machine.",
            "Waiting for `mship start` to create the deploy coordinator.",
        ]

    def test_a_signal_stops_the_wait(self, caplog):
        from modelship import driver
        from modelship.deploy import serve_utils

        caplog.set_level(logging.INFO, logger="modelship")

        def interrupted():
            driver.signal.signal.call_args_list[0].args[1](signal.SIGINT, None)

        with (
            patch.object(serve_utils, "local_ray_clusters", side_effect=interrupted),
            pytest.raises(SystemExit) as exc,
        ):
            driver._attach_when_ready()
        assert exc.value.code == 130
        assert f"Stopped waiting (signal {signal.SIGINT})." in caplog.messages

    def test_deploy_sends_to_the_deploy_coordinator_it_found(self):
        deployed = self._deploy([], existing_apps={"modelship"})
        assert deployed.send.call_args.args[2] is deployed.coordinator

    def test_deploy_creates_a_named_gateway_that_is_missing(self):
        deployed = self._deploy(["--gateway-name", "edge"], existing_apps={"modelship"})
        assert deployed.gateway.call_args.args[0] == "edge"
        assert deployed.send.call_args.args[1] == "edge"

    def test_a_gateway_deploy_creates_gets_the_head_settings(self):
        os.environ["MSHIP_STATE_STORE"] = "redis://deploy-host:6379/0"
        os.environ["MSHIP_METRICS"] = "true"
        deployed = self._deploy(["--gateway-name", "edge"], existing_apps={"modelship"})
        assert deployed.gateway.call_args.args[1] == "head-logging"
        assert deployed.gateway.call_args.args[3] == _HEAD_SETTINGS["env"]
        assert deployed.gateway.call_args.args[4] == "head-sizing"

    def test_deploy_starts_serve_with_the_head_logging(self):
        self._deploy([], existing_apps={"modelship"}).serve.assert_called_once_with("head-logging")

    def test_deploy_reuses_an_existing_gateway(self):
        deployed = self._deploy([], existing_apps={"modelship"})
        deployed.gateway.assert_not_called()
        deployed.send.assert_called_once()

    def test_deploy_exits_once_sent(self, caplog):
        caplog.set_level(logging.INFO, logger="modelship")
        deployed = self._deploy([], existing_apps={"modelship"})
        deployed.send.assert_called_once()
        assert [c.args[0] for c in deployed.get.call_args_list] == ["settings-ref"]
        assert "Follow it in the head's log; cancel it with `mship deploy --cancel r1`." in caplog.messages

    def test_deploy_with_wait_waits_for_the_outcome(self, caplog):
        caplog.set_level(logging.INFO, logger="modelship")
        deployed = self._deploy(["--wait"], existing_apps={"modelship"})
        assert [c.args[0] for c in deployed.get.call_args_list] == ["settings-ref", "outcome-ref"]
        assert "Deploy r1 succeeded; the gateway is on version 1." in caplog.messages

    @pytest.mark.parametrize(
        ("state", "message"), [("failed", "Deploy r1 failed: boom"), ("cancelled", "Deploy r1 cancelled.")]
    )
    def test_deploy_exits_nonzero_when_the_request_does_not_succeed(self, state, message, caplog):
        caplog.set_level(logging.INFO, logger="modelship")
        outcome = {"id": "r1", "state": state, "reason": "boom", "models": {}, "version": None}
        with pytest.raises(SystemExit) as exc:
            self._deploy(["--wait"], existing_apps={"modelship"}, outcome=outcome)
        assert exc.value.code == 1
        assert message in caplog.messages

    def test_a_lost_deploy_coordinator_exits_nonzero(self, caplog):
        def lost(ref):
            raise RayActorError()

        with pytest.raises(SystemExit) as exc:
            self._deploy(["--wait"], existing_apps={"modelship"}, waiting=lost)
        assert exc.value.code == 1
        assert "Deploy r1 was lost: the deploy coordinator restarted. Run the deploy again." in caplog.messages

    def test_a_signal_while_waiting_only_stops_waiting(self, caplog):
        from modelship import driver

        caplog.set_level(logging.INFO, logger="modelship")

        def interrupted(ref):
            handler = driver.signal.signal.call_args.args[1]
            handler(signal.SIGINT, None)

        with patch("modelship.infer.deploy_coordinator.DeployCoordinator") as ledger, pytest.raises(SystemExit) as exc:
            self._deploy(["--wait"], existing_apps={"modelship"}, waiting=interrupted)
        assert exc.value.code == 130
        ledger.assert_not_called()
        assert any("deploy r1 keeps running" in message for message in caplog.messages)


class TestSend:
    @pytest.fixture
    def send(self, monkeypatch, tmp_path):
        from modelship import driver

        monkeypatch.delenv("MSHIP_GATEWAY_NAME", raising=False)
        monkeypatch.chdir(tmp_path)
        coordinator = MagicMock()
        coordinator.submit.remote.side_effect = lambda request: {"id": "r1", "behind": "r0", "outcome": "ref"}
        submitted = []

        def run(argv):
            args = parse_args("deploy", argv)
            with patch("ray.get", side_effect=lambda value: value):
                receipt = driver._send(args, "gw", coordinator)
            submitted.append(coordinator.submit.remote.call_args.args[0])
            return receipt

        return SimpleNamespace(run=run, submitted=submitted)

    def test_a_config_from_the_job_is_sent_even_with_reconcile(self, send):
        from modelship import remote

        config = {"metadata": {remote.MODELS_YAML_KEY: "models:\n  - name: a\n"}}
        with patch.dict(os.environ, {remote.RAY_JOB_CONFIG_ENV_VAR: json.dumps(config)}):
            send.run(["--config-from-job", "--reconcile"])
        (request,) = send.submitted
        assert (request.mode, request.models) == ("reconcile", [{"name": "a"}])

    def test_models_given_are_sent_additively(self, send):
        receipt = send.run(["--model", "org/qwen-gguf:qwen.gguf", "--loader", "llama_server", "--usecase", "generate"])
        (request,) = send.submitted
        assert (request.gateway, request.mode, request.strategy) == ("gw", "additive", "blue_green")
        assert request.models[0]["model"] == "org/qwen-gguf:qwen.gguf"
        assert receipt["id"] == "r1"

    def test_reconcile_and_the_replace_strategy_are_carried(self, send, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text("models: []\n")
        send.run(["--config", str(config), "--reconcile", "--replace-strategy", "stop_start"])
        (request,) = send.submitted
        assert (request.mode, request.strategy, request.models) == ("reconcile", "stop_start", [])

    def test_no_models_sends_a_bare_request(self, send):
        send.run(["--reconcile"])
        assert send.submitted[0].mode == "bare"
        assert send.submitted[0].models is None

    def test_the_per_deploy_settings_ride_the_request(self, send, monkeypatch):
        monkeypatch.setenv("MSHIP_PREFLIGHT", "false")
        send.run(["--reconcile"])
        assert send.submitted[0].env["MSHIP_PREFLIGHT"] == "false"

    def test_the_state_store_is_neither_opened_nor_sent(self, send, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_STORE", "redis://deploy-host:6379/0")
        with patch("modelship.state.get_state_store") as opened:
            send.run(["--reconcile"])
        opened.assert_not_called()
        assert "MSHIP_STATE_STORE" not in send.submitted[0].env

    def test_logging_and_metrics_are_not_sent(self, send, monkeypatch):
        for name in CLUSTER_ENV_DEFAULTS | NODE_ENV_DEFAULTS:
            monkeypatch.setenv(name, "from-deploy")
        send.run(["--reconcile"])
        assert set(CLUSTER_ENV_DEFAULTS | NODE_ENV_DEFAULTS).isdisjoint(send.submitted[0].env)


class TestPrepareStateStore:
    @pytest.mark.parametrize(("uri", "warns"), [("memory://", True), ("redis://head:6379/0", False)])
    def test_seeds_the_compaction_key_and_warns_only_for_memory(self, monkeypatch, caplog, uri, warns):
        from modelship import driver

        caplog.set_level(logging.WARNING, logger="modelship")
        monkeypatch.setenv("MSHIP_STATE_STORE", uri)
        with patch("modelship.openai.compaction_crypto.ensure_key_seeded") as seeded:
            driver._prepare_state_store()
        seeded.assert_called_once()
        assert any("non-durable" in message for message in caplog.messages) is warns


class TestRemoteDeploy:
    def _run(self, argv):
        from modelship import driver

        with (
            patch.dict(os.environ),
            patch("modelship.remote.run") as run_remote,
            patch.object(driver, "_deploy") as deploy,
        ):
            os.environ.pop("MSHIP_RAY_DASHBOARD_URL", None)
            driver.run("deploy", argv)
        deploy.assert_not_called()
        return run_remote.call_args.args

    def test_goes_to_the_dashboard_with_the_config(self):
        argv = ["--ray-dashboard-url", "http://head:8265", "--config", "m.yaml", "--wait"]
        assert self._run(argv) == ("http://head:8265", argv, "m.yaml")

    @pytest.mark.parametrize("extra", [[], ["--cancel", "r1"], ["--model", "org/repo"]])
    def test_ships_no_config_without_one(self, extra):
        assert self._run(["--ray-dashboard-url", "http://head:8265", *extra])[2] is None


class TestCancelCommand:
    def _cancel(self, result, *, clusters=frozenset({"10.0.0.1:6380"}), actor=True, argv=(), waiting=None):
        from modelship import driver
        from modelship.deploy import serve_utils

        coordinator = MagicMock()
        coordinator.cancel.remote.return_value = result
        rolled_back = {"id": "r1", "state": "cancelled", "reason": "", "models": {"a": "rolled back"}}
        with (
            patch.object(serve_utils, "local_ray_clusters", return_value=set(clusters)),
            patch.object(serve_utils, "attach_cluster"),
            patch("modelship.infer.deploy_coordinator.find_coordinator", return_value=coordinator if actor else None),
            patch("ray.get", side_effect=waiting or (lambda value: rolled_back if value == "outcome-ref" else value)),
            patch.object(driver.signal, "signal"),
            patch("modelship.remote.HEAD_WAIT_S", 0),
        ):
            driver._cancel(parse_args("deploy", ["--cancel", "r1", *argv]))
        return coordinator

    def test_deploy_with_cancel_runs_the_cancel(self):
        from modelship import driver

        with (
            patch.dict(os.environ),
            patch.object(driver, "configure_logging"),
            patch.object(driver, "_cancel") as cancel,
            patch.object(driver, "_deploy") as deploy,
        ):
            driver.run("deploy", ["--cancel", "r1"])
        assert cancel.call_args.args[0].cancel == "r1"
        deploy.assert_not_called()

    def test_cancels_the_deploy(self, caplog):
        caplog.set_level(logging.INFO, logger="modelship")
        coordinator = self._cancel({"cancelled": True, "message": "deploy r1 cancelled"})
        coordinator.cancel.remote.assert_called_once_with("r1")
        assert "Deploy r1 cancelled." in caplog.messages

    def test_wait_waits_for_the_rollback(self, caplog):
        caplog.set_level(logging.INFO, logger="modelship")
        result = {
            "cancelled": True,
            "message": "deploy r1 is being cancelled and rolled back",
            "outcome": "outcome-ref",
        }
        self._cancel(result, argv=["--wait"])
        assert caplog.messages[-2:] == ["  a: rolled back", "Deploy r1 cancelled."]

    def test_wait_exits_nonzero_when_the_deploy_ends_otherwise(self, caplog):
        result = {
            "cancelled": True,
            "message": "deploy r1 is being cancelled and rolled back",
            "outcome": "outcome-ref",
        }
        failed = {"id": "r1", "state": "failed", "reason": "boom", "models": {}}

        with pytest.raises(SystemExit) as exc:
            self._cancel(result, argv=["--wait"], waiting=lambda value: failed if value == "outcome-ref" else value)
        assert exc.value.code == 1
        assert "Deploy r1 failed: boom" in caplog.messages

    def test_wait_exits_nonzero_when_the_deploy_coordinator_is_lost(self, caplog):
        result = {
            "cancelled": True,
            "message": "deploy r1 is being cancelled and rolled back",
            "outcome": "outcome-ref",
        }

        def lost(value):
            if value == "outcome-ref":
                raise RayActorError()
            return value

        with pytest.raises(SystemExit) as exc:
            self._cancel(result, argv=["--wait"], waiting=lost)
        assert exc.value.code == 1
        assert (
            "Deploy r1 was lost: the deploy coordinator restarted, which rolls back what wasn't committed."
            in caplog.messages
        )

    def test_a_signal_while_waiting_only_stops_waiting(self, caplog):
        from modelship import driver

        caplog.set_level(logging.INFO, logger="modelship")
        result = {
            "cancelled": True,
            "message": "deploy r1 is being cancelled and rolled back",
            "outcome": "outcome-ref",
        }

        def interrupted(value):
            if value != "outcome-ref":
                return value
            handler = driver.signal.signal.call_args.args[1]
            handler(signal.SIGINT, None)

        with pytest.raises(SystemExit) as exc:
            self._cancel(result, argv=["--wait"], waiting=interrupted)
        assert exc.value.code == 130
        assert f"Stopped waiting (signal {signal.SIGINT}); deploy r1 keeps rolling back." in caplog.messages

    def test_exits_nonzero_when_nothing_was_cancelled(self):
        with pytest.raises(SystemExit, match="no queued or running deploy r1"):
            self._cancel({"cancelled": False, "message": "no queued or running deploy r1"})

    def test_refuses_without_a_local_cluster(self):
        with pytest.raises(SystemExit, match="no Ray cluster"):
            self._cancel({}, clusters=frozenset())

    def test_a_cluster_without_a_deploy_coordinator_exits_4(self):
        with pytest.raises(SystemExit) as exc:
            self._cancel({}, actor=False)
        assert exc.value.code == 4


class TestStopHead:
    def _stop(self, env, pop=()):
        from modelship import driver
        from modelship.deploy import removal, serve_utils

        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(removal, "delete_model_apps") as mock_delete,
            patch.object(serve_utils, "shutdown_ray") as mock_shutdown,
        ):
            for key in pop:
                os.environ.pop(key, None)
            driver._stop_head()
        return mock_delete, mock_shutdown

    def test_deletes_every_model_app_then_stops_serve_and_ray(self):
        mock_delete, mock_shutdown = self._stop({}, pop=("RAY_REDIS_ADDRESS",))
        mock_delete.assert_called_once_with(30.0)
        mock_shutdown.assert_called_once_with()

    def test_redis_backed_gcs_keeps_the_apps(self):
        mock_delete, mock_shutdown = self._stop({"RAY_REDIS_ADDRESS": "redis:6379"})
        mock_delete.assert_not_called()
        mock_shutdown.assert_called_once_with(keep_serve=True)


class TestShutdownRay:
    def test_stops_serve_then_ray(self):
        from modelship.deploy import serve_utils

        calls = []
        with (
            patch.object(serve_utils.serve, "shutdown", side_effect=lambda: calls.append("serve")),
            patch.object(serve_utils.ray, "shutdown", side_effect=lambda: calls.append("ray")),
        ):
            serve_utils.shutdown_ray()
        assert calls == ["serve", "ray"]

    def test_keep_serve_stops_only_ray(self):
        from modelship.deploy import serve_utils

        with (
            patch.object(serve_utils.serve, "shutdown") as mock_serve,
            patch.object(serve_utils.ray, "shutdown") as mock_ray,
        ):
            serve_utils.shutdown_ray(keep_serve=True)
        mock_serve.assert_not_called()
        mock_ray.assert_called_once()


class TestSignalHandlersOutliveRay:
    """ray.init and Node() each install a SIGTERM handler of their own; ours must win."""

    @pytest.fixture(autouse=True)
    def _restore(self):
        saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        with patch.dict(os.environ, {"MSHIP_GCS_ADDRESS": "head:6380"}, clear=False):
            os.environ.pop("MSHIP_GATEWAY_NAME", None)
            os.environ.pop("MSHIP_STATE_STORE", None)
            yield
        for sig, handler in saved.items():
            signal.signal(sig, handler)

    @staticmethod
    def _ray_takes_sigterm(*_args):
        signal.signal(signal.SIGTERM, lambda *_: None)

    def test_start(self):
        from modelship import driver
        from modelship.deploy import serve_utils

        with (
            patch.object(serve_utils, "local_ray_clusters", return_value=set()),
            patch.object(serve_utils, "start_head", side_effect=self._ray_takes_sigterm),
            patch.object(serve_utils, "start_serve"),
            patch.object(serve_utils, "start_gateway"),
            patch.object(driver, "_log_cluster"),
            patch.object(driver, "_log_join_hint"),
            patch.object(driver, "_log_gpus"),
            patch.object(driver, "_prepare_state_store"),
            patch("modelship.infer.deploy_coordinator.get_or_create_coordinator"),
            patch("modelship.infer.gateway_coordinator.get_or_create_gateway_coordinator"),
            patch.object(driver, "_send"),
            patch.object(driver.signal, "pause"),
        ):
            driver._start(parse_args("start", []))
        assert signal.getsignal(signal.SIGTERM).__name__ == "_cleanup"

    def test_join(self):
        from modelship import driver
        from modelship.deploy import serve_utils

        with (
            patch.object(serve_utils, "join_cluster", side_effect=self._ray_takes_sigterm),
            patch.object(serve_utils, "supervise_join_node"),
            patch.object(driver, "_log_gpus"),
        ):
            driver._join()
        assert signal.getsignal(signal.SIGTERM).__name__ == "_leave"


class TestRepeatSignals:
    @pytest.fixture(autouse=True)
    def _restore(self):
        saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        yield
        for sig, handler in saved.items():
            signal.signal(sig, handler)

    def test_a_signal_during_the_handler_is_dropped(self):
        from modelship import driver

        calls = []

        def _stop(sig, _frame):
            calls.append(sig)
            if len(calls) == 1:
                os.kill(os.getpid(), signal.SIGINT)

        driver._on_signals(_stop)
        os.kill(os.getpid(), signal.SIGTERM)
        assert calls == [signal.SIGTERM]


class TestRandSuffix:
    def test_default_length(self):
        suffix = rand_suffix()
        assert len(suffix) == 5

    def test_custom_length(self):
        suffix = rand_suffix(10)
        assert len(suffix) == 10

    def test_chars_are_alphanumeric_lowercase(self):
        for _ in range(50):
            suffix = rand_suffix()
            assert all(c.islower() or c.isdigit() for c in suffix)


class TestBuildDeploymentOptions:
    def test_basic_options(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
            num_cpus=2,
        )
        opts = build_deployment_options(config)
        actor = opts["ray_actor_options"]
        assert actor["num_gpus"] == 1
        assert actor["num_cpus"] == 2
        assert "env_vars" in actor["runtime_env"]
        assert "pip" not in actor["runtime_env"]
        assert "placement_group_bundles" not in opts

    def test_llama_server_honors_num_gpus(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
            num_gpus=2,
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 2
        assert "placement_group_bundles" not in opts

    def test_llama_server_num_gpus_zero_stays_cpu(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
            num_gpus=0,
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0

    def test_llama_server_honors_fractional_num_gpus(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
            num_gpus=0.5,
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0.5

    def test_sherpa_onnx_num_gpus_forced_to_zero(self):
        # sherpa_onnx never touches CUDA or CoreML (CPU only); a nonzero num_gpus
        # here would just reserve GPU capacity the loader never uses.
        config = ModelshipModelConfig(
            name="test-model",
            model="kokoro-en-v0_19",
            usecase=ModelUsecase.tts,
            loader=ModelLoader.sherpa_onnx,
            num_gpus=1,
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0

    def test_sherpa_onnx_num_gpus_forced_to_zero_on_darwin_too(self):
        # Unlike the ggml loaders' Metal carve-out, sherpa_onnx has no CUDA path
        # on any platform, so the force-zero applies unconditionally.
        config = ModelshipModelConfig(
            name="test-model",
            model="kokoro-en-v0_19",
            usecase=ModelUsecase.tts,
            loader=ModelLoader.sherpa_onnx,
            num_gpus=1,
        )
        with patch("modelship.deploy.actor_options.platform.system", return_value="Darwin"):
            opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0

    def test_stable_diffusion_cpp_force_cpu_off_darwin(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.image,
            loader=ModelLoader.stable_diffusion_cpp,
            num_gpus=1,
        )
        with patch("modelship.deploy.actor_options.platform.system", return_value="Linux"):
            opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0

    def test_stable_diffusion_cpp_honors_num_gpus_on_darwin(self):
        # ggml picks up Metal via its own runtime backend registry regardless of Ray —
        # forcing 0 here would let Ray co-schedule another GPU actor onto the same GPU.
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.image,
            loader=ModelLoader.stable_diffusion_cpp,
            num_gpus=1,
        )
        with patch("modelship.deploy.actor_options.platform.system", return_value="Darwin"):
            opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 1

    def test_passthrough_env_vars_forwarded_to_replicas(self, monkeypatch):
        monkeypatch.setenv("MSHIP_METRICS", "false")
        monkeypatch.setenv("MSHIP_GATEWAY_NAME", "edge")
        monkeypatch.setenv("MSHIP_PREFLIGHT", "false")
        monkeypatch.setenv("MSHIP_RESPONSES_TTL_S", "60")
        monkeypatch.setenv("MSHIP_STATE_SWEEP_INTERVAL_S", "30")
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
        )
        env_vars = build_deployment_options(config)["ray_actor_options"]["runtime_env"]["env_vars"]
        assert env_vars["MSHIP_METRICS"] == "false"
        assert env_vars["MSHIP_GATEWAY_NAME"] == "edge"
        assert env_vars["MSHIP_PREFLIGHT"] == "false"
        # Gateway / memory-store settings: nothing in a model replica reads them.
        assert "MSHIP_RESPONSES_TTL_S" not in env_vars
        assert "MSHIP_STATE_SWEEP_INTERVAL_S" not in env_vars

    def test_the_state_store_is_this_processes_whatever_the_request_carries(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_STORE", "redis://head:6379/0")
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
        )
        env = {"MSHIP_PREFLIGHT": "false", "MSHIP_STATE_STORE": "redis://deploy-host:6379/0"}
        env_vars = build_deployment_options(config, env)["ray_actor_options"]["runtime_env"]["env_vars"]
        assert env_vars["MSHIP_STATE_STORE"] == "redis://head:6379/0"
        assert env_vars["MSHIP_PREFLIGHT"] == "false"

    def test_the_default_state_store_is_forwarded_explicitly(self, monkeypatch):
        monkeypatch.delenv("MSHIP_STATE_STORE", raising=False)
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
        )
        env_vars = build_deployment_options(config, {})["ray_actor_options"]["runtime_env"]["env_vars"]
        assert env_vars["MSHIP_STATE_STORE"] == "memory://"

    def test_unset_settings_leave_out_per_deploy_ones_and_default_cluster_ones(self, monkeypatch):
        monkeypatch.delenv("MSHIP_METRICS", raising=False)
        monkeypatch.delenv("MSHIP_PREFLIGHT", raising=False)
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
        )
        env_vars = build_deployment_options(config)["ray_actor_options"]["runtime_env"]["env_vars"]
        assert env_vars["MSHIP_METRICS"] == "true"
        assert "MSHIP_PREFLIGHT" not in env_vars

    def test_cluster_settings_are_this_processes_and_node_settings_are_not_forwarded(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
        )
        env = {"MSHIP_LOG_FORMAT": "text", "MSHIP_METRICS": "true"}
        head = {"MSHIP_LOG_FORMAT": "json", "MSHIP_METRICS": "false", "MSHIP_LOG_LEVEL": "TRACE"}
        with patch.dict(os.environ, head, clear=True):
            env_vars = build_deployment_options(config, env)["ray_actor_options"]["runtime_env"]["env_vars"]
        assert (env_vars["MSHIP_LOG_FORMAT"], env_vars["MSHIP_METRICS"]) == ("json", "false")
        assert set(NODE_ENV_DEFAULTS).isdisjoint(env_vars)
        for key in build_cache_env_vars():
            assert key in env_vars

    def test_pipeline_parallel_uses_placement_group(self):
        # num_gpus=2 + pp=2 satisfies world_size==num_gpus; the outer actor sits in
        # bundle 0 with no GPU, and vLLM workers claim the rest via the inherited placement group.
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=2,
            vllm_engine_kwargs=VllmEngineConfig(pipeline_parallel_size=2),
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0
        assert opts["placement_group_strategy"] == "STRICT_PACK"
        bundles = opts["placement_group_bundles"]
        assert len(bundles) == 2
        assert all(b["GPU"] == 1.0 for b in bundles)

    def test_tp_times_pp_builds_pg(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=4,
            vllm_engine_kwargs=VllmEngineConfig(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
            ),
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0
        assert len(opts["placement_group_bundles"]) == 4
        assert all(b["GPU"] == 1.0 for b in opts["placement_group_bundles"])

    def test_single_slot_skips_placement_group(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=0.3,
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0.3
        assert "placement_group_bundles" not in opts

    def test_max_ongoing_requests_omitted_by_default(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
        )
        opts = build_deployment_options(config)
        assert "max_ongoing_requests" not in opts

    def test_max_ongoing_requests_forwarded_when_set(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
            max_ongoing_requests=256,
        )
        opts = build_deployment_options(config)
        assert opts["max_ongoing_requests"] == 256

    def test_max_ongoing_requests_forwarded_for_multi_slot(self):
        # Multi-slot (PG) deploys carry the cap alongside placement_group_bundles.
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=2,
            vllm_engine_kwargs=VllmEngineConfig(tensor_parallel_size=2),
            max_ongoing_requests=64,
        )
        opts = build_deployment_options(config)
        assert opts["max_ongoing_requests"] == 64
        assert len(opts["placement_group_bundles"]) == 2


class TestBuildDeploymentOptionsCapabilityResources:
    """The `mship_<loader>` capability resource must gate scheduling regardless of
    num_gpus, on single-slot, multi-slot (PG), and stable_diffusion_cpp deploys."""

    def test_single_slot_requests_capability(self):
        config = ModelshipModelConfig(
            name="m", model="x", usecase=ModelUsecase.generate, loader=ModelLoader.vllm, num_gpus=0
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["resources"] == {"mship_vllm": 0.001}

    def test_multi_slot_requests_capability_on_every_bundle_not_actor(self):
        config = ModelshipModelConfig(
            name="m",
            model="x",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=2,
            vllm_engine_kwargs=VllmEngineConfig(tensor_parallel_size=2),
        )
        opts = build_deployment_options(config)
        assert "resources" not in opts["ray_actor_options"]
        assert all(b["mship_vllm"] == 0.001 for b in opts["placement_group_bundles"])

    def test_stable_diffusion_cpp_requests_capability(self):
        config = ModelshipModelConfig(
            name="m", model="x", usecase=ModelUsecase.image, loader=ModelLoader.stable_diffusion_cpp, num_gpus=0
        )
        with patch("modelship.deploy.actor_options.platform.system", return_value="Linux"):
            opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["resources"] == {"mship_stable_diffusion_cpp": 0.001}

    def test_llama_server_requests_capability_regardless_of_num_gpus(self):
        config = ModelshipModelConfig(
            name="m", model="x", usecase=ModelUsecase.generate, loader=ModelLoader.llama_server, num_gpus=0
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["resources"] == {"mship_llama_server": 0.001}


class TestReservationTotals:
    def test_single_slot_uses_actor_options(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=0.5,
            num_cpus=2,
        )
        opts = build_deployment_options(config)
        assert total_gpu_reservation(opts) == 0.5

    def test_multi_slot_sums_pg_bundles(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=4,
            num_cpus=2,
        )
        opts = build_deployment_options(config)
        assert total_gpu_reservation(opts) == 4


class TestDeleteModelApps:
    @pytest.fixture
    def serve(self, monkeypatch):
        from modelship.deploy import removal

        apps = {"gw": None, "gw.a-1234567890": None, "gw.b-1234567890": None}
        deleted: list[str] = []
        monkeypatch.setattr(removal.serve, "status", lambda: SimpleNamespace(applications=dict(apps)))
        monkeypatch.setattr(removal.serve, "delete", lambda name, _blocking=True: deleted.append(name))
        monkeypatch.setattr(removal.time, "sleep", lambda seconds: [apps.pop(name, None) for name in deleted])
        return SimpleNamespace(apps=apps, deleted=deleted)

    def test_deletes_every_model_app_but_not_the_gateway(self, serve):
        from modelship.deploy import removal

        removal.delete_model_apps(30.0)
        assert serve.deleted == ["gw.a-1234567890", "gw.b-1234567890"]
        assert set(serve.apps) == {"gw"}

    def test_gives_up_waiting_after_the_timeout(self, serve, monkeypatch, caplog):
        from modelship.deploy import removal

        monkeypatch.setattr(removal.time, "sleep", lambda seconds: None)
        removal.delete_model_apps(0.0)
        assert "2 deployment(s) still being removed after 0 s" in caplog.text


_SIZING = {
    "autoscaling_config": {"min_replicas": 1, "max_replicas": 4, "target_ongoing_requests": 64.0},
    "max_ongoing_requests": 1024,
}


class TestStartGateway:
    def _run(self, env=None, cluster_env=None, name="gw"):
        from modelship.deploy import serve_utils

        options = MagicMock()
        with (
            patch.dict(os.environ, env or {}, clear=env is not None),
            patch.object(serve_utils.ModelshipAPI, "options", options),
            patch.object(serve_utils.serve, "run") as mock_run,
        ):
            serve_utils.start_gateway(name, MagicMock(), f"/{name}", cluster_env or {}, _SIZING)
        return options.call_args.kwargs, mock_run.call_args.kwargs

    def test_route_prefix_forwarded_to_serve_run(self):
        _, run_kwargs = self._run()
        assert run_kwargs["route_prefix"] == "/gw"

    def test_the_gateway_autoscales_with_the_given_sizing(self):
        options, _ = self._run()
        assert "num_replicas" not in options
        assert options["autoscaling_config"] == _SIZING["autoscaling_config"]
        assert options["max_ongoing_requests"] == 1024

    def test_a_replica_scaled_down_has_time_to_finish_its_streams(self):
        options, _ = self._run()
        assert options["graceful_shutdown_timeout_s"] == 600

    def test_gateway_name_pinned_from_arg(self):
        options, _ = self._run(env={}, name="edge")
        assert options["ray_actor_options"]["runtime_env"]["env_vars"]["MSHIP_GATEWAY_NAME"] == "edge"

    def test_the_cluster_settings_are_the_ones_given_not_this_processes(self):
        options, _ = self._run(
            {"MSHIP_STATE_STORE": "redis://deploy-host:6379/0", "MSHIP_LOG_FORMAT": "text"},
            cluster_env={"MSHIP_STATE_STORE": "redis://head:6379/0", "MSHIP_LOG_FORMAT": "json"},
        )
        env_vars = options["ray_actor_options"]["runtime_env"]["env_vars"]
        assert (env_vars["MSHIP_STATE_STORE"], env_vars["MSHIP_LOG_FORMAT"]) == ("redis://head:6379/0", "json")


class TestGatewaySizing:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            assert gateway_sizing() == _SIZING

    def test_env_overrides(self):
        env = {
            "MSHIP_GATEWAY_MIN_REPLICAS": "2",
            "MSHIP_GATEWAY_MAX_REPLICAS": "8",
            "MSHIP_GATEWAY_TARGET_ONGOING_REQUESTS": "32.5",
            "MSHIP_GATEWAY_MAX_ONGOING_REQUESTS": "256",
        }
        with patch.dict(os.environ, env, clear=True):
            assert gateway_sizing() == {
                "autoscaling_config": {"min_replicas": 2, "max_replicas": 8, "target_ongoing_requests": 32.5},
                "max_ongoing_requests": 256,
            }

    def test_min_and_max_can_be_equal(self):
        with patch.dict(os.environ, {"MSHIP_GATEWAY_MIN_REPLICAS": "3", "MSHIP_GATEWAY_MAX_REPLICAS": "3"}, clear=True):
            assert gateway_sizing()["autoscaling_config"]["min_replicas"] == 3

    @pytest.mark.parametrize(
        "name, value",
        [
            ("MSHIP_GATEWAY_MIN_REPLICAS", "0"),
            ("MSHIP_GATEWAY_MIN_REPLICAS", "1.5"),
            ("MSHIP_GATEWAY_MAX_REPLICAS", "-2"),
            ("MSHIP_GATEWAY_TARGET_ONGOING_REQUESTS", "0"),
            ("MSHIP_GATEWAY_TARGET_ONGOING_REQUESTS", "many"),
            ("MSHIP_GATEWAY_MAX_ONGOING_REQUESTS", "notanint"),
        ],
    )
    def test_rejects_invalid_env(self, name, value):
        with patch.dict(os.environ, {name: value}, clear=True), pytest.raises(ValueError, match=name):
            gateway_sizing()

    def test_rejects_a_max_below_the_min(self):
        env = {"MSHIP_GATEWAY_MIN_REPLICAS": "3", "MSHIP_GATEWAY_MAX_REPLICAS": "2"}
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(
                ValueError, match=r"MSHIP_GATEWAY_MAX_REPLICAS \(2\) must be >= MSHIP_GATEWAY_MIN_REPLICAS \(3\)"
            ),
        ):
            gateway_sizing()


class TestGatewayRoutePrefix:
    def test_slugifies_name(self):
        from modelship.deploy import serve_utils

        assert serve_utils.gateway_route_prefix("modelship api") == "/modelship-api"
        assert serve_utils.gateway_route_prefix("llm-api") == "/llm-api"
        assert serve_utils.gateway_route_prefix("Edge_2") == "/edge_2"

    def test_no_url_safe_chars_raises(self):
        from modelship.deploy import serve_utils

        with pytest.raises(ValueError):
            serve_utils.gateway_route_prefix("!!!")


class TestGatewayFromEnv:
    @pytest.mark.parametrize("name", ["edge.eu", "edge#eu"])
    def test_a_name_with_a_dot_or_hash_is_rejected(self, monkeypatch, name):
        from modelship import driver

        monkeypatch.setenv("MSHIP_GATEWAY_NAME", name)
        with pytest.raises(SystemExit, match=r"must not contain '\.' or '#'"):
            driver._gateway_from_env()


class TestValidateNodeGpuReservation:
    """--node-num-gpus must not exceed what this container can actually see — an
    inflated value would surface much later, at a replica's model load, instead of at startup."""

    def _fake_gpus(self, count):
        from modelship.preflight import GPUInfo

        return [GPUInfo(index=i, available_bytes=0, name="test", uuid=None) for i in range(count)]

    def test_reservation_within_visible_count_passes(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_NODE_NUM_GPUS": "1"}, clear=False),
            patch.object(serve_utils, "detect_gpus", return_value=self._fake_gpus(2)),
        ):
            serve_utils._validate_node_gpu_reservation()  # no raise

    def test_reservation_equal_to_visible_count_passes(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_NODE_NUM_GPUS": "2"}, clear=False),
            patch.object(serve_utils, "detect_gpus", return_value=self._fake_gpus(2)),
        ):
            serve_utils._validate_node_gpu_reservation()  # no raise

    def test_reservation_exceeding_visible_count_raises(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_NODE_NUM_GPUS": "2"}, clear=False),
            patch.object(serve_utils, "detect_gpus", return_value=self._fake_gpus(1)),
            pytest.raises(RuntimeError, match="exceeds the 1 GPU"),
        ):
            serve_utils._validate_node_gpu_reservation()

    def test_reservation_unset_skips_check(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(serve_utils, "detect_gpus") as mock_detect,
        ):
            os.environ.pop("MSHIP_NODE_NUM_GPUS", None)
            serve_utils._validate_node_gpu_reservation()
        mock_detect.assert_not_called()

    def test_start_head_raises_on_gpu_mismatch(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_NODE_NUM_GPUS": "2"}, clear=False),
            patch.object(serve_utils, "detect_gpus", return_value=self._fake_gpus(1)),
            patch.object(serve_utils.ray, "init") as mock_init,
            pytest.raises(RuntimeError, match="exceeds the 1 GPU"),
        ):
            serve_utils.start_head(20)
        mock_init.assert_not_called()

    def test_attach_skips_check(self):
        # Attaching starts no node, so there is no reservation to check.
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_NODE_NUM_GPUS": "99"}, clear=False),
            patch.object(serve_utils, "detect_gpus") as mock_detect,
            patch.object(serve_utils.ray, "init"),
        ):
            serve_utils.attach_cluster(20)
        mock_detect.assert_not_called()


class TestStartHead:
    def _init_call(self, env, pop=()):
        """Returns the kwargs start_head passed to ray.init(). `pop` clears
        env vars before the call."""
        from modelship.deploy import serve_utils

        with patch.dict(os.environ, env, clear=False):
            for key in pop:
                os.environ.pop(key, None)
            with (
                patch.object(serve_utils.ray, "init") as mock_init,
                # Don't sweep the real /tmp/ray during tests.
                patch.object(serve_utils, "prune_ray_sessions"),
            ):
                serve_utils.start_head(20)
        _, kwargs = mock_init.call_args
        return kwargs

    def test_starts_a_new_local_instance(self):
        assert self._init_call({"RAY_ADDRESS": "10.0.0.9:6380"})["address"] == "local"

    def test_starts_head_with_metrics_port(self):
        kwargs = self._init_call({"MSHIP_METRICS": "true", "MSHIP_NODE_NUM_CPUS": "4"}, pop=("MSHIP_METRICS_PORT",))
        assert kwargs["num_cpus"] == 4
        assert kwargs["_metrics_export_port"] == 8079

    def test_metrics_port_overridable(self):
        kwargs = self._init_call({"MSHIP_METRICS": "true", "MSHIP_METRICS_PORT": "9090"})
        assert kwargs["_metrics_export_port"] == 9090

    def test_cuda_multi_gpu_left_unset_for_autodetect(self):
        """A cuda/rocm/xpu node must not be pinned to 1 GPU when MSHIP_NODE_NUM_GPUS
        is unset — Ray autodetects the real device count."""
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_accelerator", return_value="cuda"):
            kwargs = self._init_call({}, pop=("MSHIP_NODE_NUM_GPUS",))
        assert "num_gpus" not in kwargs

    def test_cpu_accelerator_forces_zero_gpus(self):
        """A torch CPU build must advertise 0 GPUs even if nvidia-smi/NVML sees hardware."""
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_accelerator", return_value="cpu"):
            kwargs = self._init_call({}, pop=("MSHIP_NODE_NUM_GPUS",))
        assert kwargs["num_gpus"] == 0

    def test_metal_detected_advertises_one_gpu(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_accelerator", return_value="metal"):
            kwargs = self._init_call({}, pop=("MSHIP_NODE_NUM_GPUS",))
        assert kwargs["num_gpus"] == 1

    def test_explicit_zero_gpus_wins_over_metal_detection(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_accelerator", return_value="metal"):
            kwargs = self._init_call({"MSHIP_NODE_NUM_GPUS": "0"})
        assert kwargs["num_gpus"] == 0

    def test_node_memory_splits_into_memory_and_object_store(self):
        kwargs = self._init_call({"MSHIP_NODE_MEMORY": str(10 * 1024**3)})
        # 30% object store (Ray's own resolve_object_store_memory), 70% schedulable 'memory'.
        assert kwargs["object_store_memory"] == int(10 * 1024**3 * 0.3)
        assert kwargs["_memory"] == 10 * 1024**3 - kwargs["object_store_memory"]

    def test_node_memory_accepts_unit_suffix(self):
        kwargs = self._init_call({"MSHIP_NODE_MEMORY": "10Gi"})
        assert kwargs["object_store_memory"] == int(10 * 1024**3 * 0.3)
        assert kwargs["_memory"] == 10 * 1024**3 - kwargs["object_store_memory"]

    def test_node_memory_absent_when_unset(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_available_ram_bytes", return_value=0):
            kwargs = self._init_call({}, pop=("MSHIP_NODE_MEMORY",))
        assert "_memory" not in kwargs
        assert "object_store_memory" not in kwargs

    def test_node_memory_auto_detected_when_unset(self):
        from modelship.deploy import serve_utils

        available = 10 * 1024**3
        with patch.object(serve_utils, "detect_available_ram_bytes", return_value=available):
            kwargs = self._init_call({}, pop=("MSHIP_NODE_MEMORY",))
        total_bytes = int(available * serve_utils._AUTO_NODE_MEMORY_HEADROOM)
        assert kwargs["object_store_memory"] == int(total_bytes * 0.3)
        assert kwargs["_memory"] == total_bytes - kwargs["object_store_memory"]

    def test_explicit_node_memory_wins_over_auto_detect(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_available_ram_bytes", return_value=999 * 1024**3):
            kwargs = self._init_call({"MSHIP_NODE_MEMORY": str(10 * 1024**3)})
        assert kwargs["object_store_memory"] == int(10 * 1024**3 * 0.3)
        assert kwargs["_memory"] == 10 * 1024**3 - kwargs["object_store_memory"]

    def test_resources_forwarded_from_capability_probe(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "node_capability_resources", return_value={"mship_vllm": 1}):
            kwargs = self._init_call({})
        assert kwargs["resources"] == {"mship_vllm": 1}

    def test_dashboard_always_on_bound_localhost(self):
        kwargs = self._init_call({}, pop=("MSHIP_RAY_DASHBOARD_HOST",))
        assert kwargs["include_dashboard"] is True
        assert kwargs["dashboard_host"] == "127.0.0.1"

    def test_dashboard_host_overridable(self):
        kwargs = self._init_call({"MSHIP_RAY_DASHBOARD_HOST": "0.0.0.0"})
        assert kwargs["include_dashboard"] is True
        assert kwargs["dashboard_host"] == "0.0.0.0"

    def test_dashboard_port_absent_when_unset(self):
        kwargs = self._init_call({}, pop=("MSHIP_RAY_DASHBOARD_PORT",))
        assert "dashboard_port" not in kwargs

    def test_dashboard_port_overridable(self):
        # Lets multiple modelship heads share one host under --network=host, where
        # Ray's own dashboard port (8265) would otherwise collide between them.
        kwargs = self._init_call({"MSHIP_RAY_DASHBOARD_PORT": "8266"})
        assert kwargs["dashboard_port"] == 8266

    @pytest.mark.parametrize("value", ["false", ""])
    def test_omits_metrics_port_when_disabled(self, value):
        kwargs = self._init_call({"MSHIP_METRICS": value})
        assert "_metrics_export_port" not in kwargs

    def test_redis_credentials_passed_with_a_redis_backed_gcs(self):
        kwargs = self._init_call({"RAY_REDIS_ADDRESS": "redis:6379", "REDIS_PASSWORD": "pw", "REDIS_USERNAME": "u"})
        assert kwargs["_redis_password"] == "pw"
        assert kwargs["_redis_username"] == "u"

    def test_redis_credentials_ignored_without_a_redis_backed_gcs(self):
        kwargs = self._init_call({"REDIS_PASSWORD": "pw", "REDIS_USERNAME": "u"}, pop=("RAY_REDIS_ADDRESS",))
        assert "_redis_password" not in kwargs
        assert "_redis_username" not in kwargs

    def test_ray_init_still_takes_the_private_kwargs(self):
        import inspect

        import ray

        source = inspect.getsource(ray.init)
        for kwarg in ("_memory", "_metrics_export_port", "_redis_password", "_redis_username"):
            assert f'"{kwarg}"' in source

    def test_gcs_port_sets_gcs_server_port(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_GCS_PORT": "6390"}, clear=False),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions"),
        ):
            os.environ.pop("RAY_GCS_SERVER_PORT", None)
            serve_utils.start_head(20)
            assert os.environ.get("RAY_GCS_SERVER_PORT") == "6390"

    def test_gcs_port_absent_defaults_gcs_server_port_to_6380(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions"),
        ):
            os.environ.pop("MSHIP_GCS_PORT", None)
            os.environ.pop("RAY_GCS_SERVER_PORT", None)
            serve_utils.start_head(20)
            # Not Ray's own 6379 default — that collides with the recommended
            # same-host Redis state store under --network=host.
            assert os.environ.get("RAY_GCS_SERVER_PORT") == "6380"

    def test_gcs_port_respects_explicit_gcs_server_port(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(
                os.environ,
                {
                    "MSHIP_GCS_PORT": "6380",
                    "RAY_GCS_SERVER_PORT": "6381",
                },
                clear=False,
            ),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions"),
        ):
            serve_utils.start_head(20)
            # setdefault: an operator's explicit RAY_GCS_SERVER_PORT always wins.
            assert os.environ["RAY_GCS_SERVER_PORT"] == "6381"

    def test_prunes_stale_sessions(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions") as mock_prune,
        ):
            serve_utils.start_head(20)
        mock_prune.assert_called_once()


class TestAttachCluster:
    @pytest.mark.parametrize(
        "error",
        [ConnectionError("wrong token"), AuthenticationError("no authentication token was found")],
        ids=["wrong_token", "missing_token"],
    )
    def test_a_refused_connection_names_the_token(self, error):
        from modelship.deploy import serve_utils

        with (
            patch.object(serve_utils.ray, "init", side_effect=error),
            pytest.raises(SystemExit, match="set MSHIP_RAY_AUTH_TOKEN to its token"),
        ):
            serve_utils.attach_cluster(logging.INFO)

    def test_connects_via_auto_without_starting_a_node(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_GCS_PORT": "6380", "MSHIP_RAY_DASHBOARD_PORT": "8266"}, clear=False),
            patch.object(serve_utils.ray, "init") as mock_init,
            patch.object(serve_utils, "prune_ray_sessions") as mock_prune,
        ):
            os.environ.pop("RAY_GCS_SERVER_PORT", None)
            serve_utils.attach_cluster(20)
            assert "RAY_GCS_SERVER_PORT" not in os.environ
        _, kwargs = mock_init.call_args
        assert kwargs["address"] == "auto"
        for key in ("_metrics_export_port", "num_cpus", "resources", "include_dashboard", "dashboard_port"):
            assert key not in kwargs
        mock_prune.assert_not_called()


class TestLocalRayClusters:
    def test_reads_a_fresh_raylet_scan_each_call(self):
        from modelship.deploy import serve_utils

        with patch("ray._private.services._find_address_from_flag", side_effect=[{"10.0.0.1:6380"}, set()]):
            assert serve_utils.local_ray_clusters() == {"10.0.0.1:6380"}
            assert serve_utils.local_ray_clusters() == set()

    def test_private_ray_scan_still_exists(self):
        # Canary: fails on a Ray bump that moves or reshapes this private helper.
        from ray._private.services import find_gcs_addresses

        find_gcs_addresses.cache_clear()
        assert isinstance(find_gcs_addresses(), frozenset)


@pytest.fixture
def _reset_join_node():
    """_join_ray_cluster sets the module-level _join_node global as soon as Node()
    succeeds; tests that go through it must reset it or isolation depends on test order."""
    from modelship.deploy import serve_utils

    serve_utils._join_node = None
    yield
    serve_utils._join_node = None


class TestJoinRayCluster:
    """_join_ray_cluster starts this container's node in-process via
    ray._private.node.Node(head=False) instead of shelling out; these tests mock
    that Ray-internal surface. TestClusterJoin exercises it for real."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_join_node):
        yield

    def _join(self, env, pop=(), bootstrap="10.0.0.1:6380"):
        from modelship.deploy import serve_utils

        mock_node = MagicMock()
        mock_node.get_temp_dir_path.return_value = "/tmp/ray"
        with patch.dict(os.environ, env, clear=False):
            for key in pop:
                os.environ.pop(key, None)
            with (
                patch("ray._private.services.canonicalize_bootstrap_address", return_value=bootstrap) as mock_canon,
                patch("ray._private.services.get_node_ip_address", return_value="10.0.0.2"),
                patch("ray._private.parameter.RayParams") as mock_params,
                patch("ray._private.node.Node", return_value=mock_node) as mock_node_cls,
                patch(
                    "ray._private.authentication.authentication_token_setup.ensure_token_if_auth_enabled"
                ) as mock_ensure,
                patch("ray._private.utils.write_ray_address") as mock_write,
            ):
                result = serve_utils._join_ray_cluster("head:6380")
        return {
            "node": mock_node,
            "node_cls": mock_node_cls,
            "params_kwargs": mock_params.call_args.kwargs,
            "canon": mock_canon,
            "ensure": mock_ensure,
            "write": mock_write,
            "result": result,
        }

    def test_builds_rayparams_with_cpus_and_gpus(self):
        kw = self._join({"MSHIP_NODE_NUM_CPUS": "4", "MSHIP_NODE_NUM_GPUS": "2"})["params_kwargs"]
        assert kw["num_cpus"] == 4
        assert kw["num_gpus"] == 2

    def test_omits_num_cpus_when_unset(self):
        kw = self._join({}, pop=("MSHIP_NODE_NUM_CPUS", "MSHIP_NODE_NUM_GPUS"))["params_kwargs"]
        assert kw["num_cpus"] is None

    def test_cuda_multi_gpu_left_unset_for_autodetect(self):
        """Regression test: a cuda/rocm/xpu joiner must NOT be pinned to 1 GPU
        when MSHIP_NODE_NUM_GPUS is unset — Ray autodetects the real count."""
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_accelerator", return_value="cuda"):
            kw = self._join({}, pop=("MSHIP_NODE_NUM_CPUS", "MSHIP_NODE_NUM_GPUS"))["params_kwargs"]
        assert kw["num_gpus"] is None

    def test_cpu_accelerator_forces_zero_gpus(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_accelerator", return_value="cpu"):
            kw = self._join({}, pop=("MSHIP_NODE_NUM_CPUS", "MSHIP_NODE_NUM_GPUS"))["params_kwargs"]
        assert kw["num_gpus"] == 0

    def test_metal_detected_advertises_one_gpu_when_unset(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_accelerator", return_value="metal"):
            kw = self._join({}, pop=("MSHIP_NODE_NUM_CPUS", "MSHIP_NODE_NUM_GPUS"))["params_kwargs"]
        assert kw["num_gpus"] == 1

    def test_explicit_zero_gpus_honored(self):
        # Thin-image case: MSHIP_NODE_NUM_GPUS=0 is a real reservation, not "unset".
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_accelerator", return_value="metal"):
            kw = self._join({"MSHIP_NODE_NUM_GPUS": "0"})["params_kwargs"]
        assert kw["num_gpus"] == 0

    def test_resources_forwarded_from_capability_probe(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "node_capability_resources", return_value={"mship_vllm": 1}):
            kw = self._join({})["params_kwargs"]
        assert kw["resources"] == {"mship_vllm": 1}

    def test_node_memory_splits_into_memory_and_object_store(self):
        kw = self._join({"MSHIP_NODE_MEMORY": str(10 * 1024**3)})["params_kwargs"]
        # 30% object store (Ray's own resolve_object_store_memory), 70% schedulable 'memory'.
        assert kw["object_store_memory"] == int(10 * 1024**3 * 0.3)
        assert kw["memory"] == 10 * 1024**3 - kw["object_store_memory"]

    def test_node_memory_absent_when_unset(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_available_ram_bytes", return_value=0):
            kw = self._join({}, pop=("MSHIP_NODE_MEMORY",))["params_kwargs"]
        assert kw["memory"] is None
        assert kw["object_store_memory"] is None

    def test_node_memory_auto_detected_when_unset(self):
        from modelship.deploy import serve_utils

        available = 10 * 1024**3
        with patch.object(serve_utils, "detect_available_ram_bytes", return_value=available):
            kw = self._join({}, pop=("MSHIP_NODE_MEMORY",))["params_kwargs"]
        total_bytes = int(available * serve_utils._AUTO_NODE_MEMORY_HEADROOM)
        assert kw["object_store_memory"] == int(total_bytes * 0.3)
        assert kw["memory"] == total_bytes - kw["object_store_memory"]

    def test_explicit_node_memory_wins_over_auto_detect(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_available_ram_bytes", return_value=999 * 1024**3):
            kw = self._join({"MSHIP_NODE_MEMORY": str(10 * 1024**3)})["params_kwargs"]
        assert kw["object_store_memory"] == int(10 * 1024**3 * 0.3)
        assert kw["memory"] == 10 * 1024**3 - kw["object_store_memory"]

    def test_metrics_port_random_when_unset(self):
        kw = self._join({}, pop=("MSHIP_METRICS_PORT",))["params_kwargs"]
        assert kw["metrics_export_port"] is None

    def test_metrics_port_pinned_when_set(self):
        kw = self._join({"MSHIP_METRICS_PORT": "8079"})["params_kwargs"]
        assert kw["metrics_export_port"] == 8079

    def test_passes_bootstrap_gcs_address(self):
        kw = self._join({}, bootstrap="10.9.9.9:6380")["params_kwargs"]
        assert kw["gcs_address"] == "10.9.9.9:6380"

    def test_creates_worker_node_supervised(self):
        out = self._join({})
        _, kwargs = out["node_cls"].call_args
        assert kwargs["head"] is False
        assert kwargs["shutdown_at_exit"] is True
        assert kwargs["spawn_reaper"] is True
        out["node"].check_version_info.assert_called_once()

    def test_writes_discovery_marker_with_bootstrap_address(self):
        out = self._join({}, bootstrap="10.9.9.9:6380")
        out["write"].assert_called_once_with("10.9.9.9:6380", "/tmp/ray")

    def test_sets_module_global_and_returns_node(self):
        from modelship.deploy import serve_utils

        out = self._join({})
        assert serve_utils._join_node is out["node"]
        assert out["result"] is out["node"]

    def test_calls_ensure_token_preflight(self):
        self._join({})["ensure"].assert_called_once()

    def test_unresolvable_address_raises(self):
        from modelship.deploy import serve_utils

        with (
            patch("ray._private.services.canonicalize_bootstrap_address", return_value=None),
            pytest.raises(RuntimeError, match="Could not resolve the Ray head address"),
        ):
            serve_utils._join_ray_cluster("bogus:1")


class TestJoinCluster:
    """join_cluster brings up this machine's node via _join_ray_cluster (mocked here —
    TestJoinRayCluster covers its internals) and connects no driver."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_join_node):
        yield

    def test_starts_the_node_without_a_driver(self):
        from modelship.deploy import serve_utils

        with (
            patch.object(serve_utils, "_join_ray_cluster") as mock_join,
            patch.object(serve_utils, "prune_ray_sessions") as mock_prune,
            patch.object(serve_utils.ray, "init") as mock_init,
        ):
            serve_utils.join_cluster("head:6380")
        mock_join.assert_called_once_with("head:6380")
        mock_prune.assert_called_once()
        mock_init.assert_not_called()

    @pytest.mark.parametrize(
        ("address", "expected"),
        [
            ("head", "head:6380"),
            ("head:6390", "head:6390"),
            ("10.0.0.5", "10.0.0.5:6380"),
            ("[fd00::5]", "[fd00::5]:6380"),
            ("[fd00::5]:6390", "[fd00::5]:6390"),
        ],
    )
    def test_the_gcs_port_defaults_to_6380(self, address, expected):
        from modelship.deploy import serve_utils

        with (
            patch.object(serve_utils, "_join_ray_cluster") as mock_join,
            patch.object(serve_utils, "prune_ray_sessions"),
        ):
            serve_utils.join_cluster(address)
        mock_join.assert_called_once_with(expected)


class TestLeaveRayCluster:
    @pytest.fixture(autouse=True)
    def _reset(self, _reset_join_node):
        yield

    def test_leave_tears_down_only_the_join_node(self):
        from modelship.deploy import serve_utils

        mock_node = MagicMock()
        serve_utils._join_node = mock_node
        with patch.object(serve_utils.ray, "shutdown") as mock_shutdown:
            serve_utils.leave_ray_cluster()
        mock_shutdown.assert_called_once()
        # allow_graceful lets the raylet drain hosted actors; check_alive=False
        # because a partially-started node may not have every process up.
        mock_node.kill_all_processes.assert_called_once_with(check_alive=False, allow_graceful=True)

    def test_leave_noop_when_not_joined(self):
        from modelship.deploy import serve_utils

        assert serve_utils._join_node is None
        with patch.object(serve_utils.ray, "shutdown") as mock_shutdown:
            serve_utils.leave_ray_cluster()  # must not raise with no node to stop
        mock_shutdown.assert_called_once()


class TestSuperviseJoinNode:
    @pytest.fixture(autouse=True)
    def _reset(self, _reset_join_node):
        yield

    def test_exits_nonzero_and_kills_when_core_process_dies(self):
        from modelship.deploy import serve_utils

        node = MagicMock()
        dead = MagicMock()
        dead.returncode = 1  # not a graceful SIGTERM/0 exit
        node.dead_processes.return_value = [("raylet", dead)]
        serve_utils._join_node = node
        with (
            patch.object(serve_utils.time, "sleep"),
            pytest.raises(SystemExit) as exc,
        ):
            serve_utils.supervise_join_node()
        assert exc.value.code == 1
        node.kill_all_processes.assert_called_once_with(check_alive=False, allow_graceful=False)

    def test_ignores_graceful_exits_and_keeps_supervising(self):
        from modelship.deploy import serve_utils

        node = MagicMock()
        graceful = MagicMock()
        graceful.returncode = 0  # in _GRACEFUL_EXIT_CODES — expected, not a failure
        node.dead_processes.return_value = [("agent", graceful)]
        serve_utils._join_node = node
        # First sleep returns, second breaks the otherwise-infinite loop so the
        # test can assert the graceful exit was NOT treated as a failure.
        with (
            patch.object(serve_utils.time, "sleep", side_effect=[None, RuntimeError("stop")]),
            pytest.raises(RuntimeError, match="stop"),
        ):
            serve_utils.supervise_join_node()
        node.kill_all_processes.assert_not_called()


class TestStartAuthEnv:
    @pytest.mark.parametrize(
        ("env", "stray"),
        [
            ({"MSHIP_RAY_AUTH_TOKEN": "t"}, ["MSHIP_RAY_AUTH_TOKEN"]),
            ({"RAY_AUTH_TOKEN": "t"}, ["RAY_AUTH_TOKEN"]),
            ({"RAY_AUTH_MODE": "token"}, ["RAY_AUTH_MODE=token"]),
            ({"RAY_AUTH_MODE": "disabled"}, []),
            ({"MSHIP_RAY_AUTH_TOKEN": "t", "MSHIP_RAY_AUTH": "true"}, []),
            ({}, []),
        ],
    )
    def test_token_env_without_auth(self, env, stray):
        from modelship.utils import ray_auth

        with patch.dict(os.environ, env):
            for key in ("MSHIP_RAY_AUTH", "MSHIP_RAY_AUTH_TOKEN", "RAY_AUTH_TOKEN", "RAY_AUTH_MODE"):
                if key not in env:
                    os.environ.pop(key, None)
            assert ray_auth.token_env_without_auth() == stray

    def test_start_refuses_a_token_without_auth(self):
        from modelship import driver

        with patch.dict(os.environ, {"MSHIP_RAY_AUTH_TOKEN": "t"}), patch.object(driver, "_start") as start:
            os.environ.pop("MSHIP_RAY_AUTH", None)
            with pytest.raises(SystemExit, match="MSHIP_RAY_AUTH_TOKEN is set but Ray auth is off"):
                driver.run("start", [])
        start.assert_not_called()

    @pytest.mark.parametrize(
        ("host", "loopback"),
        [("127.0.0.1", True), ("127.0.0.2", True), ("::1", True), ("localhost", True), ("0.0.0.0", False)],
    )
    def test_is_loopback(self, host, loopback):
        from modelship.utils import ray_auth

        assert ray_auth.is_loopback(host) is loopback


class TestJoinHint:
    @pytest.mark.parametrize(
        ("env", "hint"),
        [
            ({}, "(see docs"),
            ({"RAY_AUTH_MODE": "token", "MSHIP_RAY_AUTH_TOKEN": "t"}, "(the one this head was started with)"),
            ({"RAY_AUTH_MODE": "token", "RAY_AUTH_TOKEN": "t"}, "(RAY_AUTH_TOKEN in this head's environment)"),
            ({"RAY_AUTH_MODE": "token"}, "(~/.ray/auth_token on this machine)"),
        ],
    )
    def test_names_where_the_token_is(self, env, hint, caplog):
        from modelship import driver

        caplog.set_level(logging.INFO, logger="modelship")
        with patch.dict(os.environ, env), patch("ray.get_runtime_context") as context:
            for key in ("RAY_AUTH_MODE", "MSHIP_RAY_AUTH_TOKEN", "RAY_AUTH_TOKEN", "RAY_GCS_SERVER_PORT"):
                if key not in env:
                    os.environ.pop(key, None)
            context.return_value.gcs_address = "10.0.0.1:6380"
            driver._log_join_hint()
        (message,) = [m for m in caplog.messages if m.startswith("To add a machine")]
        assert "mship join --gcs-address=10.0.0.1:6380" in message
        assert hint in message
        assert ("MSHIP_RAY_AUTH_TOKEN set" in message) is bool(env)


class TestResolveRayAuthEnv:
    """resolve_ray_auth_env front-runs Ray's import-time RAY_AUTH_MODE latch,
    translating MSHIP_RAY_AUTH/MSHIP_RAY_AUTH_TOKEN into RAY_AUTH_MODE/RAY_AUTH_TOKEN
    before the driver imports ray."""

    def _resolve(self, env):
        from modelship.utils import ray_auth

        with patch.dict(os.environ, env, clear=False):
            for key in ["MSHIP_RAY_AUTH", "MSHIP_RAY_AUTH_TOKEN", "RAY_AUTH_MODE", "RAY_AUTH_TOKEN"]:
                if key not in env:
                    os.environ.pop(key, None)
            ray_auth.resolve_ray_auth_env()
            return os.environ.get("RAY_AUTH_MODE"), os.environ.get("RAY_AUTH_TOKEN")

    def test_enabled_auth_sets_mode(self):
        mode, token = self._resolve({"MSHIP_RAY_AUTH": "true"})
        assert mode == "token"
        assert token is None

    def test_a_token_sets_mode_and_token(self):
        mode, token = self._resolve({"MSHIP_RAY_AUTH_TOKEN": "secret"})
        assert mode == "token"
        assert token == "secret"

    def test_neither_disables_auth(self):
        assert self._resolve({}) == ("disabled", None)

    def test_disabled_auth_disables_auth(self):
        assert self._resolve({"MSHIP_RAY_AUTH": "false"}) == ("disabled", None)

    def test_explicit_ray_auth_mode_wins(self):
        mode, _ = self._resolve({"MSHIP_RAY_AUTH": "true", "RAY_AUTH_MODE": "disabled"})
        # setdefault: an operator's explicit RAY_AUTH_MODE always wins.
        assert mode == "disabled"


class TestPruneRaySessions:
    """`prune_ray_sessions` resolves the temp root via Ray's own `get_default_ray_temp_dir()`
    (`<RAY_TMPDIR>/ray`), so pointing RAY_TMPDIR at a tmp dir isolates these tests."""

    def _temp_root(self, tmp_path):
        root = tmp_path / "ray"
        root.mkdir()
        return root

    def _make_session(self, root, pid, name=None):
        session = root / (name or f"session_2026-06-19_10-00-00_000000_{pid}")
        (session / "logs").mkdir(parents=True)
        (session / "logs" / "raylet.out").write_text("log")
        return session

    def test_removes_dead_pid_session(self, tmp_path):
        from modelship.deploy import serve_utils

        root = self._temp_root(tmp_path)
        dead = self._make_session(root, 111)
        with (
            patch.dict(
                os.environ,
                {"RAY_TMPDIR": str(tmp_path), "MSHIP_PRUNE_RAY_SESSIONS": "true"},
                clear=False,
            ),
            patch.object(serve_utils, "_pid_alive", return_value=False),
        ):
            serve_utils.prune_ray_sessions()
        assert not dead.exists()

    def test_keeps_live_pid_session(self, tmp_path):
        from modelship.deploy import serve_utils

        root = self._temp_root(tmp_path)
        live = self._make_session(root, 222)
        with (
            patch.dict(
                os.environ,
                {"RAY_TMPDIR": str(tmp_path), "MSHIP_PRUNE_RAY_SESSIONS": "true"},
                clear=False,
            ),
            patch.object(serve_utils, "_pid_alive", return_value=True),
        ):
            serve_utils.prune_ray_sessions()
        assert live.exists()

    def test_skips_symlink_and_non_session_entries(self, tmp_path):
        from modelship.deploy import serve_utils

        root = self._temp_root(tmp_path)
        dead = self._make_session(root, 333)
        latest = root / "session_latest"
        latest.symlink_to(dead)
        marker = root / "ray_current_cluster"
        marker.write_text("127.0.0.1:6379")
        unrelated = root / "not_a_session"
        unrelated.mkdir()
        with (
            patch.dict(
                os.environ,
                {"RAY_TMPDIR": str(tmp_path), "MSHIP_PRUNE_RAY_SESSIONS": "true"},
                clear=False,
            ),
            patch.object(serve_utils, "_pid_alive", return_value=False),
        ):
            serve_utils.prune_ray_sessions()
        assert not dead.exists()
        assert latest.is_symlink()  # survives, now dangling
        assert marker.exists()
        assert unrelated.exists()

    def test_disabled_via_env_keeps_everything(self, tmp_path):
        from modelship.deploy import serve_utils

        root = self._temp_root(tmp_path)
        dead = self._make_session(root, 444)
        with (
            patch.dict(
                os.environ,
                {"RAY_TMPDIR": str(tmp_path), "MSHIP_PRUNE_RAY_SESSIONS": "false"},
                clear=False,
            ),
            patch.object(serve_utils, "_pid_alive", return_value=False),
        ):
            serve_utils.prune_ray_sessions()
        assert dead.exists()

    def test_missing_temp_root_is_noop(self, tmp_path):
        from modelship.deploy import serve_utils

        # No <tmp>/ray dir exists — must not raise.
        with patch.dict(
            os.environ,
            {"RAY_TMPDIR": str(tmp_path), "MSHIP_PRUNE_RAY_SESSIONS": "true"},
            clear=False,
        ):
            serve_utils.prune_ray_sessions()

    def test_pid_alive_true_for_current_process(self):
        from modelship.deploy import serve_utils

        assert serve_utils._pid_alive(os.getpid()) is True

    def test_pid_alive_false_for_reaped_pid(self):
        import subprocess

        from modelship.deploy import serve_utils

        proc = subprocess.Popen(["true"])
        proc.wait()
        assert serve_utils._pid_alive(proc.pid) is False
