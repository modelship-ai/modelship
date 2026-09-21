"""Tests for the start/join/deploy CLI parsing and driver helpers."""

import os
import signal
from unittest.mock import MagicMock, patch

import pytest

from modelship.deploy.actor_options import (
    build_cache_env_vars,
    build_deployment_options,
    total_cpu_reservation,
    total_gpu_reservation,
)
from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase, VllmEngineConfig
from modelship.utils import parse_memory_bytes, rand_suffix
from modelship.utils.cli import apply_args_to_env, parse_args
from modelship.utils.runtime_env import MODEL_ENV_VARS, build_env_vars


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
        assert args.replace_strategy == "blue_green"

    def test_reconcile_with_stop_start_strategy(self):
        args = parse_args("deploy", ["--reconcile", "--replace-strategy", "stop_start"])
        assert args.reconcile is True
        assert args.replace_strategy == "stop_start"

    @pytest.mark.parametrize(
        ("command", "argv", "attr", "expected"),
        [
            ("start", ["--config", "/some/path/models.yaml"], "config", "/some/path/models.yaml"),
            ("start", ["--gateway-replicas", "3"], "gateway_replicas", 3),
            ("deploy", ["--gateway-name", "my-gateway"], "gateway_name", "my-gateway"),
            ("start", ["--ray-auth", "token"], "ray_auth", "token"),
            ("deploy", ["--ray-auth", "token"], "ray_auth", "token"),
            ("start", ["--ray-port", "6380"], "ray_port", 6380),
            ("start", ["--dashboard-port", "8266"], "dashboard_port", 8266),
            ("join", ["--cluster", "mship-head:6380"], "cluster", "mship-head:6380"),
            ("join", ["--cluster", "h:1", "--token", "secret"], "token", "secret"),
            ("deploy", ["--token", "secret"], "token", "secret"),
            ("start", ["--node-num-cpus", "4"], "node_num_cpus", 4),
            ("join", ["--cluster", "h:1", "--node-num-gpus", "2"], "node_num_gpus", 2),
            ("join", ["--cluster", "h:1", "--node-memory", "8Gi"], "node_memory", 8 * 1024**3),
            ("join", ["--cluster", "h:1", "--api-keys", "k1"], "api_keys", "k1"),
            ("deploy", ["--responses-ttl-s", "60"], "responses_ttl_s", 60.0),
            ("start", ["--state-sweep-interval-s", "30"], "state_sweep_interval_s", 30.0),
        ],
    )
    def test_flag_parses(self, command, argv, attr, expected):
        assert getattr(parse_args(command, argv), attr) == expected

    @pytest.mark.parametrize(
        ("command", "argv"),
        [
            ("deploy", ["--ray-port", "6380"]),
            ("deploy", ["--node-num-cpus", "4"]),
            ("deploy", ["--prune-ray-sessions", "false"]),
            ("deploy", ["--api-keys", "k1"]),
            ("deploy", ["--cluster", "h:1"]),
            ("start", ["--cluster", "h:1"]),
            ("start", ["--token", "secret"]),
            ("start", ["--replace-strategy", "stop_start"]),
            ("join", ["--cluster", "h:1", "--config", "models.yaml"]),
            ("join", ["--cluster", "h:1", "--model", "org/repo"]),
            ("join", ["--cluster", "h:1", "--ray-port", "6380"]),
            ("join", ["--cluster", "h:1", "--state-store", "redis://h:6379/0"]),
        ],
    )
    def test_flag_owned_by_another_command_is_rejected(self, command, argv):
        with pytest.raises(SystemExit):
            parse_args(command, argv)

    @pytest.mark.parametrize("command", ["start", "join", "deploy"])
    @pytest.mark.parametrize("flag", ["--use-existing-ray-cluster", "--address=h:1"])
    def test_removed_flags_are_rejected(self, command, flag):
        with pytest.raises(SystemExit):
            parse_args(command, [flag])

    def test_join_requires_a_cluster(self, monkeypatch):
        monkeypatch.delenv("MSHIP_CLUSTER", raising=False)
        with pytest.raises(SystemExit):
            parse_args("join", [])

    def test_join_takes_the_cluster_from_env(self, monkeypatch):
        monkeypatch.setenv("MSHIP_CLUSTER", "mship-head:6380")
        assert parse_args("join", []).cluster is None


class TestApplyArgsToEnv:
    @pytest.mark.parametrize(
        ("command", "argv", "env_var", "expected"),
        [
            ("deploy", ["--state-store", "redis://cache:6379/0"], "MSHIP_STATE_STORE", "redis://cache:6379/0"),
            ("start", ["--gateway-replicas", "4"], "MSHIP_GATEWAY_REPLICAS", "4"),
            ("deploy", ["--ray-auth", "token"], "MSHIP_RAY_AUTH", "token"),
            ("start", ["--ray-port", "6380"], "MSHIP_RAY_PORT", "6380"),
            ("start", ["--dashboard-port", "8266"], "MSHIP_RAY_DASHBOARD_PORT", "8266"),
            ("join", ["--cluster", "mship-head:6380"], "MSHIP_CLUSTER", "mship-head:6380"),
            ("deploy", ["--token", "secret"], "MSHIP_RAY_AUTH_TOKEN", "secret"),
            ("join", ["--cluster", "h:1", "--node-num-cpus", "4"], "MSHIP_NODE_NUM_CPUS", "4"),
            ("start", ["--node-num-gpus", "2"], "MSHIP_NODE_NUM_GPUS", "2"),
            ("start", ["--node-memory", "8Gi"], "MSHIP_NODE_MEMORY", str(8 * 1024**3)),
            ("join", ["--cluster", "h:1", "--prune-ray-sessions", "false"], "MSHIP_PRUNE_RAY_SESSIONS", "false"),
            ("start", ["--no-preflight"], "MSHIP_PREFLIGHT", "false"),
            ("deploy", ["--no-metrics"], "MSHIP_METRICS", "false"),
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

    def test_flag_overrides_preset_env(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_STORE", "redis://from-env:6379/0")
        apply_args_to_env(parse_args("deploy", ["--state-store", "redis://from-flag:6379/0"]))
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


class TestDriverVerbs:
    @pytest.fixture(autouse=True)
    def _isolate(self):
        with patch.dict(os.environ, {}, clear=False), patch("modelship.driver.signal.signal"):
            for key in ("MSHIP_GATEWAY_NAME", "MSHIP_STATE_STORE"):
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

    def _deploy(self, argv, existing_apps, fatally_failed=()):
        from modelship import driver
        from modelship.deploy import serve_utils

        with (
            patch.object(serve_utils, "local_ray_clusters", return_value={"10.0.0.1:6380"}),
            patch.object(serve_utils, "attach_cluster"),
            patch.object(serve_utils, "start_serve"),
            patch.object(serve_utils, "get_existing_apps", return_value=existing_apps),
            patch.object(serve_utils, "start_gateway") as mock_gateway,
            patch.object(driver, "_log_cluster"),
            patch.object(driver, "_apply", return_value=list(fatally_failed)) as mock_apply,
        ):
            args = parse_args("deploy", argv)
            apply_args_to_env(args)
            driver._deploy(args)
        return mock_gateway, mock_apply

    def test_deploy_refuses_a_missing_default_gateway(self):
        with pytest.raises(SystemExit, match="no gateway 'modelship'"):
            self._deploy([], existing_apps=set())

    def test_deploy_creates_a_named_gateway_that_is_missing(self):
        mock_gateway, mock_apply = self._deploy(["--gateway-name", "edge"], existing_apps={"modelship"})
        assert mock_gateway.call_args.args[0] == "edge"
        mock_apply.assert_called_once()

    def test_deploy_reuses_an_existing_gateway(self):
        mock_gateway, mock_apply = self._deploy([], existing_apps={"modelship"})
        mock_gateway.assert_not_called()
        mock_apply.assert_called_once()

    def test_deploy_exits_nonzero_on_fatal_failures(self):
        with pytest.raises(SystemExit) as exc:
            self._deploy([], existing_apps={"modelship"}, fatally_failed=[(MagicMock(), "boom")])
        assert exc.value.code == 1


class TestSignalHandlersOutliveRay:
    """ray.init and Node() each install a SIGTERM handler of their own; ours must win."""

    @pytest.fixture(autouse=True)
    def _restore(self):
        saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        with patch.dict(os.environ, {"MSHIP_CLUSTER": "head:6380"}, clear=False):
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
            patch.object(driver, "_apply", return_value=[]),
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
        # --no-metrics / logging / gateway set on the driver must reach the replica
        # via runtime_env, else the replica defaults to metrics-on (inconsistent).
        monkeypatch.setenv("MSHIP_METRICS", "false")
        monkeypatch.setenv("MSHIP_GATEWAY_NAME", "edge")
        monkeypatch.setenv("MSHIP_PREFLIGHT", "false")
        monkeypatch.setenv("MSHIP_RESPONSES_TTL_S", "60")
        monkeypatch.setenv("MSHIP_STATE_SWEEP_INTERVAL_S", "30")
        monkeypatch.setenv("MSHIP_STATE_STORE", "redis://host:6379/0")
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
        # Unread by the replica, but relayed if it recreates the coordinator.
        assert env_vars["MSHIP_STATE_STORE"] == "redis://host:6379/0"

    def test_unset_passthrough_env_vars_not_forwarded(self, monkeypatch):
        # Unset on the driver → not forwarded, so the replica keeps its own default.
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
        assert "MSHIP_METRICS" not in env_vars
        assert "MSHIP_PREFLIGHT" not in env_vars

    def test_log_level_in_forwarded_and_deployment_env(self):
        # MSHIP_LOG_LEVEL must reach a model deployment's runtime_env, alongside the
        # cache vars (which the gateway path omits).
        with patch.dict(os.environ, {"MSHIP_LOG_LEVEL": "TRACE"}, clear=True):
            assert build_env_vars(MODEL_ENV_VARS)["MSHIP_LOG_LEVEL"] == "TRACE"

            config = ModelshipModelConfig(
                name="test-model",
                model="some-model",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
                num_gpus=1,
            )
            env_vars = build_deployment_options(config)["ray_actor_options"]["runtime_env"]["env_vars"]
            assert env_vars["MSHIP_LOG_LEVEL"] == "TRACE"
            # Cache vars still present (the model path keeps them).
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
        assert total_cpu_reservation(opts) == 2

    def test_multi_slot_sums_pg_bundles(self):
        # 4 slots, each bundle reserves num_cpus from the cluster; the outer
        # actor's CPU sits inside bundle 0 and is not additive.
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
        assert total_cpu_reservation(opts) == 8


class TestRemoveApps:
    # remove_apps lives in deploy.removal, not serve_utils.
    def test_noop_on_empty_list(self):
        from modelship.deploy import removal

        replica_coordinator = MagicMock()
        with patch("modelship.deploy.removal.serve.delete") as mock_delete:
            removal.remove_apps([], replica_coordinator, "gw")
        replica_coordinator.unregister_deployment.remote.assert_not_called()
        mock_delete.assert_not_called()

    def test_unregisters_then_deletes(self):
        from modelship.deploy import removal

        replica_coordinator = MagicMock()
        apps = ["qwen-aaaaaaaaaa", "kokoro-bbbbbbbbbb"]
        with (
            patch("modelship.deploy.removal.ray.get") as mock_get,
            patch("modelship.deploy.removal.serve.delete") as mock_delete,
        ):
            removal.remove_apps(apps, replica_coordinator, "gw")

        # Each app is dropped from the replica coordinator's registry (bumping the
        # gateway generation so replicas stop routing) before serve.delete tears it down.
        replica_coordinator.unregister_deployment.remote.assert_any_call("gw", "qwen-aaaaaaaaaa")
        replica_coordinator.unregister_deployment.remote.assert_any_call("gw", "kokoro-bbbbbbbbbb")
        mock_get.assert_called_once()  # batched ray.get over the unregister calls
        assert mock_delete.call_args_list == [(("qwen-aaaaaaaaaa",),), (("kokoro-bbbbbbbbbb",),)]

    def test_continues_on_serve_delete_error(self):
        from modelship.deploy import removal

        replica_coordinator = MagicMock()
        with (
            patch("modelship.deploy.removal.ray.get"),
            patch("modelship.deploy.removal.serve.delete", side_effect=[Exception("gone"), None]) as mock_delete,
        ):
            removal.remove_apps(["a-1234567890", "b-1234567890"], replica_coordinator, "gw")
        # Both deletes attempted even though the first raised.
        assert mock_delete.call_count == 2


class TestStartGateway:
    def _run(self, env):
        from modelship.deploy import serve_utils

        bound = MagicMock()
        options = MagicMock()
        options.return_value.bind.return_value = bound
        logging_config = MagicMock()
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(serve_utils.ModelshipAPI, "options", options),
            patch.object(serve_utils.serve, "run") as mock_run,
        ):
            serve_utils.start_gateway("gw", logging_config, "/gw")
        return options, mock_run

    def test_route_prefix_forwarded_to_serve_run(self):
        _, mock_run = self._run({"MSHIP_GATEWAY_REPLICAS": "1", "MSHIP_GATEWAY_MAX_ONGOING": "1024"})
        _, kwargs = mock_run.call_args
        assert kwargs["route_prefix"] == "/gw"

    def test_defaults(self):
        # Ensure no leftover env from the ambient process leaks the assertion.
        options, mock_run = self._run({"MSHIP_GATEWAY_REPLICAS": "1", "MSHIP_GATEWAY_MAX_ONGOING": "1024"})
        _, kwargs = options.call_args
        assert kwargs["num_replicas"] == 1
        assert kwargs["max_ongoing_requests"] == 1024
        mock_run.assert_called_once()

    def test_env_overrides(self):
        options, _ = self._run({"MSHIP_GATEWAY_REPLICAS": "3", "MSHIP_GATEWAY_MAX_ONGOING": "256"})
        _, kwargs = options.call_args
        assert kwargs["num_replicas"] == 3
        assert kwargs["max_ongoing_requests"] == 256

    def test_forwards_log_level_to_gateway_replica(self):
        # The gateway replica must inherit MSHIP_LOG_LEVEL (and the gateway name)
        # via runtime_env, else it can't configure logging at the driver's level.
        options, _ = self._run(
            {
                "MSHIP_GATEWAY_REPLICAS": "1",
                "MSHIP_GATEWAY_MAX_ONGOING": "1024",
                "MSHIP_LOG_LEVEL": "TRACE",
            }
        )
        _, kwargs = options.call_args
        env_vars = kwargs["ray_actor_options"]["runtime_env"]["env_vars"]
        assert env_vars["MSHIP_LOG_LEVEL"] == "TRACE"

    def test_gateway_name_pinned_from_arg(self):
        # MSHIP_GATEWAY_NAME is forwarded from the gateway_name arg even when absent
        # from os.environ, so metrics stamping stays correct on isolated environments.
        from modelship.deploy import serve_utils

        bound = MagicMock()
        options = MagicMock()
        options.return_value.bind.return_value = bound
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(serve_utils.ModelshipAPI, "options", options),
            patch.object(serve_utils.serve, "run"),
        ):
            serve_utils.start_gateway("edge", MagicMock(), "/edge")
        _, kwargs = options.call_args
        assert kwargs["ray_actor_options"]["runtime_env"]["env_vars"]["MSHIP_GATEWAY_NAME"] == "edge"

    @pytest.mark.parametrize(
        "name, value",
        [
            ("MSHIP_GATEWAY_REPLICAS", "0"),
            ("MSHIP_GATEWAY_REPLICAS", "-2"),
            ("MSHIP_GATEWAY_MAX_ONGOING", "0"),
            ("MSHIP_GATEWAY_MAX_ONGOING", "notanint"),
        ],
    )
    def test_rejects_invalid_env(self, name, value):
        with pytest.raises(ValueError, match=name):
            self._run({name: value})


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
        kwargs = self._init_call(
            {"MSHIP_METRICS": "true", "RAY_METRICS_EXPORT_PORT": "8079", "MSHIP_NODE_NUM_CPUS": "4"}
        )
        assert kwargs["num_cpus"] == 4
        # Guards the private ray.init kwarg that pins Ray's metrics agent port.
        assert kwargs["_metrics_export_port"] == 8079

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
        kwargs = self._init_call({}, pop=("MSHIP_RAY_DASHBOARD",))
        assert kwargs["include_dashboard"] is True
        assert kwargs["dashboard_host"] == "127.0.0.1"

    def test_dashboard_host_overridable(self):
        kwargs = self._init_call({"MSHIP_RAY_DASHBOARD": "0.0.0.0"})
        # Still on — MSHIP_RAY_DASHBOARD only ever changes the bind host now, never on/off.
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

    def test_omits_metrics_port_when_disabled(self):
        kwargs = self._init_call({"MSHIP_METRICS": "false"})
        assert "_metrics_export_port" not in kwargs

    def test_ray_port_sets_gcs_server_port(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_RAY_PORT": "6390"}, clear=False),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions"),
        ):
            os.environ.pop("RAY_GCS_SERVER_PORT", None)
            serve_utils.start_head(20)
            assert os.environ.get("RAY_GCS_SERVER_PORT") == "6390"

    def test_ray_port_absent_defaults_gcs_server_port_to_6380(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions"),
        ):
            os.environ.pop("MSHIP_RAY_PORT", None)
            os.environ.pop("RAY_GCS_SERVER_PORT", None)
            serve_utils.start_head(20)
            # Not Ray's own 6379 default — that collides with the recommended
            # same-host Redis state store under --network=host.
            assert os.environ.get("RAY_GCS_SERVER_PORT") == "6380"

    def test_ray_port_respects_explicit_gcs_server_port(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(
                os.environ,
                {
                    "MSHIP_RAY_PORT": "6380",
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
    def test_connects_via_auto_without_starting_a_node(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_RAY_PORT": "6380", "MSHIP_RAY_DASHBOARD_PORT": "8266"}, clear=False),
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
    def test_reads_the_raylet_scan(self):
        from modelship.deploy import serve_utils

        with patch("ray._private.services.find_gcs_addresses", return_value={"10.0.0.1:6380"}):
            assert serve_utils.local_ray_clusters() == {"10.0.0.1:6380"}

    def test_private_ray_scan_still_exists(self):
        # Canary: fails on a Ray bump that moves or reshapes this private helper.
        from ray._private.services import find_gcs_addresses

        assert isinstance(find_gcs_addresses(), set)


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

    def test_metrics_export_port_always_none(self):
        # A joining node never pins its metrics port — only the head's port needs to be
        # fixed/predictable; the same fixed value under --network=host would collide with it.
        kw = self._join({"MSHIP_METRICS": "true", "RAY_METRICS_EXPORT_PORT": "9999"})["params_kwargs"]
        assert kw["metrics_export_port"] is None

        kw = self._join({"MSHIP_METRICS": "false"})["params_kwargs"]
        assert kw["metrics_export_port"] is None

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

    def test_ray_auth_token_sets_mode(self):
        mode, token = self._resolve({"MSHIP_RAY_AUTH": "token"})
        assert mode == "token"
        assert token is None

    def test_a_token_sets_mode_and_token(self):
        mode, token = self._resolve({"MSHIP_RAY_AUTH_TOKEN": "secret"})
        assert mode == "token"
        assert token == "secret"

    def test_neither_leaves_auth_unset(self):
        assert self._resolve({}) == (None, None)

    def test_ray_auth_none_leaves_auth_unset(self):
        assert self._resolve({"MSHIP_RAY_AUTH": "none"}) == (None, None)

    def test_explicit_ray_auth_mode_wins(self):
        mode, _ = self._resolve({"MSHIP_RAY_AUTH": "token", "RAY_AUTH_MODE": "disabled"})
        # setdefault: an operator's explicit RAY_AUTH_MODE always wins.
        assert mode == "disabled"


class TestPruneRaySessions:
    """`prune_ray_sessions` resolves the temp root via Ray's own `get_ray_temp_dir()`
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


class TestSeedExpectedModels:
    """The readiness baseline the gateway's /readyz measures against."""

    @staticmethod
    def _conf():
        from modelship.infer.infer_config import ModelshipConfig

        return ModelshipConfig.model_validate(
            {
                "models": [
                    {"name": n, "model": f"org/{n}", "usecase": "generate", "loader": "vllm"}
                    for n in ("qwen", "kokoro")
                ]
            }
        )

    def _seed(self, **kwargs) -> list[str]:
        from modelship.deploy import serve_utils

        replica_coordinator = MagicMock()
        with patch("modelship.deploy.serve_utils.ray.get"):
            serve_utils.seed_expected_models(replica_coordinator, "gw", self._conf(), **kwargs)
        return replica_coordinator.set_expected.remote.call_args.args[1]

    def test_seeds_every_configured_model(self):
        assert self._seed() == ["qwen", "kokoro"]

    def test_excluded_models_are_left_out(self):
        # A model the driver gave up on: still in the effective config for a later
        # retry, but nothing is pursuing it now, so /readyz must stop waiting.
        assert self._seed(exclude={"qwen"}) == ["kokoro"]
