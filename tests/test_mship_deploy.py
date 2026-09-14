"""Tests for mship_deploy.py CLI argument parsing and helpers."""

import os
from unittest.mock import MagicMock, patch

import pytest

from modelship.deploy.actor_options import (
    build_cache_env_vars,
    build_deployment_options,
    build_passthrough_env_vars,
    total_cpu_reservation,
    total_gpu_reservation,
)
from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase, VllmEngineConfig
from modelship.utils import parse_memory_bytes, rand_suffix
from modelship.utils.cli import apply_args_to_env, parse_args


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
    def test_defaults(self):
        args = parse_args([])
        assert args.config is None
        assert args.reconcile is False
        assert args.gateway_name is None
        assert args.use_existing_ray_cluster is None

    def test_reconcile_flag(self):
        args = parse_args(["--reconcile"])
        assert args.reconcile is True
        assert args.replace_strategy == "blue_green"

    def test_reconcile_with_stop_start_strategy(self):
        args = parse_args(["--reconcile", "--replace-strategy", "stop_start"])
        assert args.reconcile is True
        assert args.replace_strategy == "stop_start"

    def test_config_path(self):
        args = parse_args(["--config", "/some/path/models.yaml"])
        assert args.config == "/some/path/models.yaml"

    def test_gateway_replicas(self):
        assert parse_args(["--gateway-replicas", "3"]).gateway_replicas == 3

    def test_gateway_replicas_defaults_to_none(self):
        assert parse_args([]).gateway_replicas is None

    def test_gateway_name(self):
        args = parse_args(["--gateway-name", "my-gateway"])
        assert args.gateway_name == "my-gateway"

    def test_ray_auth(self):
        assert parse_args(["--ray-auth", "none"]).ray_auth == "none"

    def test_ray_auth_defaults_to_none(self):
        assert parse_args([]).ray_auth is None

    def test_ray_port(self):
        assert parse_args(["--ray-port", "6380"]).ray_port == 6380

    def test_ray_port_defaults_to_none(self):
        assert parse_args([]).ray_port is None

    def test_dashboard_port(self):
        assert parse_args(["--dashboard-port", "8266"]).dashboard_port == 8266

    def test_dashboard_port_defaults_to_none(self):
        assert parse_args([]).dashboard_port is None

    def test_address(self):
        assert parse_args(["--address", "mship-head:6380"]).address == "mship-head:6380"

    def test_address_defaults_to_none(self):
        assert parse_args([]).address is None

    def test_token(self):
        assert parse_args(["--token", "secret"]).token == "secret"

    def test_token_defaults_to_none(self):
        assert parse_args([]).token is None

    def test_node_num_cpus(self):
        assert parse_args(["--node-num-cpus", "4"]).node_num_cpus == 4

    def test_node_num_cpus_defaults_to_none(self):
        assert parse_args([]).node_num_cpus is None

    def test_node_num_gpus(self):
        assert parse_args(["--node-num-gpus", "2"]).node_num_gpus == 2

    def test_node_num_gpus_defaults_to_none(self):
        assert parse_args([]).node_num_gpus is None

    def test_node_memory(self):
        assert parse_args(["--node-memory", "8Gi"]).node_memory == 8 * 1024**3

    def test_node_memory_defaults_to_none(self):
        assert parse_args([]).node_memory is None

    def test_responses_ttl_s(self):
        assert parse_args(["--responses-ttl-s", "60"]).responses_ttl_s == 60.0

    def test_responses_ttl_s_defaults_to_none(self):
        assert parse_args([]).responses_ttl_s is None

    def test_state_sweep_interval_s(self):
        assert parse_args(["--state-sweep-interval-s", "30"]).state_sweep_interval_s == 30.0

    def test_state_sweep_interval_s_defaults_to_none(self):
        assert parse_args([]).state_sweep_interval_s is None

    def test_all_flags_combined(self):
        args = parse_args(
            [
                "--config",
                "llm.yaml",
                "--gateway-name",
                "llm-api",
                "--reconcile",
                "--use-existing-ray-cluster",
            ]
        )
        assert args.config == "llm.yaml"
        assert args.gateway_name == "llm-api"
        assert args.reconcile is True
        assert args.use_existing_ray_cluster is True


class TestApplyArgsToEnv:
    def test_state_store_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_STATE_STORE", raising=False)
        apply_args_to_env(parse_args(["--state-store", "redis://cache:6379/0"]))
        assert os.environ["MSHIP_STATE_STORE"] == "redis://cache:6379/0"

    def test_state_store_flag_overrides_preset_env(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_STORE", "redis://from-env:6379/0")
        apply_args_to_env(parse_args(["--state-store", "redis://from-flag:6379/0"]))
        assert os.environ["MSHIP_STATE_STORE"] == "redis://from-flag:6379/0"

    def test_no_state_store_leaves_env_untouched(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_STORE", "redis://preexisting:6379/0")
        apply_args_to_env(parse_args([]))
        assert os.environ["MSHIP_STATE_STORE"] == "redis://preexisting:6379/0"

    def test_gateway_replicas_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_GATEWAY_REPLICAS", raising=False)
        apply_args_to_env(parse_args(["--gateway-replicas", "4"]))
        assert os.environ["MSHIP_GATEWAY_REPLICAS"] == "4"

    def test_ray_auth_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RAY_AUTH", raising=False)
        apply_args_to_env(parse_args(["--ray-auth", "none"]))
        assert os.environ["MSHIP_RAY_AUTH"] == "none"

    def test_ray_auth_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RAY_AUTH", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_RAY_AUTH" not in os.environ

    def test_ray_port_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RAY_PORT", raising=False)
        apply_args_to_env(parse_args(["--ray-port", "6380"]))
        assert os.environ["MSHIP_RAY_PORT"] == "6380"

    def test_ray_port_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RAY_PORT", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_RAY_PORT" not in os.environ

    def test_dashboard_port_sets_env(self):
        # patch.dict (not monkeypatch.delenv) so the env write is reverted on exit —
        # see test_node_num_gpus_sets_env's comment for why delenv alone doesn't do it.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_RAY_DASHBOARD_PORT", None)
            apply_args_to_env(parse_args(["--dashboard-port", "8266"]))
            assert os.environ["MSHIP_RAY_DASHBOARD_PORT"] == "8266"

    def test_dashboard_port_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RAY_DASHBOARD_PORT", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_RAY_DASHBOARD_PORT" not in os.environ

    def test_address_sets_env(self):
        # patch.dict reverts the env write on exit; monkeypatch.delenv wouldn't undo
        # it here. A leaked MSHIP_ADDRESS would flip later TestConnectRay(Join) tests.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_ADDRESS", None)
            apply_args_to_env(parse_args(["--address", "mship-head:6380"]))
            assert os.environ["MSHIP_ADDRESS"] == "mship-head:6380"

    def test_address_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_ADDRESS", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_ADDRESS" not in os.environ

    def test_token_sets_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_RAY_AUTH_TOKEN", None)
            apply_args_to_env(parse_args(["--token", "secret"]))
            assert os.environ["MSHIP_RAY_AUTH_TOKEN"] == "secret"

    def test_token_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RAY_AUTH_TOKEN", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_RAY_AUTH_TOKEN" not in os.environ

    def test_node_num_cpus_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_NODE_NUM_CPUS", raising=False)
        apply_args_to_env(parse_args(["--node-num-cpus", "4"]))
        assert os.environ["MSHIP_NODE_NUM_CPUS"] == "4"

    def test_node_num_cpus_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_NODE_NUM_CPUS", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_NODE_NUM_CPUS" not in os.environ

    def test_node_num_gpus_sets_env(self):
        # patch.dict (not monkeypatch.delenv) reverts the env write on exit —
        # delenv on an already-absent var registers no cleanup, leaking "2" into later tests.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_NODE_NUM_GPUS", None)
            apply_args_to_env(parse_args(["--node-num-gpus", "2"]))
            assert os.environ["MSHIP_NODE_NUM_GPUS"] == "2"

    def test_node_num_gpus_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_NODE_NUM_GPUS", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_NODE_NUM_GPUS" not in os.environ

    def test_node_memory_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_NODE_MEMORY", raising=False)
        apply_args_to_env(parse_args(["--node-memory", "8Gi"]))
        assert os.environ["MSHIP_NODE_MEMORY"] == str(8 * 1024**3)

    def test_node_memory_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_NODE_MEMORY", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_NODE_MEMORY" not in os.environ

    def test_prune_ray_sessions_false_sets_env(self):
        # patch.dict (not monkeypatch.delenv) so the env write is reverted on exit
        # — otherwise MSHIP_PRUNE_RAY_SESSIONS=false leaks into the prune tests.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_PRUNE_RAY_SESSIONS", None)
            apply_args_to_env(parse_args(["--prune-ray-sessions", "false"]))
            assert os.environ["MSHIP_PRUNE_RAY_SESSIONS"] == "false"

    def test_prune_ray_sessions_true_sets_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_PRUNE_RAY_SESSIONS", None)
            apply_args_to_env(parse_args(["--prune-ray-sessions", "true"]))
            assert os.environ["MSHIP_PRUNE_RAY_SESSIONS"] == "true"

    def test_prune_ray_sessions_absent_leaves_env_untouched(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_PRUNE_RAY_SESSIONS", None)
            apply_args_to_env(parse_args([]))
            assert "MSHIP_PRUNE_RAY_SESSIONS" not in os.environ

    def test_no_preflight_sets_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_PREFLIGHT", None)
            apply_args_to_env(parse_args(["--no-preflight"]))
            assert os.environ["MSHIP_PREFLIGHT"] == "false"

    def test_no_preflight_absent_leaves_env_untouched(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_PREFLIGHT", None)
            apply_args_to_env(parse_args([]))
            assert "MSHIP_PREFLIGHT" not in os.environ

    def test_responses_ttl_s_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RESPONSES_TTL_S", raising=False)
        apply_args_to_env(parse_args(["--responses-ttl-s", "60"]))
        assert os.environ["MSHIP_RESPONSES_TTL_S"] == "60.0"

    def test_state_sweep_interval_s_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_STATE_SWEEP_INTERVAL_S", raising=False)
        apply_args_to_env(parse_args(["--state-sweep-interval-s", "30"]))
        assert os.environ["MSHIP_STATE_SWEEP_INTERVAL_S"] == "30.0"


class TestDriverCacheEnv:
    def test_cache_dir_flags_reach_the_import_latched_vars(self):
        class _StopError(Exception):
            pass

        seen: dict[str, str] = {}

        def _capture_env():
            seen.update(os.environ)
            raise _StopError

        from modelship.driver import main as driver_main

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("modelship.driver.resolve_ray_auth_env", side_effect=_capture_env),
            pytest.raises(_StopError),
        ):
            driver_main(["--cache-dir", "/custom/shared", "--node-cache-dir", "/custom/node"])

        assert seen["HF_HOME"] == "/custom/shared/huggingface"
        assert seen["VLLM_CACHE_ROOT"] == "/custom/node/vllm"
        assert seen["FLASHINFER_WORKSPACE_BASE"] == "/custom/node/flashinfer"


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
        assert env_vars["MSHIP_RESPONSES_TTL_S"] == "60"
        assert env_vars["MSHIP_STATE_SWEEP_INTERVAL_S"] == "30"

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

    def test_log_level_in_passthrough_and_deployment_env(self):
        # MSHIP_LOG_LEVEL must flow through the shared passthrough helper into a model
        # deployment's runtime_env, alongside cache vars (which the gateway path omits).
        with patch.dict(os.environ, {"MSHIP_LOG_LEVEL": "TRACE"}, clear=True):
            assert build_passthrough_env_vars()["MSHIP_LOG_LEVEL"] == "TRACE"

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

    def test_own_cluster_connect_ray_raises_on_gpu_mismatch(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(
                os.environ,
                {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_NODE_NUM_GPUS": "2"},
                clear=False,
            ),
            patch.object(serve_utils, "detect_gpus", return_value=self._fake_gpus(1)),
            patch.object(serve_utils.ray, "init") as mock_init,
            pytest.raises(RuntimeError, match="exceeds the 1 GPU"),
        ):
            serve_utils.connect_ray(20)
        mock_init.assert_not_called()

    def test_existing_cluster_branch_skips_check(self):
        # KubeRay sizes the pod itself; MSHIP_NODE_NUM_GPUS isn't read there at all.
        from modelship.deploy import serve_utils

        with (
            patch.dict(
                os.environ,
                {"MSHIP_USE_EXISTING_RAY_CLUSTER": "true", "MSHIP_NODE_NUM_GPUS": "99"},
                clear=False,
            ),
            patch.object(serve_utils, "detect_gpus") as mock_detect,
            patch.object(serve_utils.ray, "init"),
        ):
            serve_utils.connect_ray(20)
        mock_detect.assert_not_called()


class TestConnectRay:
    def _init_call(self, env, pop=()):
        """Returns the kwargs connect_ray passed to ray.init(). `pop` clears
        env vars before the call."""
        from modelship.deploy import serve_utils

        with patch.dict(os.environ, env, clear=False):
            for key in pop:
                os.environ.pop(key, None)
            with (
                patch.object(serve_utils.ray, "init") as mock_init,
                # Don't let the own-cluster branch sweep the real /tmp/ray during tests.
                patch.object(serve_utils, "prune_ray_sessions"),
            ):
                serve_utils.connect_ray(20)
        _, kwargs = mock_init.call_args
        return kwargs

    def test_existing_cluster_connects_via_auto(self):
        kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "true"})
        assert kwargs["address"] == "auto"
        # No head is started: resource/metrics kwargs must be absent.
        assert "_metrics_export_port" not in kwargs
        assert "num_cpus" not in kwargs

    def test_own_cluster_starts_head_with_metrics_port(self):
        kwargs = self._init_call(
            {
                "MSHIP_USE_EXISTING_RAY_CLUSTER": "false",
                "MSHIP_METRICS": "true",
                "RAY_METRICS_EXPORT_PORT": "8079",
                "MSHIP_NODE_NUM_CPUS": "4",
            }
        )
        assert "address" not in kwargs
        assert kwargs["num_cpus"] == 4
        # Guards the private ray.init kwarg that pins Ray's metrics agent port.
        assert kwargs["_metrics_export_port"] == 8079

    def test_own_cluster_cuda_multi_gpu_left_unset_for_autodetect(self):
        """A cuda/rocm/xpu node must not be pinned to 1 GPU when MSHIP_NODE_NUM_GPUS
        is unset — Ray autodetects the real device count."""
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_accelerator", return_value="cuda"):
            kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, pop=("MSHIP_NODE_NUM_GPUS",))
        assert "num_gpus" not in kwargs

    def test_own_cluster_cpu_accelerator_forces_zero_gpus(self):
        """A torch CPU build must advertise 0 GPUs even if nvidia-smi/NVML sees hardware."""
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_accelerator", return_value="cpu"):
            kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, pop=("MSHIP_NODE_NUM_GPUS",))
        assert kwargs["num_gpus"] == 0

    def test_own_cluster_metal_detected_advertises_one_gpu(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_accelerator", return_value="metal"):
            kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, pop=("MSHIP_NODE_NUM_GPUS",))
        assert kwargs["num_gpus"] == 1

    def test_own_cluster_explicit_zero_gpus_wins_over_metal_detection(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_accelerator", return_value="metal"):
            kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_NODE_NUM_GPUS": "0"})
        assert kwargs["num_gpus"] == 0

    def test_own_cluster_node_memory_splits_into_memory_and_object_store(self):
        kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_NODE_MEMORY": str(10 * 1024**3)})
        # 30% object store (Ray's own resolve_object_store_memory), 70% schedulable 'memory'.
        assert kwargs["object_store_memory"] == int(10 * 1024**3 * 0.3)
        assert kwargs["_memory"] == 10 * 1024**3 - kwargs["object_store_memory"]

    def test_own_cluster_node_memory_accepts_unit_suffix(self):
        kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_NODE_MEMORY": "10Gi"})
        assert kwargs["object_store_memory"] == int(10 * 1024**3 * 0.3)
        assert kwargs["_memory"] == 10 * 1024**3 - kwargs["object_store_memory"]

    def test_own_cluster_node_memory_absent_when_unset(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_available_ram_bytes", return_value=0):
            kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, pop=("MSHIP_NODE_MEMORY",))
        assert "_memory" not in kwargs
        assert "object_store_memory" not in kwargs

    def test_own_cluster_node_memory_auto_detected_when_unset(self):
        from modelship.deploy import serve_utils

        available = 10 * 1024**3
        with patch.object(serve_utils, "detect_available_ram_bytes", return_value=available):
            kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, pop=("MSHIP_NODE_MEMORY",))
        total_bytes = int(available * serve_utils._AUTO_NODE_MEMORY_HEADROOM)
        assert kwargs["object_store_memory"] == int(total_bytes * 0.3)
        assert kwargs["_memory"] == total_bytes - kwargs["object_store_memory"]

    def test_own_cluster_explicit_node_memory_wins_over_auto_detect(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "detect_available_ram_bytes", return_value=999 * 1024**3):
            kwargs = self._init_call(
                {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_NODE_MEMORY": str(10 * 1024**3)}
            )
        assert kwargs["object_store_memory"] == int(10 * 1024**3 * 0.3)
        assert kwargs["_memory"] == 10 * 1024**3 - kwargs["object_store_memory"]

    def test_own_cluster_resources_forwarded_from_capability_probe(self):
        from modelship.deploy import serve_utils

        with patch.object(serve_utils, "node_capability_resources", return_value={"mship_vllm": 1}):
            kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"})
        assert kwargs["resources"] == {"mship_vllm": 1}

    def test_own_cluster_dashboard_always_on_bound_localhost(self):
        kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, pop=("MSHIP_RAY_DASHBOARD",))
        assert kwargs["include_dashboard"] is True
        assert kwargs["dashboard_host"] == "127.0.0.1"

    def test_own_cluster_dashboard_host_overridable(self):
        kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_RAY_DASHBOARD": "0.0.0.0"})
        # Still on — MSHIP_RAY_DASHBOARD only ever changes the bind host now, never on/off.
        assert kwargs["include_dashboard"] is True
        assert kwargs["dashboard_host"] == "0.0.0.0"

    def test_own_cluster_dashboard_port_absent_when_unset(self):
        kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, pop=("MSHIP_RAY_DASHBOARD_PORT",))
        assert "dashboard_port" not in kwargs

    def test_own_cluster_dashboard_port_overridable(self):
        # Lets multiple modelship heads share one host under --network=host, where
        # Ray's own dashboard port (8265) would otherwise collide between them.
        kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_RAY_DASHBOARD_PORT": "8266"})
        assert kwargs["dashboard_port"] == 8266

    def test_existing_cluster_never_sets_dashboard_port(self):
        kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "true", "MSHIP_RAY_DASHBOARD_PORT": "8266"})
        assert "dashboard_port" not in kwargs

    def test_existing_cluster_never_sets_dashboard_kwargs(self):
        kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "true"})
        assert "include_dashboard" not in kwargs
        assert "dashboard_host" not in kwargs

    def test_own_cluster_omits_metrics_port_when_disabled(self):
        kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_METRICS": "false"})
        assert "address" not in kwargs
        assert "_metrics_export_port" not in kwargs

    def test_own_cluster_ray_port_sets_gcs_server_port(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_RAY_PORT": "6390"}, clear=False),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions"),
        ):
            os.environ.pop("RAY_GCS_SERVER_PORT", None)
            serve_utils.connect_ray(20)
            assert os.environ.get("RAY_GCS_SERVER_PORT") == "6390"

    def test_own_cluster_ray_port_absent_defaults_gcs_server_port_to_6380(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, clear=False),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions"),
        ):
            os.environ.pop("MSHIP_RAY_PORT", None)
            os.environ.pop("RAY_GCS_SERVER_PORT", None)
            serve_utils.connect_ray(20)
            # Not Ray's own 6379 default — that collides with the recommended
            # same-host Redis state store under --network=host.
            assert os.environ.get("RAY_GCS_SERVER_PORT") == "6380"

    def test_own_cluster_ray_port_respects_explicit_gcs_server_port(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(
                os.environ,
                {
                    "MSHIP_USE_EXISTING_RAY_CLUSTER": "false",
                    "MSHIP_RAY_PORT": "6380",
                    "RAY_GCS_SERVER_PORT": "6381",
                },
                clear=False,
            ),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions"),
        ):
            serve_utils.connect_ray(20)
            # setdefault: an operator's explicit RAY_GCS_SERVER_PORT always wins.
            assert os.environ["RAY_GCS_SERVER_PORT"] == "6381"

    def test_existing_cluster_never_sets_gcs_server_port(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "true", "MSHIP_RAY_PORT": "6380"}, clear=False),
            patch.object(serve_utils.ray, "init"),
        ):
            os.environ.pop("RAY_GCS_SERVER_PORT", None)
            serve_utils.connect_ray(20)
            assert "RAY_GCS_SERVER_PORT" not in os.environ

    def test_prunes_stale_sessions_on_own_cluster(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, clear=False),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions") as mock_prune,
        ):
            serve_utils.connect_ray(20)
        mock_prune.assert_called_once()

    def test_skips_prune_on_existing_cluster(self):
        from modelship.deploy import serve_utils

        # We don't own the temp root on an external cluster — never sweep it.
        with (
            patch.dict(os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "true"}, clear=False),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions") as mock_prune,
        ):
            serve_utils.connect_ray(20)
        mock_prune.assert_not_called()


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


class TestConnectRayJoinBranch:
    """connect_ray's MSHIP_ADDRESS branch: brings up the local node via
    _join_ray_cluster (mocked here — TestJoinRayCluster covers its internals),
    then attaches the driver with ray.init(address='auto')."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_join_node):
        yield

    def test_join_branch_creates_node_then_attaches_via_auto(self):
        from modelship.deploy import serve_utils

        with patch.dict(os.environ, {"MSHIP_ADDRESS": "head:6380"}, clear=False):
            os.environ.pop("MSHIP_USE_EXISTING_RAY_CLUSTER", None)
            with (
                patch.object(serve_utils, "_join_ray_cluster") as mock_join,
                patch.object(serve_utils, "prune_ray_sessions") as mock_prune,
                patch.object(serve_utils.ray, "init") as mock_init,
            ):
                serve_utils.connect_ray(20)
            mock_join.assert_called_once_with("head:6380")
            mock_prune.assert_called_once()
            # address="auto", not a bare init — a bare init would silently form a
            # split-brain cluster if local discovery somehow failed.
            assert mock_init.call_args.kwargs["address"] == "auto"

    def test_address_and_existing_cluster_mutually_exclusive_raises(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(
                os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "true", "MSHIP_ADDRESS": "head:6380"}, clear=False
            ),
            pytest.raises(RuntimeError, match="mutually exclusive"),
        ):
            serve_utils.connect_ray(20)


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
    translating MSHIP_* auth/join vars into RAY_AUTH_MODE/RAY_AUTH_TOKEN before
    mship_deploy imports ray."""

    def _resolve(self, env):
        from modelship.utils import ray_auth

        base = dict.fromkeys(
            ["MSHIP_ADDRESS", "MSHIP_USE_EXISTING_RAY_CLUSTER", "MSHIP_RAY_AUTH", "MSHIP_RAY_AUTH_TOKEN"], ""
        )
        with patch.dict(os.environ, {**base, **env}, clear=False):
            for key in ["MSHIP_ADDRESS", "MSHIP_USE_EXISTING_RAY_CLUSTER", "MSHIP_RAY_AUTH", "MSHIP_RAY_AUTH_TOKEN"]:
                if not os.environ.get(key):
                    os.environ.pop(key, None)
            os.environ.pop("RAY_AUTH_MODE", None)
            os.environ.pop("RAY_AUTH_TOKEN", None)
            ray_auth.resolve_ray_auth_env()
            return os.environ.get("RAY_AUTH_MODE"), os.environ.get("RAY_AUTH_TOKEN")

    def test_own_head_token_sets_mode(self):
        mode, _ = self._resolve({"MSHIP_RAY_AUTH": "token"})
        assert mode == "token"

    def test_join_with_token_sets_mode_and_token(self):
        mode, token = self._resolve({"MSHIP_ADDRESS": "head:6380", "MSHIP_RAY_AUTH_TOKEN": "secret"})
        assert mode == "token"
        assert token == "secret"

    def test_join_without_token_leaves_auth_unset(self):
        mode, token = self._resolve({"MSHIP_ADDRESS": "head:6380"})
        assert mode is None
        assert token is None

    def test_existing_cluster_never_sets_mode(self):
        mode, _ = self._resolve({"MSHIP_USE_EXISTING_RAY_CLUSTER": "true", "MSHIP_RAY_AUTH": "token"})
        assert mode is None

    def test_explicit_ray_auth_mode_wins(self):
        from modelship.utils import ray_auth

        with patch.dict(os.environ, {"MSHIP_RAY_AUTH": "token", "RAY_AUTH_MODE": "disabled"}, clear=False):
            os.environ.pop("MSHIP_ADDRESS", None)
            os.environ.pop("MSHIP_USE_EXISTING_RAY_CLUSTER", None)
            ray_auth.resolve_ray_auth_env()
            # setdefault: an operator's explicit RAY_AUTH_MODE always wins.
            assert os.environ["RAY_AUTH_MODE"] == "disabled"


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
