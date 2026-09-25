"""Tests for Modelship model configuration parsing and validation."""

import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from modelship.infer.infer_config import (
    AutoscalingConfig,
    LlamaServerConfig,
    ModelLoader,
    ModelshipConfig,
    ModelshipModelConfig,
    ModelUsecase,
    VllmEngineConfig,
    resolve_gpu_memory_utilization,
)
from modelship.utils.config_schema import parse_deployment_name


class TestLlamaServerConfig:
    def test_defaults(self):
        config = LlamaServerConfig()
        assert config.n_ctx == 2048
        assert config.n_batch == 512
        assert config.n_gpu_layers == -1
        assert config.threads is None
        assert config.parallel == 1
        assert config.chat_template is None
        assert config.cache_reuse == 0
        assert config.context_shift is False
        assert config.cache_ram_mib is None
        assert config.ubatch_size == 512
        assert config.flash_attn == "auto"
        assert config.cache_type_k == "f16"
        assert config.cache_type_v == "f16"
        assert config.tensor_split is None

    def test_custom_values(self):
        config = LlamaServerConfig(
            n_ctx=4096,
            n_batch=1024,
            n_gpu_layers=33,
            threads=8,
            parallel=4,
            chat_template="chatml",
            cache_reuse=256,
            context_shift=True,
            cache_ram_mib=4096,
            ubatch_size=256,
            flash_attn="off",
            cache_type_k="q8_0",
            cache_type_v="q8_0",
            tensor_split=[3.0, 1.0],
        )
        assert config.n_ctx == 4096
        assert config.n_batch == 1024
        assert config.n_gpu_layers == 33
        assert config.threads == 8
        assert config.parallel == 4
        assert config.chat_template == "chatml"
        assert config.cache_reuse == 256
        assert config.context_shift is True
        assert config.cache_ram_mib == 4096
        assert config.ubatch_size == 256
        assert config.flash_attn == "off"
        assert config.cache_type_k == "q8_0"
        assert config.cache_type_v == "q8_0"
        assert config.tensor_split == [3.0, 1.0]

    def test_empty_tensor_split_rejected(self):
        with pytest.raises(ValidationError):
            LlamaServerConfig(tensor_split=[])

    def test_non_positive_ubatch_size_rejected(self):
        with pytest.raises(ValidationError):
            LlamaServerConfig(ubatch_size=0)

    def _num_gpus_model(self, num_gpus: float) -> ModelshipModelConfig:
        return ModelshipModelConfig(
            name="test-model",
            model="repo/Qwen-GGUF:*Q4_K_M.gguf",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
            num_gpus=num_gpus,
        )

    def test_num_gpus_integer_allowed(self):
        config = self._num_gpus_model(1)
        assert config.num_gpus == 1

    def test_num_gpus_zero_allowed(self):
        config = self._num_gpus_model(0)
        assert config.num_gpus == 0

    def test_num_gpus_fractional_allowed(self):
        config = self._num_gpus_model(0.5)
        assert config.num_gpus == 0.5

    def test_num_gpus_non_integer_at_or_above_one_rejected(self):
        with pytest.raises(ValidationError, match="not allowed for the llama_server loader"):
            self._num_gpus_model(1.5)

    def test_llama_server_model_config(self):
        config = ModelshipModelConfig(
            name="llama-3",
            model="meta-llama/Llama-3-8B-Instruct-GGUF:*Q4_K_M.gguf",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
            llama_server_config=LlamaServerConfig(parallel=4),
        )
        assert config.loader == ModelLoader.llama_server
        assert config.llama_server_config is not None
        assert config.llama_server_config.parallel == 4


class TestWholeGpuOnlyLoadersNumGpus:
    def test_whispercpp_fractional_allowed(self):
        config = ModelshipModelConfig(
            name="test-stt",
            model="some-model",
            usecase=ModelUsecase.transcription,
            loader=ModelLoader.whispercpp,
            num_gpus=0.5,
        )
        assert config.num_gpus == 0.5

    def test_whispercpp_non_integer_at_or_above_one_rejected(self):
        with pytest.raises(ValidationError, match="not allowed for the whispercpp loader"):
            ModelshipModelConfig(
                name="test-stt",
                model="some-model",
                usecase=ModelUsecase.transcription,
                loader=ModelLoader.whispercpp,
                num_gpus=1.5,
            )

    def test_sherpa_onnx_fractional_accepted(self):
        # sherpa_onnx never touches CUDA (actor_options forces num_gpus to 0), so
        # it's exempt from the whole-GPU-only validator entirely.
        config = ModelshipModelConfig(
            name="tts",
            model="kokoro-en-v0_19",
            usecase=ModelUsecase.tts,
            loader=ModelLoader.sherpa_onnx,
            num_gpus=0.5,
        )
        assert config.num_gpus == 0.5


class TestModelshipModelConfig:
    def test_minimal_vllm_model(self):
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-org/some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
        )
        assert config.name == "test-llm"
        assert config.loader == ModelLoader.vllm
        assert config.num_gpus == 0
        assert config.num_cpus == 0.1
        assert config.chat_template_kwargs == {}

    def test_chat_template_kwargs_round_trips(self):
        config = ModelshipModelConfig.model_validate(
            {
                "name": "qwen3",
                "model": "some-org/qwen3",
                "usecase": ModelUsecase.generate,
                "loader": ModelLoader.llama_server,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        assert config.chat_template_kwargs == {"enable_thinking": False}

    def test_model_required(self):
        with pytest.raises(ValidationError, match="`model:` is required for loader"):
            ModelshipModelConfig(
                name="test-llm",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
            )

    def test_loader_required(self):
        with pytest.raises(ValidationError, match="Field required"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
            )

    def test_diffusers_usecase_defaults_to_image(self):
        config = ModelshipModelConfig(
            name="test-image",
            model="stabilityai/sdxl-turbo",
            loader=ModelLoader.diffusers,
        )
        assert config.usecase is ModelUsecase.image

    def test_diffusers_explicit_image_usecase_ok(self):
        config = ModelshipModelConfig(
            name="test-image",
            model="stabilityai/sdxl-turbo",
            usecase=ModelUsecase.image,
            loader=ModelLoader.diffusers,
        )
        assert config.usecase is ModelUsecase.image

    def test_diffusers_rejects_non_image_usecase(self):
        with pytest.raises(ValidationError, match="loader='diffusers' only supports usecase='image'"):
            ModelshipModelConfig(
                name="test-image",
                model="stabilityai/sdxl-turbo",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.diffusers,
            )

    def test_stable_diffusion_cpp_usecase_defaults_to_image(self):
        config = ModelshipModelConfig(
            name="test-image",
            model="org/sd-gguf:*.gguf",
            loader=ModelLoader.stable_diffusion_cpp,
        )
        assert config.usecase is ModelUsecase.image

    def test_stable_diffusion_cpp_rejects_non_image_usecase(self):
        with pytest.raises(ValidationError, match="loader='stable_diffusion_cpp' only supports usecase='image'"):
            ModelshipModelConfig(
                name="test-image",
                model="org/sd-gguf:*.gguf",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.stable_diffusion_cpp,
            )

    def test_stable_diffusion_cpp_requires_model(self):
        with pytest.raises(ValidationError, match="`model:` is required"):
            ModelshipModelConfig(
                name="test-image",
                usecase=ModelUsecase.image,
                loader=ModelLoader.stable_diffusion_cpp,
            )

    def test_gpu_allocation_fraction(self):
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=0.70,
        )
        assert config.num_gpus == 0.70

    @pytest.mark.parametrize(("num_gpus", "expected"), [(0.5, 0.5), (1, 0.9), (0, 0.4)])
    def test_gpu_memory_utilization_resolves_from_num_gpus(self, num_gpus, expected):
        """A fractional num_gpus is the share Ray reserved, so it caps vLLM too;
        anything else takes the loader-appropriate default (0.4 on CPU)."""
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=num_gpus,
        )
        assert resolve_gpu_memory_utilization(config) == expected

    @pytest.mark.parametrize(("key", "value"), [("gpu_memory_utilization", 0.6), ("model", "other/m")])
    @pytest.mark.parametrize("num_gpus", [0, 0.5, 1])
    def test_derived_engine_kwargs_rejected(self, key, value, num_gpus):
        with pytest.raises(ValidationError, match="cannot be set"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
                num_gpus=num_gpus,
                vllm_engine_kwargs={key: value},
            )

    def test_num_gpus_integer_required_above_one(self):
        with pytest.raises(ValidationError, match="must be integers"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
                num_gpus=1.5,
            )

    def test_num_gpus_auto_derives_tp(self):
        # num_gpus=3 with default tp/pp -> tp becomes 3, num_gpus normalizes to per-slot share.
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=3,
        )
        assert config.vllm_engine_kwargs.tensor_parallel_size == 3
        assert config.num_gpus == 1.0

    def test_explicit_tp_matching_num_gpus_accepted(self):
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=4,
            vllm_engine_kwargs=VllmEngineConfig(tensor_parallel_size=2, pipeline_parallel_size=2),
        )
        assert config.vllm_engine_kwargs.tensor_parallel_size == 2
        assert config.vllm_engine_kwargs.pipeline_parallel_size == 2
        assert config.num_gpus == 1.0

    def test_explicit_tp_inconsistent_with_num_gpus_rejected(self):
        with pytest.raises(ValidationError, match="does not match tensor_parallel_size"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
                num_gpus=2,
                vllm_engine_kwargs=VllmEngineConfig(tensor_parallel_size=3),
            )

    def test_fractional_num_gpus_with_tp_rejected(self):
        with pytest.raises(ValidationError, match=r"fractional.*not compatible.*tensor_parallel"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
                num_gpus=0.3,
                vllm_engine_kwargs=VllmEngineConfig(tensor_parallel_size=2),
            )

    def test_fractional_num_gpus_with_pp_rejected(self):
        with pytest.raises(ValidationError, match=r"fractional.*not compatible.*tensor_parallel"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
                num_gpus=0.5,
                vllm_engine_kwargs=VllmEngineConfig(pipeline_parallel_size=2),
            )

    def test_num_gpus_redundant_with_tp_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="config"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
                num_gpus=2,
                vllm_engine_kwargs=VllmEngineConfig(tensor_parallel_size=2),
            )
        assert any("redundant" in rec.message for rec in caplog.records)

    def test_non_vllm_loader_skips_tp_derivation(self):
        # Non-vllm loaders have no parallelism config; num_gpus stays as-is
        # for the loader to interpret directly.
        config = ModelshipModelConfig(
            name="test-stt",
            model="some-model",
            usecase=ModelUsecase.transcription,
            loader=ModelLoader.whispercpp,
            num_gpus=2,
        )
        assert config.num_gpus == 2

    def test_all_usecases_valid(self):
        for usecase in ModelUsecase:
            config = ModelshipModelConfig(
                name=f"test-{usecase.value}",
                model="some-model",
                usecase=usecase,
                loader=ModelLoader.vllm,
            )
            assert config.usecase == usecase

    def test_all_loaders_valid(self):
        image_only = (ModelLoader.diffusers, ModelLoader.stable_diffusion_cpp)
        for loader in ModelLoader:
            # diffusers / stable_diffusion_cpp are image-only; sherpa_onnx is tts-only (and needs
            # a registry name, not an arbitrary model string); everything else supports generate.
            if loader is ModelLoader.sherpa_onnx:
                usecase, model = ModelUsecase.tts, "kokoro-en-v0_19"
            elif loader in image_only:
                usecase, model = ModelUsecase.image, "some-model"
            else:
                usecase, model = ModelUsecase.generate, "some-model"
            kwargs = {"name": "test", "model": model, "usecase": usecase}
            config = ModelshipModelConfig(loader=loader, **kwargs)
            assert config.loader == loader


class TestVllmEngineConfig:
    def test_defaults(self):
        config = VllmEngineConfig()
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
        assert config.dtype == "auto"
        assert config.trust_remote_code is False

    def test_custom_values(self):
        config = VllmEngineConfig(
            tensor_parallel_size=2,
            max_model_len=12288,
            enable_auto_tool_choice=True,
            tool_call_parser="llama3_json",
        )
        assert config.tensor_parallel_size == 2
        assert config.max_model_len == 12288
        assert config.enable_auto_tool_choice is True
        assert config.tool_call_parser == "llama3_json"


class TestModelshipConfig:
    def test_multi_model_config(self):
        config = ModelshipConfig(
            models=[
                ModelshipModelConfig(
                    name="llm",
                    model="some-org/some-llm",
                    usecase=ModelUsecase.generate,
                    loader=ModelLoader.vllm,
                    num_gpus=0.70,
                ),
                ModelshipModelConfig(
                    name="tts",
                    model="kokoro-en-v0_19",
                    usecase=ModelUsecase.tts,
                    loader=ModelLoader.sherpa_onnx,
                    num_gpus=0,
                ),
            ]
        )
        assert len(config.models) == 2
        assert config.models[0].name == "llm"
        assert config.models[1].name == "tts"

    def test_empty_models_list(self):
        config = ModelshipConfig(models=[])
        assert len(config.models) == 0

    def test_duplicate_name_different_config_rejected(self):
        # A model name maps to exactly one deployment: same name with a different config is a hard error.
        with pytest.raises(ValidationError, match="duplicate model name"):
            ModelshipConfig(
                models=[
                    ModelshipModelConfig(
                        name="kokoro",
                        model="kokoro-en-v0_19",
                        usecase=ModelUsecase.tts,
                        loader=ModelLoader.sherpa_onnx,
                        num_gpus=0,
                        num_cpus=1,
                    ),
                    ModelshipModelConfig(
                        name="kokoro",
                        model="kokoro-en-v0_19",
                        usecase=ModelUsecase.tts,
                        loader=ModelLoader.sherpa_onnx,
                        num_gpus=0,
                        num_cpus=2,
                    ),
                ]
            )

    def test_duplicate_name_and_fingerprint_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate model entries"):
            ModelshipConfig(
                models=[
                    ModelshipModelConfig(
                        name="qwen",
                        model="Qwen/Qwen-7B",
                        usecase=ModelUsecase.generate,
                        loader=ModelLoader.vllm,
                        num_gpus=0.5,
                    ),
                    ModelshipModelConfig(
                        name="qwen",
                        model="Qwen/Qwen-7B",
                        usecase=ModelUsecase.generate,
                        loader=ModelLoader.vllm,
                        num_gpus=0.5,
                    ),
                ]
            )


class TestFingerprint:
    def _cfg(self, **overrides):
        base = dict(
            name="qwen",
            model="Qwen/Qwen-7B",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=0.5,
        )
        base.update(overrides)
        return ModelshipModelConfig(**base)

    def test_stable_across_instances(self):
        assert self._cfg().fingerprint() == self._cfg().fingerprint()

    def test_changes_when_num_gpus_differs(self):
        assert self._cfg(num_gpus=0.7).fingerprint() != self._cfg(num_gpus=0.8).fingerprint()

    def test_unaffected_by_name(self):
        # Same config under a different name should fingerprint identically;
        # the name is the deployment-name prefix, not part of the hash.
        assert self._cfg(name="a").fingerprint() == self._cfg(name="b").fingerprint()

    def test_unaffected_by_num_replicas(self):
        # Replica count is a Ray Serve in-place rebind, not a config drift.
        assert self._cfg(num_replicas=1).fingerprint() == self._cfg(num_replicas=4).fingerprint()

    def test_changes_when_loader_differs(self):
        assert (
            self._cfg(loader=ModelLoader.vllm).fingerprint()
            != self._cfg(loader=ModelLoader.llama_server, num_gpus=0).fingerprint()
        )

    def test_deployment_name_is_gateway_then_name_and_fingerprint(self):
        cfg = self._cfg()
        assert cfg.deployment_name("gw") == f"gw.{cfg.name}-{cfg.fingerprint()}"
        assert len(cfg.fingerprint()) == 10

    def test_deployment_name_distinct_per_gateway(self):
        cfg = self._cfg()
        assert cfg.deployment_name("gw-a") != cfg.deployment_name("gw-b")


class TestParseDeploymentName:
    def _name(self, model: str, gateway: str = "gw") -> str:
        cfg = ModelshipModelConfig(name=model, model="org/m", usecase="generate", loader="llama_server")
        return cfg.deployment_name(gateway)

    @pytest.mark.parametrize("model", ["qwen", "qwen2.5-7b", "a-0123456789", "org/model"])
    def test_round_trips_a_deployment_name(self, model):
        assert parse_deployment_name(self._name(model, "edge-1")) == ("edge-1", model)

    @pytest.mark.parametrize(
        "app", ["modelship", "qwen-0123456789", "gw.qwen", "gw.qwen-012345678", "gw.qwen-0123456789a", ".q-0123456789"]
    )
    def test_rejects_other_app_names(self, app):
        assert parse_deployment_name(app) is None


class TestNumReplicas:
    def test_default_num_replicas(self):
        config = ModelshipModelConfig(
            name="test",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
        )
        assert config.num_replicas == 1

    def test_custom_num_replicas(self):
        config = ModelshipModelConfig(
            name="test",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_replicas=3,
        )
        assert config.num_replicas == 3


class TestAutoscalingConfig:
    def _model(self, **overrides):
        base = dict(
            name="test",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
        )
        base.update(overrides)
        return ModelshipModelConfig(**base)

    def test_default_is_none(self):
        assert self._model().autoscaling_config is None

    def test_to_serve_dict_omits_unset_tunables(self):
        cfg = AutoscalingConfig(min_replicas=1, max_replicas=4)
        assert cfg.to_serve_dict() == {"min_replicas": 1, "max_replicas": 4}

    def test_to_serve_dict_includes_set_tunables(self):
        cfg = AutoscalingConfig(
            min_replicas=0,
            max_replicas=8,
            initial_replicas=2,
            target_ongoing_requests=5,
            upscale_delay_s=10,
            downscale_delay_s=600,
        )
        assert cfg.to_serve_dict() == {
            "min_replicas": 0,
            "max_replicas": 8,
            "initial_replicas": 2,
            "target_ongoing_requests": 5,
            "upscale_delay_s": 10,
            "downscale_delay_s": 600,
        }

    def test_scale_to_zero_allowed(self):
        cfg = AutoscalingConfig(min_replicas=0, max_replicas=3)
        assert cfg.min_replicas == 0

    def test_max_below_min_rejected(self):
        with pytest.raises(ValidationError, match=r"max_replicas .* must be >= "):
            AutoscalingConfig(min_replicas=4, max_replicas=2)

    def test_initial_outside_bounds_rejected(self):
        with pytest.raises(ValidationError, match=r"initial_replicas .* must be within"):
            AutoscalingConfig(min_replicas=1, max_replicas=4, initial_replicas=9)

    def test_negative_min_rejected(self):
        with pytest.raises(ValidationError):
            AutoscalingConfig(min_replicas=-1, max_replicas=4)

    def test_accepted_on_model(self):
        config = self._model(autoscaling_config={"min_replicas": 1, "max_replicas": 5})
        assert config.autoscaling_config is not None
        assert config.autoscaling_config.max_replicas == 5

    def test_explicit_num_replicas_with_autoscaling_rejected(self):
        with pytest.raises(ValidationError, match="either num_replicas or autoscaling_config"):
            self._model(num_replicas=2, autoscaling_config={"min_replicas": 1, "max_replicas": 4})

    def test_default_num_replicas_with_autoscaling_allowed(self):
        # An untouched num_replicas default must not trip the mutual-exclusivity check.
        config = self._model(autoscaling_config={"min_replicas": 1, "max_replicas": 4})
        assert config.autoscaling_config is not None

    def test_excluded_from_fingerprint(self):
        # Changing scaling bounds is an in-place Serve rebind, not config drift.
        a = self._model(autoscaling_config={"min_replicas": 1, "max_replicas": 2})
        b = self._model(autoscaling_config={"min_replicas": 3, "max_replicas": 9})
        plain = self._model()
        assert a.fingerprint() == b.fingerprint() == plain.fingerprint()


def _repo_yaml_configs() -> list[Path]:
    """Every models.yaml checked into the repo."""
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / "config" / "examples").glob("*.yaml")) + sorted((root / "bench" / "configs").glob("*.yaml"))
    assert len(paths) > 10, f"config discovery found only {len(paths)} files — did a directory move?"
    return paths


class TestStrictSchema:
    """`extra="forbid"`: an unknown key is a typo, and reaches the same error from
    models.yaml and from the CLI flags generated off these same fields."""

    _BASE: ClassVar[dict] = {"name": "m", "model": "x.gguf", "usecase": "generate", "loader": "llama_server"}

    def test_unknown_root_key_rejected(self):
        with pytest.raises(ValidationError, match="n_ctx"):
            ModelshipModelConfig.model_validate({**self._BASE, "n_ctx": 4096})

    def test_unknown_nested_key_rejected(self):
        with pytest.raises(ValidationError, match="n_ctxx"):
            ModelshipModelConfig.model_validate({**self._BASE, "llama_server_config": {"n_ctxx": 4096}})

    def test_vllm_engine_kwargs_model_rejected(self):
        raw = {**self._BASE, "model": "org/m", "loader": "vllm", "vllm_engine_kwargs": {"model": "other/m"}}
        with pytest.raises(ValidationError, match=r"vllm_engine_kwargs\.model cannot be set"):
            ModelshipModelConfig.model_validate(raw)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("num_gpus", -1),
            ("num_gpus", -0.5),
            ("num_cpus", -2),
            ("num_replicas", 0),
            ("num_replicas", -1),
            ("max_ongoing_requests", 0),
        ],
    )
    def test_out_of_range_resource_values_rejected(self, key, value):
        """A negative num_gpus read as "not fractional" all the way down to Ray,
        which then got a reservation it can't satisfy."""
        with pytest.raises(ValidationError, match="greater than or equal"):
            ModelshipModelConfig.model_validate({**self._BASE, key: value})

    def test_num_gpus_zero_and_fractional_still_allowed(self):
        for value in (0, 0.5, 2):
            assert ModelshipModelConfig.model_validate({**self._BASE, "num_gpus": value}).num_gpus == value

    @pytest.mark.parametrize(
        "path",
        sorted(_repo_yaml_configs()),
        ids=lambda p: f"{p.parent.name}/{p.name}",
    )
    def test_checked_in_configs_validate(self, path):
        """bench/configs included: nothing else in CI validates them."""
        from modelship.deploy.config import load_yaml_config

        assert load_yaml_config(str(path)).models


class TestPreRayImportChain:
    """These modules run before resolve_ray_auth_env(), and ray latches RAY_AUTH_MODE at
    import; subprocess-based since ray is already in sys.modules by suite time.
    huggingface_hub latches HF_HOME the same way."""

    @pytest.mark.parametrize(
        "module",
        [
            "modelship.utils.model_ref",
            "modelship.utils.config_schema",
            "modelship.utils.cli",
            "modelship.utils.model_flags",
            "modelship.deploy.config",
            "modelship.launcher",
        ],
    )
    @pytest.mark.parametrize("heavy", ["ray", "huggingface_hub"])
    def test_import_stays_light(self, module, heavy):
        code = f"import sys; import {module}; print({heavy!r} in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "False", f"{module} imports {heavy}"


class TestMaxModelLenValidation:
    """A context length or nothing. vLLM's -1 auto-fit sentinel is derived by the
    actor, so preflight's `x or fallback` reads can never see a negative."""

    def _cfg(self, value):
        from modelship.utils.config_schema import VllmEngineConfig

        return VllmEngineConfig(max_model_len=value)

    def test_a_positive_context_length_is_accepted(self):
        assert self._cfg(4096).max_model_len == 4096

    def test_unset_stays_none(self):
        from modelship.utils.config_schema import VllmEngineConfig

        assert VllmEngineConfig().max_model_len is None

    @pytest.mark.parametrize("value", [0, -1, -2, -7])
    def test_non_positive_values_are_rejected(self, value):
        with pytest.raises(ValidationError):
            self._cfg(value)


class TestResolveMaxModelLen:
    """The engine-side half of the split: unset becomes vLLM's auto-fit sentinel."""

    def test_a_set_length_passes_through(self):
        from modelship.infer.vllm.vllm_infer import resolve_max_model_len
        from modelship.utils.config_schema import VllmEngineConfig

        assert resolve_max_model_len(VllmEngineConfig(max_model_len=8192)) == 8192

    def test_unset_becomes_the_auto_fit_sentinel(self):
        from modelship.infer.vllm.vllm_infer import _AUTO_FIT_MAX_MODEL_LEN, resolve_max_model_len
        from modelship.utils.config_schema import VllmEngineConfig

        assert resolve_max_model_len(VllmEngineConfig()) == _AUTO_FIT_MAX_MODEL_LEN
