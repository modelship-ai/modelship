"""The `--model` flags must produce models.yaml entries indistinguishable from the
hand-written ones."""

import os
import subprocess
import sys

import pytest
from pydantic import BaseModel, ValidationError

from modelship.deploy.config import resolve_input_models, validate_models
from modelship.utils.cli import infer_model_name, parse_args
from modelship.utils.config_schema import ModelshipModelConfig
from modelship.utils.model_flags import GENERATED_MODEL_ARGS, _block_model, flag_locations


def _args(*argv):
    return parse_args("deploy", list(argv))


def _raw(*argv):
    models = _args(*argv).models
    assert models is not None and len(models) == 1
    return models[0]


def _error(capsys, *argv) -> str:
    with pytest.raises(SystemExit) as exc:
        _args(*argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


class TestParityWithYaml:
    def test_fingerprint_and_fields_set_match_the_yaml_entry(self):
        yaml_entry = {
            "name": "qwen",
            "model": "lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf",
            "usecase": "generate",
            "loader": "llama_server",
            "num_cpus": 3.0,
        }
        cli_entry = _raw(
            "--model", "lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf",
            "--model.name", "qwen",
            "--model.usecase", "generate",
            "--model.loader", "llama_server",
            "--model.num-cpus", "3.0",
        )  # fmt: skip
        assert cli_entry == yaml_entry

        from_yaml = validate_models([yaml_entry]).models[0]
        from_cli = validate_models([cli_entry]).models[0]
        assert from_cli.fingerprint() == from_yaml.fingerprint()
        assert from_cli.model_fields_set == from_yaml.model_fields_set

    def test_unset_flags_are_omitted_not_defaulted(self):
        raw = _raw("--model", "Qwen/Qwen3-8B", "--model.usecase", "generate", "--model.loader", "vllm")
        assert raw == {"name": "qwen3-8b", "model": "Qwen/Qwen3-8B", "usecase": "generate", "loader": "vllm"}
        assert validate_models([raw]).models[0].model_fields_set == {"name", "model", "usecase", "loader"}

    def test_num_replicas_left_unset_does_not_block_autoscaling(self):
        """The check rejects num_replicas alongside autoscaling_config via
        model_fields_set — a materialized CLI default would trip it."""
        raw = _raw(
            "--model", "Qwen/Qwen3-8B",
            "--model.usecase", "generate",
            "--model.loader", "vllm",
            "--model.autoscaling-config", "{min_replicas: 1, max_replicas: 3}",
            "--model.vllm-engine-kwargs.max-model-len", "4096",
        )  # fmt: skip
        validate_models([raw])  # no raise

    def test_image_loader_still_defaults_usecase(self):
        raw = _raw("--model", "stabilityai/sdxl-turbo", "--model.loader", "diffusers")
        assert "usecase" not in raw
        assert validate_models([raw]).models[0].usecase.value == "image"

    def test_schema_rejects_missing_loader(self):
        with pytest.raises(ValidationError, match="loader"):
            validate_models([_raw("--model", "Qwen/Qwen3-8B", "--model.usecase", "generate")])


class TestModelGroups:
    """`--model REF` opens a model; the `--model.*` flags after it belong to it."""

    def test_each_model_gets_the_flags_that_follow_it(self):
        models = _args(
            "--model", "org/qwen", "--model.loader", "vllm", "--model.num-gpus", "1",
            "--model", "kokoro-en-v0_19", "--model.loader", "sherpa_onnx", "--model.usecase", "tts",
        ).models  # fmt: skip
        assert models == [
            {"name": "qwen", "model": "org/qwen", "loader": "vllm", "num_gpus": 1},
            {"name": "kokoro-en-v0_19", "model": "kokoro-en-v0_19", "loader": "sherpa_onnx", "usecase": "tts"},
        ]

    def test_other_flags_may_go_anywhere(self):
        args = _args("--model", "org/qwen", "--wait", "--model.loader", "vllm", "--gateway-name", "home")
        assert args.wait is True
        assert args.gateway_name == "home"
        assert args.models == [{"name": "qwen", "model": "org/qwen", "loader": "vllm"}]

    def test_flags_take_inline_values(self):
        assert _raw("--model=org/qwen", "--model.loader=vllm", "--model.num-gpus=0.5") == {
            "name": "qwen",
            "model": "org/qwen",
            "loader": "vllm",
            "num_gpus": 0.5,
        }

    def test_no_model_leaves_models_unset(self):
        assert _args("--wait").models is None

    def test_a_flag_before_any_model_is_rejected(self, capsys):
        err = _error(capsys, "--model.num-gpus", "1", "--model", "org/qwen")
        assert "--model.num-gpus sets a key of the model a --model REF opens" in err

    def test_a_flag_given_twice_for_one_model_is_rejected(self, capsys):
        err = _error(capsys, "--model", "org/qwen", "--model.num-gpus", "1", "--model.num-gpus", "2")
        assert "--model.num-gpus is given twice" in err

    def test_the_same_flag_on_two_models_is_fine(self):
        models = _args("--model", "a/one", "--model.num-gpus", "1", "--model", "b/two", "--model.num-gpus", "2").models
        assert models is not None
        assert [m["num_gpus"] for m in models] == [1, 2]

    @pytest.mark.parametrize("argv", [("--model",), ("--model", "org/qwen", "--model.num-gpus")])
    def test_a_flag_without_its_value_is_rejected(self, capsys, argv):
        assert "takes a value" in _error(capsys, *argv)


class TestInferModelName:
    @pytest.mark.parametrize(
        ("ref", "expected"),
        [
            ("lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf", "qwen3-8b"),
            ("bartowski/Llama-3.3-70B-Instruct-GGUF:*Q8_0*-of-*.gguf", "llama-3.3-70b-instruct"),
            ("Qwen/Qwen3-8B", "qwen3-8b"),
            ("nomic-ai/nomic-embed-text-v1.5-GGUF:nomic-embed-text-v1.5.Q4_K_M.gguf", "nomic-embed-text-v1.5"),
            ("/models/qwen3-8b-instruct.Q4_K_M.gguf", "qwen3-8b-instruct"),
            ("/models/Qwen2.5-0.5B-Instruct.IQ4_XS.gguf", "qwen2.5-0.5b-instruct"),
            ("/models/Qwen3-8B/", "qwen3-8b"),
            ("./weights/model.f16.gguf", "model"),
            ("kokoro-en-v0_19", "kokoro-en-v0_19"),
            ("base.en", "base.en"),
        ],
    )
    def test_inference(self, ref, expected):
        assert infer_model_name(ref) == expected

    def test_explicit_name_wins(self):
        assert _raw("--model", "Qwen/Qwen3-8B", "--model.name", "custom")["name"] == "custom"

    def test_uninferrable_ref_raises(self):
        with pytest.raises(ValueError, match=r"pass --model\.name"):
            infer_model_name("///")

    def test_uninferrable_ref_is_a_usage_error(self, capsys):
        assert "pass --model.name" in _error(capsys, "--model", "///")

    def test_same_ref_infers_the_same_name(self):
        """Re-running a deploy must be idempotent: the name is the additive
        merge's replace-by-name key."""
        ref = "lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf"
        assert infer_model_name(ref) == infer_model_name(ref)


class TestResolveInputModels:
    def test_config_file_is_read_when_no_model_flag(self, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text("models:\n  - name: m\n    model: x.gguf\n    loader: llama_server\n    usecase: generate\n")
        raw = resolve_input_models(_args("--config", str(config)))
        assert raw == [{"name": "m", "model": "x.gguf", "loader": "llama_server", "usecase": "generate"}]

    def test_neither_returns_none(self):
        assert resolve_input_models(_args()) is None

    def test_cli_models_are_returned_as_given(self):
        args = _args("--model", "a/one", "--model.loader", "vllm", "--model", "b/two", "--model.loader", "vllm")
        assert resolve_input_models(args) == args.models

    def test_both_is_rejected_at_parse_time(self, tmp_path, capsys):
        config = tmp_path / "models.yaml"
        config.write_text("models: []\n")
        assert "mutually exclusive" in _error(capsys, "--model", "Qwen/Qwen3-8B", "--config", str(config))

    def test_both_is_rejected_for_a_hand_built_namespace(self, tmp_path):
        args = _args("--config", str(tmp_path / "models.yaml"))
        args.models = [{"name": "qwen3-8b", "model": "Qwen/Qwen3-8B"}]
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_input_models(args)

    def test_cancel_takes_no_model(self, capsys):
        assert "--cancel takes no model options: --model" in _error(capsys, "--cancel", "abc", "--model", "org/q")


class TestFailsBeforeRay:
    def test_bad_model_flag_exits_without_importing_ray(self):
        code = (
            "import sys; from modelship.launcher import _cmd_run\n"
            "try:\n"
            "    _cmd_run('start', ['--model', 'Qwen/Qwen3-8B'])\n"
            "except SystemExit as e:\n"
            "    print('exit', e.code, 'ray' in sys.modules)\n"
        )
        env = {**os.environ, "MSHIP_VARIANT": "cpu"}
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "exit 1 False"


class TestNestedFlags:
    """Every key at every depth has a flag named for its config path, valued as YAML text."""

    _LLAMA = ("--model", "x.gguf", "--model.usecase", "generate", "--model.loader", "llama_server")
    _VLLM = ("--model", "org/qwen", "--model.name", "qwen", "--model.usecase", "generate", "--model.loader", "vllm")

    def test_fingerprint_and_fields_set_match_the_yaml_entry(self):
        yaml_entry = {
            "name": "qwen",
            "model": "org/qwen",
            "usecase": "generate",
            "loader": "vllm",
            "num_gpus": 2,
            "vllm_engine_kwargs": {
                "max_model_len": 8192,
                "enforce_eager": True,
                "limit_mm_per_prompt": {"image": 2},
            },
        }
        cli_entry = _raw(
            *self._VLLM,
            "--model.num-gpus", "2",
            "--model.vllm-engine-kwargs.max-model-len", "8192",
            "--model.vllm-engine-kwargs.enforce-eager", "true",
            "--model.vllm-engine-kwargs.limit-mm-per-prompt", "{image: 2}",
        )  # fmt: skip
        assert cli_entry == yaml_entry

        from_yaml = validate_models([yaml_entry]).models[0]
        from_cli = validate_models([cli_entry]).models[0]
        assert from_cli.fingerprint() == from_yaml.fingerprint()
        assert from_cli.model_fields_set == from_yaml.model_fields_set
        assert from_cli.vllm_engine_kwargs.model_fields_set == from_yaml.vllm_engine_kwargs.model_fields_set

    def test_untouched_block_stays_absent(self):
        """Absent, not an empty dict: `llama_server_config: {}` is a different
        fingerprint and a different `model_fields_set`."""
        raw = _raw(*self._LLAMA)
        assert "llama_server_config" not in raw
        assert validate_models([raw]).models[0].llama_server_config is None

    def test_explicit_null_is_a_value_not_an_omission(self):
        raw = _raw(*self._LLAMA, "--model.llama-server-config.threads", "null")
        assert raw["llama_server_config"] == {"threads": None}
        config = validate_models([raw]).models[0]
        assert config.llama_server_config is not None
        assert "threads" in config.llama_server_config.model_fields_set

    def test_a_block_flag_sets_the_whole_block(self):
        raw = _raw(*self._LLAMA, "--model.llama-server-config", "{n_ctx: 4096, parallel: 2}")
        assert raw["llama_server_config"] == {"n_ctx": 4096, "parallel": 2}

    def test_deeper_flags_set_keys_inside_a_block_flag_in_any_order(self):
        raw = _raw(
            *self._LLAMA,
            "--model.llama-server-config.n-ctx", "8192",
            "--model.llama-server-config", "{n_ctx: 4096, parallel: 2}",
        )  # fmt: skip
        assert raw["llama_server_config"] == {"n_ctx": 8192, "parallel": 2}

    def test_a_key_inside_a_non_mapping_block_is_rejected(self, capsys):
        err = _error(
            capsys, *self._LLAMA, "--model.llama-server-config", "null", "--model.llama-server-config.n-ctx", "1"
        )
        assert "--model.llama-server-config is set to None, not a mapping" in err

    def test_keys_three_levels_deep_have_flags(self):
        raw = _raw(*self._VLLM, "--model.vllm-engine-kwargs.reasoning-config.reasoning-end-str", "'</think>'")
        assert raw["vllm_engine_kwargs"] == {"reasoning_config": {"reasoning_end_str": "</think>"}}

    @pytest.mark.parametrize(
        ("flag", "text", "path", "expected"),
        [
            ("--model.llama-server-config.n-ctx", "8192", ("llama_server_config", "n_ctx"), 8192),
            ("--model.llama-server-config.context-shift", "true", ("llama_server_config", "context_shift"), True),
            ("--model.llama-server-config.tensor-split", "[3, 1]", ("llama_server_config", "tensor_split"), [3, 1]),
            (
                "--model.llama-server-config.override-tensor",
                "[exps=CPU]",
                ("llama_server_config", "override_tensor"),
                ["exps=CPU"],
            ),
            ("--model.diffusers-config.guidance-scale", "0.0", ("diffusers_config", "guidance_scale"), 0.0),
            (
                "--model.vllm-engine-kwargs.default-chat-template-kwargs",
                "{enable_thinking: false}",
                ("vllm_engine_kwargs", "default_chat_template_kwargs"),
                {"enable_thinking": False},
            ),
            ("--model.node-selector", "{gpu: big}", ("node_selector",), {"gpu": "big"}),
        ],
    )
    def test_values_are_read_as_yaml(self, flag, text, path, expected):
        node = _raw(*self._LLAMA, flag, text)
        for key in path:
            node = node[key]
        assert node == expected

    @pytest.mark.parametrize("value", ["on", "off", "auto"])
    def test_llama_server_reasoning_keeps_llamas_own_words(self, value):
        raw = _raw(*self._LLAMA, "--model.llama-server-config.reasoning", value)
        config = validate_models([raw]).models[0]
        assert config.llama_server_config is not None
        assert config.llama_server_config.reasoning == value

    def test_unparseable_yaml_value_is_rejected(self, capsys):
        err = _error(capsys, *self._VLLM, "--model.vllm-engine-kwargs.limit-mm-per-prompt", "{image: ")
        assert "not valid YAML" in err

    def test_every_schema_key_is_settable(self):
        """Derived from the schema independently of the generator: a new field is
        covered the moment it lands, and a dropped one fails here."""

        def paths(model: type[BaseModel], parent: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
            out: set[tuple[str, ...]] = set()
            for name, info in model.model_fields.items():
                out.add((*parent, name))
                block = _block_model(info.annotation)
                if block is not None:
                    out |= paths(block, (*parent, name))
            return out

        assert {arg.path for arg in GENERATED_MODEL_ARGS} == paths(ModelshipModelConfig) - {("model",)}

    def test_options_are_distinct(self):
        options = [arg.option for arg in GENERATED_MODEL_ARGS]
        assert len(options) == len(set(options))

    @pytest.mark.parametrize(
        "flag", ["--model.vllm-engine-kwargs.gpu-memory-utilization", "--model.vllm-engine-kwargs.model"]
    )
    def test_derived_keys_have_no_flag(self, flag, capsys):
        """They aren't schema fields, so the generator can't emit them either."""
        assert f"unrecognized model flag {flag}" in _error(capsys, *self._VLLM, flag, "0.8")

    def test_unknown_nested_flag_is_rejected(self, capsys):
        err = _error(capsys, *self._LLAMA, "--model.llama-server-config.n-ctxx", "8192")
        assert "unrecognized model flag --model.llama-server-config.n-ctxx" in err


class TestFlagLocations:
    _ERROR = "models.0.llama_server_config.n_ctx\n  Input should be a valid integer"

    def test_one_model_names_the_flag(self):
        assert flag_locations(self._ERROR, 1).startswith("--model.llama-server-config.n-ctx\n")

    def test_several_models_also_name_which_one(self):
        text = flag_locations("models.1.num_gpus\n  Input should be greater than or equal to 0", 2)
        assert text.startswith("--model.num-gpus (--model #2)\n")

    def test_any_depth(self):
        text = flag_locations("models.0.vllm_engine_kwargs.reasoning_config.reasoning_end_str\n  x", 1)
        assert text.startswith("--model.vllm-engine-kwargs.reasoning-config.reasoning-end-str\n")


class TestIntegrationCliRouting:
    """conftest routes half the single-model call sites through the flags and half
    through a models.yaml, so both surfaces stay covered."""

    def test_every_config_round_trips_through_the_flags(self):
        from tests.conftest import MODEL_CONFIGS, _model_flags

        for name, config in MODEL_CONFIGS.items():
            assert _args(*_model_flags(config)).models == [config], name

    def test_several_configs_round_trip_in_one_command(self):
        from tests.conftest import MODEL_CONFIGS, _model_flags

        configs = list(MODEL_CONFIGS.values())[:3]
        argv = [flag for config in configs for flag in _model_flags(config)]
        assert _args(*argv).models == configs

    def test_routing_splits_the_call_sites_in_half(self):
        from tests.conftest import CLI_ROUTED, MODEL_CONFIGS

        assert set(MODEL_CONFIGS) >= CLI_ROUTED
        assert len(CLI_ROUTED) == (len(MODEL_CONFIGS) + 1) // 2

    def test_routed_half_covers_nested_blocks_and_both_text_loaders(self):
        """A split that drifted to root-scalar-only entries would leave the nested
        flags with no integration coverage."""
        from tests.conftest import CLI_ROUTED, MODEL_CONFIGS

        routed = [MODEL_CONFIGS[name] for name in CLI_ROUTED]
        assert any(isinstance(value, dict) for config in routed for value in config.values())
        assert {config["loader"] for config in routed} >= {"vllm", "llama_server"}
