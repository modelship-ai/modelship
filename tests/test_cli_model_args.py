"""The `--model` flags must produce a models.yaml entry indistinguishable from the
hand-written one."""

import subprocess
import sys
from typing import get_args
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

from modelship.deploy.config import resolve_input_models, validate_models
from modelship.utils.cli import MODEL_ARG_KEYS, infer_model_name, model_from_args, parse_args
from modelship.utils.config_schema import ModelshipModelConfig
from modelship.utils.model_flags import GENERATED_MODEL_ARGS


def _args(*argv):
    return parse_args("deploy", list(argv))


def _raw(*argv):
    return model_from_args(_args(*argv))


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
            "--model",
            "lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf",
            "--name",
            "qwen",
            "--usecase",
            "generate",
            "--loader",
            "llama_server",
            "--num-cpus",
            "3",
        )
        assert cli_entry == yaml_entry

        from_yaml = validate_models([yaml_entry]).models[0]
        from_cli = validate_models([cli_entry]).models[0]
        assert from_cli.fingerprint() == from_yaml.fingerprint()
        assert from_cli.model_fields_set == from_yaml.model_fields_set

    def test_unset_flags_are_omitted_not_defaulted(self):
        raw = _raw("--model", "Qwen/Qwen3-8B", "--usecase", "generate", "--loader", "vllm")
        assert raw == {"name": "qwen3-8b", "model": "Qwen/Qwen3-8B", "usecase": "generate", "loader": "vllm"}
        assert validate_models([raw]).models[0].model_fields_set == {"name", "model", "usecase", "loader"}

    def test_num_replicas_left_unset_does_not_block_autoscaling(self):
        """The check rejects num_replicas alongside autoscaling_config via
        model_fields_set — a materialized CLI default would trip it."""
        raw = _raw("--model", "Qwen/Qwen3-8B", "--usecase", "generate", "--loader", "vllm")
        merged = {**raw, "autoscaling_config": {"min_replicas": 1, "max_replicas": 3}}
        validate_models([merged])  # no raise

    def test_image_loader_still_defaults_usecase(self):
        raw = _raw("--model", "stabilityai/sdxl-turbo", "--loader", "diffusers")
        assert "usecase" not in raw
        assert validate_models([raw]).models[0].usecase.value == "image"

    def test_schema_rejects_missing_loader(self):
        with pytest.raises(ValidationError, match="loader"):
            validate_models([_raw("--model", "Qwen/Qwen3-8B", "--usecase", "generate")])

    def test_model_arg_keys_covers_every_flag_the_builder_emits(self):
        raw = _raw(
            "--model", "Qwen/Qwen3-8B",
            "--name", "n",
            "--usecase", "generate",
            "--loader", "vllm",
            "--num-gpus", "0.5",
            "--num-cpus", "2",
            "--num-replicas", "2",
            "--max-ongoing-requests", "8",
        )  # fmt: skip
        assert set(raw) == set(MODEL_ARG_KEYS)


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
        assert _raw("--model", "Qwen/Qwen3-8B", "--name", "custom")["name"] == "custom"

    def test_uninferrable_ref_raises(self):
        with pytest.raises(ValueError, match="pass --name"):
            infer_model_name("///")

    def test_same_ref_infers_the_same_name(self):
        """Re-running a deploy must be idempotent: the name is the additive
        merge's replace-by-name key."""
        ref = "lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf"
        assert infer_model_name(ref) == infer_model_name(ref)


class TestResolveInputModels:
    def test_model_flag_wins_over_the_default_config_file(self, tmp_path):
        default = tmp_path / "models.yaml"
        default.write_text("models:\n  - name: from-file\n    model: x.gguf\n    loader: llama_server\n")
        with patch("modelship.deploy.config.default_config_path", return_value=default):
            raw = resolve_input_models(_args("--model", "Qwen/Qwen3-8B", "--loader", "vllm", "--usecase", "generate"))
        assert raw is not None
        assert [m["name"] for m in raw] == ["qwen3-8b"]

    def test_config_file_is_read_when_no_model_flag(self, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text("models:\n  - name: m\n    model: x.gguf\n    loader: llama_server\n    usecase: generate\n")
        raw = resolve_input_models(_args("--config", str(config)))
        assert raw == [{"name": "m", "model": "x.gguf", "loader": "llama_server", "usecase": "generate"}]

    def test_neither_returns_none(self, tmp_path):
        with patch("modelship.deploy.config.default_config_path", return_value=tmp_path / "nope.yaml"):
            assert resolve_input_models(_args()) is None

    def test_both_is_rejected_at_parse_time(self, tmp_path, capsys):
        config = tmp_path / "models.yaml"
        config.write_text("models: []\n")
        with pytest.raises(SystemExit) as exc:
            _args("--model", "Qwen/Qwen3-8B", "--config", str(config))
        assert exc.value.code == 2
        assert "mutually exclusive" in capsys.readouterr().err

    def test_both_is_rejected_for_a_hand_built_namespace(self, tmp_path):
        args = _args("--config", str(tmp_path / "models.yaml"))
        args.model = "Qwen/Qwen3-8B"
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_input_models(args)


class TestFailsBeforeRay:
    def test_bad_model_flag_exits_without_importing_ray(self):
        code = (
            "import sys; from modelship.launcher import _cmd_run\n"
            "try:\n"
            "    _cmd_run('start', ['--model', 'Qwen/Qwen3-8B'])\n"
            "except SystemExit as e:\n"
            "    print('exit', e.code, 'ray' in sys.modules)\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "exit 1 False"


class TestNestedFlags:
    """The generated flags — one per nested-block key, named for its config path,
    valued as YAML text."""

    _LLAMA = ("--model", "x.gguf", "--usecase", "generate", "--loader", "llama_server")
    _VLLM = ("--model", "org/qwen", "--name", "qwen", "--usecase", "generate", "--loader", "vllm")

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
            "--num-gpus", "2",
            "--vllm-engine-kwargs.max-model-len", "8192",
            "--vllm-engine-kwargs.enforce-eager", "true",
            "--vllm-engine-kwargs.limit-mm-per-prompt", "{image: 2}",
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
        raw = _raw(*self._LLAMA, "--llama-server-config.threads", "null")
        assert raw["llama_server_config"] == {"threads": None}
        config = validate_models([raw]).models[0]
        assert config.llama_server_config is not None
        assert "threads" in config.llama_server_config.model_fields_set

    @pytest.mark.parametrize(
        ("flag", "text", "path", "expected"),
        [
            ("--llama-server-config.n-ctx", "8192", ("llama_server_config", "n_ctx"), 8192),
            ("--llama-server-config.context-shift", "true", ("llama_server_config", "context_shift"), True),
            (
                "--llama-server-config.tensor-split",
                "[3, 1]",
                ("llama_server_config", "tensor_split"),
                [3.0, 1.0],
            ),
            ("--diffusers-config.guidance-scale", "0.0", ("diffusers_config", "guidance_scale"), 0.0),
            (
                "--chat-template-kwargs",
                "{enable_thinking: false}",
                ("chat_template_kwargs",),
                {"enable_thinking": False},
            ),
        ],
    )
    def test_values_are_read_as_yaml(self, flag, text, path, expected):
        node = _raw(*self._LLAMA, flag, text)
        for key in path:
            node = node[key]
        assert node == expected

    def test_unparseable_yaml_value_is_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _args(*self._VLLM, "--vllm-engine-kwargs.limit-mm-per-prompt", "{image: ")
        assert exc.value.code == 2
        assert "not valid YAML" in capsys.readouterr().err

    def test_every_schema_field_is_settable(self):
        """Derived from the schema independently of the generator: a new field is
        covered the moment it lands, and a dropped one fails here."""
        expected: set[tuple[str, ...]] = set()
        for name, field in ModelshipModelConfig.model_fields.items():
            block = next(
                (
                    candidate
                    for candidate in (field.annotation, *get_args(field.annotation))
                    if isinstance(candidate, type) and issubclass(candidate, BaseModel)
                ),
                None,
            )
            expected |= {(name, sub) for sub in block.model_fields} if block else {(name,)}

        settable = {arg.path for arg in GENERATED_MODEL_ARGS} | {(key,) for key in MODEL_ARG_KEYS}
        assert settable == expected

    def test_generated_options_are_distinct_from_the_root_flags(self):
        options = [arg.option for arg in GENERATED_MODEL_ARGS]
        assert len(options) == len(set(options))
        assert not set(options) & {f"--{key.replace('_', '-')}" for key in MODEL_ARG_KEYS}

    @pytest.mark.parametrize("flag", ["--vllm-engine-kwargs.gpu-memory-utilization", "--vllm-engine-kwargs.model"])
    def test_derived_keys_have_no_flag(self, flag, capsys):
        """They aren't schema fields, so the generator can't emit them either."""
        with pytest.raises(SystemExit) as exc:
            _args(*self._VLLM, flag, "0.8")
        assert exc.value.code == 2
        assert "unrecognized arguments" in capsys.readouterr().err

    def test_tuning_flag_without_model_is_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _args("--llama-server-config.n-ctx", "8192")
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "--llama-server-config.n-ctx" in err
        assert "pass --model" in err

    def test_tuning_flag_with_config_says_to_edit_the_file(self, capsys):
        """--config is already what they passed, so pointing back at it doesn't help."""
        with pytest.raises(SystemExit) as exc:
            _args("--config", "models.yaml", "--llama-server-config.n-ctx", "8192")
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "--llama-server-config.n-ctx" in err
        assert "entry in the config file" in err
        assert "pass --model" not in err

    def test_the_error_names_the_flags_that_triggered_it(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _args(
                "--llama-server-config.n-ctx",
                "8192",
                "--llama-server-config.parallel",
                "2",
                "--autoscaling-config.max-replicas",
                "3",
                "--chat-template-kwargs",
                "{}",
            )
        assert exc.value.code == 2
        err = capsys.readouterr().err
        # Capped at three, so a wide invocation doesn't dump every flag it set.
        assert err.count("--") >= 3
        assert ", ..." in err

    def test_unknown_nested_flag_is_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _args(*self._LLAMA, "--llama-server-config.n-ctxx", "8192")
        assert exc.value.code == 2
        assert "unrecognized arguments" in capsys.readouterr().err


class TestIntegrationCliRouting:
    """conftest routes half the single-model call sites through the flags and half
    through a models.yaml, so both surfaces stay covered."""

    def test_every_config_round_trips_through_the_flags(self):
        from tests.conftest import MODEL_CONFIGS, _model_flags

        for name, config in MODEL_CONFIGS.items():
            assert model_from_args(_args(*_model_flags(config))) == config, name

    def test_routing_splits_the_call_sites_in_half(self):
        from tests.conftest import CLI_ROUTED, MODEL_CONFIGS

        assert set(MODEL_CONFIGS) >= CLI_ROUTED
        assert len(CLI_ROUTED) == (len(MODEL_CONFIGS) + 1) // 2

    def test_routed_half_covers_nested_blocks_and_both_text_loaders(self):
        """A split that drifted to root-scalar-only entries would leave the nested
        flags with no integration coverage."""
        from tests.conftest import CLI_ROUTED, MODEL_CONFIGS

        routed = [MODEL_CONFIGS[name] for name in CLI_ROUTED]
        assert any(set(config) - set(MODEL_ARG_KEYS) for config in routed)
        assert {config["loader"] for config in routed} >= {"vllm", "llama_server"}
