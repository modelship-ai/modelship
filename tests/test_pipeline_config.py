"""The `pipelines:` config: its schema, the `--pipeline` flags, reading it from a
models.yaml, and the deploy refusing pipelines until they can be deployed."""

import argparse

import pytest
from pydantic import BaseModel, ValidationError

from modelship import launcher
from modelship.deploy.config import parse_raw_models, parse_raw_pipelines, resolve_input_pipelines, validate_models
from modelship.utils.cli import parse_args
from modelship.utils.config_schema import (
    ModelshipConfig,
    PipelineConfig,
    parse_deployment_name,
    parse_gateway_app_name,
    parse_pipeline_app_name,
)
from modelship.utils.model_flags import GENERATED_PIPELINE_ARGS, _block_model

_QWEN = {"name": "qwen", "model": "org/qwen", "usecase": "generate", "loader": "vllm"}


def _args(command="deploy", *argv):
    return parse_args(command, list(argv))


def _error(capsys, *argv) -> str:
    with pytest.raises(SystemExit) as exc:
        _args("deploy", *argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


class TestSchema:
    def test_a_pipeline_names_its_models(self):
        pipeline = PipelineConfig(name="assistant", stt="whisper", llm="qwen", tts="kokoro")
        assert (pipeline.stt, pipeline.llm, pipeline.tts, pipeline.num_replicas) == ("whisper", "qwen", "kokoro", 1)

    @pytest.mark.parametrize("models", [{"llm": "qwen"}, {"stt": "whisper"}])
    def test_stt_or_llm_alone_is_enough(self, models):
        PipelineConfig(name="assistant", **models)

    def test_neither_stt_nor_llm_is_rejected(self):
        with pytest.raises(ValidationError, match="names no stt and no llm"):
            PipelineConfig(name="assistant", tts="kokoro")

    def test_session_defaults_take_openais_session_fields(self):
        defaults = PipelineConfig(
            name="assistant",
            llm="qwen",
            session_defaults={
                "instructions": "Be brief.",
                "output_modalities": ["text"],
                "max_output_tokens": "inf",
                "reasoning": {"effort": "low"},
            },
        ).session_defaults
        assert defaults is not None
        assert defaults.reasoning is not None
        assert (
            defaults.instructions,
            defaults.output_modalities,
            defaults.max_output_tokens,
            defaults.reasoning.effort,
        ) == ("Be brief.", ["text"], "inf", "low")

    @pytest.mark.parametrize(
        "session_defaults",
        [
            {"max_output_tokens": 0},
            {"output_modalities": ["video"]},
            {"output_modalities": []},
            {"output_modalities": ["text", "audio"]},
            {"voice": "af_heart"},
            {"turn_detection": None},
            {"reasoning": {"effort": "huge"}},
            {"audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}},
            {"audio": {"input": {"turn_detection": {"type": "server_vad", "idle_timeout_ms": 5000}}}},
            {"audio": {"input": {"turn_detection": {"type": "server_vad", "threshold": 1.5}}}},
            {"audio": {"input": {"format": {"type": "audio/pcm"}}}},
        ],
    )
    def test_session_defaults_reject_what_they_dont_hold(self, session_defaults):
        with pytest.raises(ValidationError):
            PipelineConfig(name="assistant", stt="whisper", llm="qwen", session_defaults=session_defaults)

    def test_turn_detection_defaults_take_server_vad_or_null(self):
        tuned = {"type": "server_vad", "silence_duration_ms": 800, "create_response": False}
        pipeline = PipelineConfig(
            name="assistant", stt="whisper", session_defaults={"audio": {"input": {"turn_detection": tuned}}}
        )
        assert pipeline.session_defaults is not None and pipeline.session_defaults.audio is not None
        given = pipeline.session_defaults.audio.input
        assert given is not None and given.turn_detection is not None
        assert (given.turn_detection.silence_duration_ms, given.turn_detection.threshold) == (800, 0.5)
        manual = PipelineConfig(
            name="assistant", stt="whisper", session_defaults={"audio": {"input": {"turn_detection": None}}}
        )
        assert manual.session_defaults is not None and manual.session_defaults.audio is not None
        assert manual.session_defaults.audio.input is not None
        assert "turn_detection" in manual.session_defaults.audio.input.model_fields_set

    def test_turn_detection_needs_an_stt_model(self):
        vad = {"audio": {"input": {"turn_detection": {"type": "server_vad"}}}}
        with pytest.raises(ValidationError, match="turn_detection needs an stt model"):
            PipelineConfig(name="assistant", llm="qwen", session_defaults=vad)

    def test_audio_output_needs_a_tts_model(self):
        with pytest.raises(ValidationError, match="asks for audio, but it names no tts model"):
            PipelineConfig(name="assistant", llm="qwen", session_defaults={"output_modalities": ["audio"]})
        spoken = PipelineConfig(
            name="assistant", llm="qwen", tts="kokoro", session_defaults={"output_modalities": ["audio"]}
        )
        assert spoken.session_defaults is not None
        assert spoken.session_defaults.output_modalities == ["audio"]

    @pytest.mark.parametrize("extra", [{"num_gpus": 1}, {"num_cpus": 1}, {"vad": "silero"}])
    def test_unknown_keys_are_rejected(self, extra):
        with pytest.raises(ValidationError, match="Extra inputs"):
            PipelineConfig(name="assistant", llm="qwen", **extra)

    def test_a_name_with_a_hash_is_rejected(self):
        with pytest.raises(ValidationError, match="must not contain '#'"):
            PipelineConfig(name="a#b", llm="qwen")

    def test_pipeline_names_are_unique(self):
        with pytest.raises(ValidationError, match="duplicate pipeline name 'assistant'"):
            ModelshipConfig(models=[], pipelines=[{"name": "assistant", "llm": "q"}, {"name": "assistant", "stt": "w"}])

    def test_pipelines_default_to_none(self):
        assert ModelshipConfig(models=[]).pipelines == []

    def test_a_pipeline_is_full_duplex_with_a_half_second_interruption_by_default(self):
        pipeline = PipelineConfig(name="voice", stt="whisper")
        assert (pipeline.duplex, pipeline.min_interruption_ms) == ("full", 500)
        half = PipelineConfig(name="voice", stt="whisper", duplex="half", min_interruption_ms=0)
        assert (half.duplex, half.min_interruption_ms) == ("half", 0)

    @pytest.mark.parametrize(
        "given", [{"duplex": "simplex"}, {"min_interruption_ms": -1}, {"min_interruption_ms": 5001}]
    )
    def test_echo_settings_out_of_range_are_rejected(self, given):
        with pytest.raises(ValidationError):
            PipelineConfig(name="voice", stt="whisper", **given)


class TestPipelineFlags:
    def test_each_pipeline_gets_the_flags_that_follow_it(self):
        args = _args(
            "deploy",
            "--pipeline", "assistant", "--pipeline.llm", "qwen", "--pipeline.num-replicas", "2",
            "--pipeline", "dictation", "--pipeline.stt", "whisper",
        )  # fmt: skip
        assert args.pipelines == [
            {"name": "assistant", "llm": "qwen", "num_replicas": 2},
            {"name": "dictation", "stt": "whisper"},
        ]
        assert args.models is None

    def test_models_and_pipelines_mix_in_one_command(self):
        args = _args(
            "deploy",
            "--model", "org/qwen", "--model.loader", "vllm",
            "--pipeline", "assistant", "--pipeline.llm", "qwen",
            "--model", "kokoro-en-v0_19", "--model.loader", "sherpa_onnx",
        )  # fmt: skip
        assert [m["name"] for m in args.models] == ["qwen", "kokoro-en-v0_19"]
        assert args.pipelines == [{"name": "assistant", "llm": "qwen"}]

    def test_start_takes_pipelines_too(self):
        assert _args("start", "--pipeline", "assistant", "--pipeline.llm", "qwen").pipelines == [
            {"name": "assistant", "llm": "qwen"}
        ]

    def test_a_block_flag_and_its_keys_merge(self):
        args = _args(
            "deploy",
            "--pipeline", "assistant", "--pipeline.llm", "qwen",
            "--pipeline.session-defaults.max-output-tokens", "256",
            "--pipeline.session-defaults", "{instructions: Be brief., max_output_tokens: 1024}",
        )  # fmt: skip
        assert args.pipelines[0]["session_defaults"] == {"instructions": "Be brief.", "max_output_tokens": 256}

    def test_a_model_flag_after_a_pipeline_is_rejected(self, capsys):
        err = _error(capsys, "--model", "org/qwen", "--pipeline", "assistant", "--model.num-gpus", "1")
        assert "--model.num-gpus sets a key of a model, but follows --pipeline NAME" in err

    def test_a_pipeline_flag_after_a_model_is_rejected(self, capsys):
        err = _error(capsys, "--pipeline", "assistant", "--model", "org/qwen", "--pipeline.llm", "qwen")
        assert "--pipeline.llm sets a key of a pipeline, but follows --model REF" in err

    def test_a_pipeline_flag_before_any_pipeline_is_rejected(self, capsys):
        assert "put it after one" in _error(capsys, "--pipeline.llm", "qwen")

    def test_an_unknown_pipeline_flag_is_rejected(self, capsys):
        err = _error(capsys, "--pipeline", "assistant", "--pipeline.vad", "silero")
        assert "unrecognized pipeline flag --pipeline.vad" in err

    def test_pipelines_and_config_are_mutually_exclusive(self, capsys):
        err = _error(capsys, "--config", "models.yaml", "--pipeline", "assistant", "--pipeline.llm", "qwen")
        assert "--pipeline and --config are mutually exclusive" in err

    def test_cancel_takes_no_pipeline(self, capsys):
        err = _error(capsys, "--cancel", "abc", "--pipeline", "assistant")
        assert "--cancel takes no model options: --pipeline" in err

    def test_every_pipeline_key_has_a_flag(self):
        def paths(model: type[BaseModel], parent: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
            out: set[tuple[str, ...]] = set()
            for name, info in model.model_fields.items():
                out.add((*parent, name))
                block = _block_model(info.annotation)
                if block is not None:
                    out |= paths(block, (*parent, name))
            return out

        assert {arg.path for arg in GENERATED_PIPELINE_ARGS} == paths(PipelineConfig) - {("name",)}


class TestConfigFile:
    def test_pipelines_are_read_beside_models(self):
        text = "models:\n  - {name: qwen, model: org/qwen, usecase: generate, loader: vllm}\npipelines:\n  - {name: a, llm: qwen}\n"
        assert parse_raw_models(text) == [_QWEN]
        assert parse_raw_pipelines(text) == [{"name": "a", "llm": "qwen"}]

    def test_an_unknown_top_level_key_is_rejected(self):
        with pytest.raises(ValueError, match="unknown top-level key 'pipeline'"):
            parse_raw_models("models: []\npipeline: []\n")

    def test_unknown_keys_of_mixed_types_are_reported(self):
        with pytest.raises(ValueError, match="unknown top-level key 1"):
            parse_raw_models("1: []\nother: []\nmodels: []\n")

    def test_pipelines_must_be_a_list(self):
        with pytest.raises(ValueError, match="'pipelines' must be a list"):
            parse_raw_pipelines("models: []\npipelines: {name: a}\n")

    def test_a_file_without_pipelines_asks_for_none(self, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text("models: []\n")
        assert resolve_input_pipelines(_args("deploy", "--config", str(config))) is None

    def test_validate_takes_both(self):
        config = validate_models([_QWEN], [{"name": "assistant", "llm": "qwen"}])
        assert [p.name for p in config.pipelines] == ["assistant"]


class TestLauncherValidation:
    def test_a_cli_pipeline_alone_validates_and_asks_for_no_models(self):
        assert launcher._validate_config(_args("deploy", "--pipeline", "assistant", "--pipeline.llm", "qwen")) is None

    def test_a_file_pipeline_validates_with_its_models(self, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text(
            "models:\n  - {name: qwen, model: org/qwen, usecase: generate, loader: vllm}\n"
            "pipelines:\n  - {name: assistant, llm: qwen}\n"
        )
        validated = launcher._validate_config(_args("start", "--config", str(config)))
        assert validated is not None
        assert [p.name for p in validated.pipelines] == ["assistant"]

    def test_a_pipeline_named_like_a_model_is_refused(self, tmp_path, capsys):
        config = tmp_path / "models.yaml"
        config.write_text(
            "models:\n  - {name: qwen, model: org/qwen, usecase: generate, loader: vllm}\n"
            "pipelines:\n  - {name: qwen, llm: qwen}\n"
        )
        with pytest.raises(SystemExit) as exc:
            launcher._validate_config(_args("start", "--config", str(config)))
        assert exc.value.code == 1
        assert "pipeline 'qwen' has the name of a model" in capsys.readouterr().err

    def test_an_invalid_cli_pipeline_names_its_flag(self, capsys):
        with pytest.raises(SystemExit):
            launcher._validate_config(
                _args("deploy", "--pipeline", "assistant", "--pipeline.llm", "q", "--pipeline.num-replicas", "0")
            )
        err = capsys.readouterr().err
        assert "--pipeline.num-replicas" in err
        assert "pipelines.0." not in err

    def test_models_alone_still_validate(self):
        config = launcher._validate_config(
            argparse.Namespace(config=None, models=[_QWEN], pipelines=None, config_from_job=False)
        )
        assert config is not None
        assert [m.name for m in config.models] == ["qwen"]


class TestAudioAppName:
    def test_it_reads_back_as_a_pipeline_and_never_as_a_model(self):
        name = PipelineConfig(name="voice", stt="whisper").deployment_name("modelship")
        assert name.startswith("modelship.voice-p")
        assert parse_pipeline_app_name(name) == ("modelship", "voice")
        assert parse_deployment_name(name) is None
        assert parse_gateway_app_name(name) == ("modelship", "voice")

    def test_a_model_app_never_reads_as_a_pipeline(self):
        assert parse_pipeline_app_name("modelship.qwen-p-3f9a0c12de") is None
        assert parse_pipeline_app_name("modelship.qwen-3f9a0c12de") is None

    def test_settings_outside_the_audio_app_keep_its_name(self):
        base = PipelineConfig(name="voice", stt="whisper")
        changed = PipelineConfig(
            name="voice", stt="other", llm="qwen", num_replicas=3, duplex="half", min_interruption_ms=800
        )
        assert base.deployment_name("g") == changed.deployment_name("g")

    def test_only_a_pipeline_with_stt_has_an_audio_app(self):
        assert PipelineConfig(name="voice", stt="whisper").has_audio_app()
        assert not PipelineConfig(name="chat", llm="qwen").has_audio_app()
