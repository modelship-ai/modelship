"""The `pipelines:` config: its schema per pipeline type, the `--pipeline` flags, reading it from a models.yaml, and
the app name each pipeline deploys under."""

import argparse

import pytest
from pydantic import BaseModel, ValidationError

from modelship import launcher
from modelship.deploy.actor_options import build_pipeline_deployment_options
from modelship.deploy.config import parse_raw_models, parse_raw_pipelines, resolve_input_pipelines, validate_models
from modelship.utils.cli import parse_args
from modelship.utils.config_schema import (
    AssistantPipelineConfig,
    ModelshipConfig,
    ModelUsecase,
    parse_deployment_name,
    parse_gateway_app_name,
    parse_pipeline,
    parse_pipeline_app_name,
)
from modelship.utils.model_flags import GENERATED_PIPELINE_ARGS, _block_model

_QWEN = {"name": "qwen", "model": "org/qwen", "usecase": "generate", "loader": "vllm"}

# what every pipeline type has at its top level
_SHARED = {"name", "num_replicas", "node_selector"}


def _assistant(**fields) -> AssistantPipelineConfig:
    """An assistant from flat *fields*: the shared ones at the top, the rest in its assistant_config."""
    top = {key: value for key, value in fields.items() if key in _SHARED}
    own = {key: value for key, value in fields.items() if key not in _SHARED}
    return AssistantPipelineConfig(type="assistant", assistant_config=own, **top)


def _args(command="deploy", *argv):
    return parse_args(command, list(argv))


def _error(capsys, *argv) -> str:
    with pytest.raises(SystemExit) as exc:
        _args("deploy", *argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


class TestTypes:
    def test_an_assistant_serves_realtime(self):
        pipeline = parse_pipeline({"name": "voice", "type": "assistant", "assistant_config": {"stt": "whisper"}})
        assert isinstance(pipeline, AssistantPipelineConfig)
        assert pipeline.USECASE is ModelUsecase.realtime

    def test_a_pipeline_names_its_type(self):
        with pytest.raises(ValidationError, match="type"):
            parse_pipeline({"name": "voice", "assistant_config": {"stt": "whisper"}})

    def test_an_unknown_type_is_rejected(self):
        with pytest.raises(ValidationError, match="type"):
            parse_pipeline({"name": "voice", "type": "router", "assistant_config": {"stt": "whisper"}})

    def test_a_model_cannot_take_the_realtime_usecase(self):
        with pytest.raises(ValidationError, match="no loader serves usecase 'realtime'"):
            ModelshipConfig(models=[{**_QWEN, "usecase": "realtime"}])

    def test_an_assistant_lists_the_models_it_calls(self):
        assert _assistant(name="a", stt="whisper", tts="kokoro", llm="qwen").models() == ["whisper", "qwen", "kokoro"]
        assert _assistant(name="a", llm="qwen").models() == ["qwen"]


class TestAssistantSchema:
    def test_a_pipeline_names_its_models_in_its_assistant_config(self):
        pipeline = _assistant(name="assistant", stt="whisper", llm="qwen", tts="kokoro")
        own = pipeline.assistant_config
        assert (own.stt, own.llm, own.tts, pipeline.num_replicas) == ("whisper", "qwen", "kokoro", 1)

    @pytest.mark.parametrize("key", ["stt", "llm", "tts", "session_defaults", "duplex", "min_interruption_ms"])
    def test_the_assistants_settings_at_the_top_level_are_rejected(self, key):
        with pytest.raises(ValidationError, match="Extra inputs"):
            parse_pipeline({"name": "a", "type": "assistant", "assistant_config": {"llm": "qwen"}, key: "x"})

    def test_an_assistant_needs_its_assistant_config(self):
        with pytest.raises(ValidationError, match="assistant_config"):
            parse_pipeline({"name": "a", "type": "assistant"})

    @pytest.mark.parametrize("models", [{"llm": "qwen"}, {"stt": "whisper"}])
    def test_stt_or_llm_alone_is_enough(self, models):
        _assistant(name="assistant", **models)

    def test_neither_stt_nor_llm_is_rejected(self):
        with pytest.raises(ValidationError, match="names no stt and no llm"):
            _assistant(name="assistant", tts="kokoro")

    def test_session_defaults_take_openais_session_fields(self):
        defaults = _assistant(
            name="assistant",
            llm="qwen",
            session_defaults={
                "instructions": "Be brief.",
                "output_modalities": ["text"],
                "max_output_tokens": "inf",
                "reasoning": {"effort": "low"},
            },
        ).assistant_config.session_defaults
        assert defaults is not None
        assert defaults.reasoning is not None
        assert (
            defaults.instructions,
            defaults.output_modalities,
            defaults.max_output_tokens,
            defaults.reasoning.effort,
        ) == ("Be brief.", ["text"], "inf", "low")

    def test_session_defaults_take_a_reasoning_effort_of_none(self):
        defaults = _assistant(
            name="assistant", llm="qwen", session_defaults={"reasoning": {"effort": "none"}}
        ).assistant_config.session_defaults
        assert defaults is not None and defaults.reasoning is not None
        assert defaults.reasoning.effort == "none"

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
            {"reasoning": {"effort": "max"}},
            {"audio": {"input": {"turn_detection": {"type": "semantic_vad", "silence_duration_ms": 800}}}},
            {"audio": {"input": {"turn_detection": {"type": "semantic_vad", "eagerness": "eager"}}}},
            {"audio": {"input": {"turn_detection": {"type": "server_vad", "eagerness": "high"}}}},
            {"audio": {"input": {"turn_detection": {"type": "server_vad", "idle_timeout_ms": 5000}}}},
            {"audio": {"input": {"turn_detection": {"type": "server_vad", "threshold": 1.5}}}},
            {"audio": {"input": {"format": {"type": "audio/pcm"}}}},
        ],
    )
    def test_session_defaults_reject_what_they_dont_hold(self, session_defaults):
        with pytest.raises(ValidationError):
            _assistant(name="assistant", stt="whisper", llm="qwen", session_defaults=session_defaults)

    def test_turn_detection_defaults_take_server_vad_or_null(self):
        tuned = {"type": "server_vad", "silence_duration_ms": 800, "create_response": False}
        pipeline = _assistant(
            name="assistant", stt="whisper", session_defaults={"audio": {"input": {"turn_detection": tuned}}}
        )
        defaults = pipeline.assistant_config.session_defaults
        assert defaults is not None and defaults.audio is not None
        given = defaults.audio.input
        assert given is not None and given.turn_detection is not None
        assert (given.turn_detection.silence_duration_ms, given.turn_detection.threshold) == (800, 0.5)
        manual = _assistant(
            name="assistant", stt="whisper", session_defaults={"audio": {"input": {"turn_detection": None}}}
        ).assistant_config.session_defaults
        assert manual is not None and manual.audio is not None
        assert manual.audio.input is not None
        assert "turn_detection" in manual.audio.input.model_fields_set

    def test_turn_detection_defaults_take_semantic_vad(self):
        semantic = {"type": "semantic_vad", "eagerness": "low", "interrupt_response": False}
        pipeline = _assistant(
            name="assistant", stt="whisper", session_defaults={"audio": {"input": {"turn_detection": semantic}}}
        )
        defaults = pipeline.assistant_config.session_defaults
        assert defaults is not None and defaults.audio is not None and defaults.audio.input is not None
        given = defaults.audio.input.turn_detection
        assert given is not None and given.type == "semantic_vad"
        assert (given.eagerness, given.create_response, given.interrupt_response) == ("low", True, False)

    def test_turn_detection_needs_an_stt_model(self):
        vad = {"audio": {"input": {"turn_detection": {"type": "server_vad"}}}}
        with pytest.raises(ValidationError, match="turn_detection needs an stt model"):
            _assistant(name="assistant", llm="qwen", session_defaults=vad)

    def test_audio_output_needs_a_tts_model(self):
        with pytest.raises(ValidationError, match="asks for audio, but it names no tts model"):
            _assistant(name="assistant", llm="qwen", session_defaults={"output_modalities": ["audio"]})
        spoken = _assistant(
            name="assistant", llm="qwen", tts="kokoro", session_defaults={"output_modalities": ["audio"]}
        ).assistant_config.session_defaults
        assert spoken is not None
        assert spoken.output_modalities == ["audio"]

    @pytest.mark.parametrize("extra", [{"num_gpus": 1}, {"num_cpus": 1}, {"vad": "silero"}])
    def test_unknown_keys_are_rejected(self, extra):
        with pytest.raises(ValidationError, match="Extra inputs"):
            _assistant(name="assistant", llm="qwen", **extra)
        with pytest.raises(ValidationError, match="Extra inputs"):
            parse_pipeline({"name": "a", "type": "assistant", "assistant_config": {"llm": "qwen"}, **extra})

    def test_a_name_with_a_hash_is_rejected(self):
        with pytest.raises(ValidationError, match="must not contain '#'"):
            _assistant(name="a#b", llm="qwen")

    def test_pipeline_names_are_unique(self):
        with pytest.raises(ValidationError, match="duplicate pipeline name 'assistant'"):
            ModelshipConfig(
                models=[],
                pipelines=[
                    {"name": "assistant", "type": "assistant", "assistant_config": {"llm": "q"}},
                    {"name": "assistant", "type": "assistant", "assistant_config": {"stt": "w"}},
                ],
            )

    def test_pipelines_default_to_none(self):
        assert ModelshipConfig(models=[]).pipelines == []

    def test_a_pipeline_is_full_duplex_with_a_half_second_interruption_by_default(self):
        own = _assistant(name="voice", stt="whisper").assistant_config
        assert (own.duplex, own.min_interruption_ms) == ("full", 500)
        half = _assistant(name="voice", stt="whisper", duplex="half", min_interruption_ms=0).assistant_config
        assert (half.duplex, half.min_interruption_ms) == ("half", 0)

    @pytest.mark.parametrize(
        "given", [{"duplex": "simplex"}, {"min_interruption_ms": -1}, {"min_interruption_ms": 5001}]
    )
    def test_echo_settings_out_of_range_are_rejected(self, given):
        with pytest.raises(ValidationError):
            _assistant(name="voice", stt="whisper", **given)

    def test_a_pipeline_has_no_wake_words_by_default(self):
        assert _assistant(name="voice", stt="whisper").assistant_config.wake is None

    def test_wake_words_are_built_in_names_or_manifests_each_with_an_optional_cutoff(self):
        words = ["okay_nabu", {"model": "hey_jarvis", "cutoff": 0.9}, "/models/hey_skippy.json"]
        wake = _assistant(name="voice", stt="whisper", wake={"words": words}).assistant_config.wake
        assert wake is not None
        assert [(word.model, word.cutoff) for word in wake.words] == [
            ("okay_nabu", None),
            ("hey_jarvis", 0.9),
            ("/models/hey_skippy.json", None),
        ]
        assert (wake.listen_s, wake.follow_up.mode, wake.follow_up.listen_s) == (10.0, "off", 10.0)

    @pytest.mark.parametrize(
        ("wake", "problem"),
        [
            ({"words": ["hey_siri"]}, "neither a built-in word"),
            ({"words": ["/models/word.tflite"]}, "neither a built-in word"),
            ({"words": []}, "at least 1 item"),
            ({"words": [{"model": "hey_jarvis", "cutoff": 1.0}]}, "less than 1"),
            ({"words": ["hey_jarvis"], "listen_s": 0}, "greater than 0"),
            ({"words": ["hey_jarvis"], "follow_up": {"mode": "sometimes"}}, "follow_up.mode"),
            ({"words": ["hey_jarvis"], "keywords": ["x"]}, "Extra inputs"),
        ],
    )
    def test_wake_settings_it_cant_use_are_rejected(self, wake, problem):
        with pytest.raises(ValidationError, match=problem):
            _assistant(name="voice", stt="whisper", wake=wake)

    def test_wake_words_need_an_stt_model(self):
        with pytest.raises(ValidationError, match="wake needs an stt model"):
            _assistant(name="voice", llm="qwen", wake={"words": ["hey_jarvis"]})


class TestPipelineFlags:
    def test_each_pipeline_gets_the_flags_that_follow_it(self):
        args = _args(
            "deploy",
            "--pipeline", "assistant", "--pipeline.type", "assistant", "--pipeline.assistant-config.llm", "qwen",
            "--pipeline.num-replicas", "2",
            "--pipeline", "dictation", "--pipeline.type", "assistant", "--pipeline.assistant-config.stt", "whisper",
        )  # fmt: skip
        assert args.pipelines == [
            {"name": "assistant", "type": "assistant", "assistant_config": {"llm": "qwen"}, "num_replicas": 2},
            {"name": "dictation", "type": "assistant", "assistant_config": {"stt": "whisper"}},
        ]
        assert args.models is None

    def test_models_and_pipelines_mix_in_one_command(self):
        args = _args(
            "deploy",
            "--model", "org/qwen", "--model.loader", "vllm",
            "--pipeline", "assistant", "--pipeline.type", "assistant", "--pipeline.assistant-config.llm", "qwen",
            "--model", "kokoro-en-v0_19", "--model.loader", "sherpa_onnx",
        )  # fmt: skip
        assert [m["name"] for m in args.models] == ["qwen", "kokoro-en-v0_19"]
        assert args.pipelines == [{"name": "assistant", "type": "assistant", "assistant_config": {"llm": "qwen"}}]

    def test_a_pipeline_takes_a_node_selector_flag(self):
        args = _args(
            "deploy",
            "--pipeline", "assistant", "--pipeline.type", "assistant", "--pipeline.assistant-config.llm", "qwen",
            "--pipeline.node-selector", "{modelship/hostname: head}",
        )  # fmt: skip
        assert args.pipelines[0]["node_selector"] == {"modelship/hostname": "head"}

    def test_start_takes_pipelines_too(self):
        args = _args(
            "start",
            "--pipeline", "assistant", "--pipeline.type", "assistant", "--pipeline.assistant-config.llm", "qwen",
        )  # fmt: skip
        assert args.pipelines == [{"name": "assistant", "type": "assistant", "assistant_config": {"llm": "qwen"}}]

    def test_a_block_flag_and_its_keys_merge(self):
        args = _args(
            "deploy",
            "--pipeline", "assistant", "--pipeline.type", "assistant",
            "--pipeline.assistant-config", "{llm: qwen}",
            "--pipeline.assistant-config.session-defaults.max-output-tokens", "256",
            "--pipeline.assistant-config.session-defaults", "{instructions: Be brief., max_output_tokens: 1024}",
        )  # fmt: skip
        assert args.pipelines[0]["assistant_config"] == {
            "llm": "qwen",
            "session_defaults": {"instructions": "Be brief.", "max_output_tokens": 256},
        }

    def test_wake_words_are_set_by_flags(self):
        args = _args(
            "deploy",
            "--pipeline", "voice", "--pipeline.type", "assistant", "--pipeline.assistant-config.stt", "whisper",
            "--pipeline.assistant-config.wake.words", "[okay_nabu, {model: hey_jarvis, cutoff: 0.9}]",
            "--pipeline.assistant-config.wake.follow-up.mode", "question",
        )  # fmt: skip
        wake = parse_pipeline(args.pipelines[0]).assistant_config.wake
        assert wake is not None
        assert [word.model for word in wake.words] == ["okay_nabu", "hey_jarvis"]
        assert wake.follow_up.mode == "question"

    def test_a_model_flag_after_a_pipeline_is_rejected(self, capsys):
        err = _error(capsys, "--model", "org/qwen", "--pipeline", "assistant", "--model.num-gpus", "1")
        assert "--model.num-gpus sets a key of a model, but follows --pipeline NAME" in err

    def test_a_pipeline_flag_after_a_model_is_rejected(self, capsys):
        err = _error(
            capsys, "--pipeline", "assistant", "--model", "org/qwen", "--pipeline.assistant-config.llm", "qwen"
        )
        assert "--pipeline.assistant-config.llm sets a key of a pipeline, but follows --model REF" in err

    def test_a_pipeline_flag_before_any_pipeline_is_rejected(self, capsys):
        assert "put it after one" in _error(capsys, "--pipeline.assistant-config.llm", "qwen")

    @pytest.mark.parametrize("flag", ["--pipeline.vad", "--pipeline.llm"])
    def test_an_unknown_pipeline_flag_is_rejected(self, capsys, flag):
        err = _error(capsys, "--pipeline", "assistant", flag, "silero")
        assert f"unrecognized pipeline flag {flag}" in err

    def test_pipelines_and_config_are_mutually_exclusive(self, capsys):
        err = _error(
            capsys, "--config", "models.yaml", "--pipeline", "assistant", "--pipeline.assistant-config.llm", "qwen"
        )
        assert "--pipeline and --config are mutually exclusive" in err

    def test_cancel_takes_no_pipeline(self, capsys):
        err = _error(capsys, "--cancel", "abc", "--pipeline", "assistant")
        assert "--cancel takes no model options: --pipeline" in err

    def test_every_key_of_every_pipeline_type_has_a_flag(self):
        def paths(model: type[BaseModel], parent: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
            out: set[tuple[str, ...]] = set()
            for name, info in model.model_fields.items():
                out.add((*parent, name))
                block = _block_model(info.annotation)
                if block is not None:
                    out |= paths(block, (*parent, name))
            return out

        assert {arg.path for arg in GENERATED_PIPELINE_ARGS} == paths(AssistantPipelineConfig) - {("name",)}


class TestConfigFile:
    def test_pipelines_are_read_beside_models(self):
        text = (
            "models:\n  - {name: qwen, model: org/qwen, usecase: generate, loader: vllm}\n"
            "pipelines:\n  - {name: a, type: assistant, assistant_config: {llm: qwen}}\n"
        )
        assert parse_raw_models(text) == [_QWEN]
        assert parse_raw_pipelines(text) == [{"name": "a", "type": "assistant", "assistant_config": {"llm": "qwen"}}]

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
        pipeline = {"name": "assistant", "type": "assistant", "assistant_config": {"llm": "qwen"}}
        config = validate_models([_QWEN], [pipeline])
        assert [p.name for p in config.pipelines] == ["assistant"]


class TestLauncherValidation:
    def test_a_cli_pipeline_alone_validates_and_asks_for_no_models(self):
        args = _args(
            "deploy", "--pipeline", "assistant", "--pipeline.type", "assistant", "--pipeline.assistant-config.llm", "q"
        )
        assert launcher._validate_config(args) is None

    def test_a_file_pipeline_validates_with_its_models(self, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text(
            "models:\n  - {name: qwen, model: org/qwen, usecase: generate, loader: vllm}\n"
            "pipelines:\n  - {name: assistant, type: assistant, assistant_config: {llm: qwen}}\n"
        )
        validated = launcher._validate_config(_args("start", "--config", str(config)))
        assert validated is not None
        assert [p.name for p in validated.pipelines] == ["assistant"]

    def test_a_pipeline_named_like_a_model_is_refused(self, tmp_path, capsys):
        config = tmp_path / "models.yaml"
        config.write_text(
            "models:\n  - {name: qwen, model: org/qwen, usecase: generate, loader: vllm}\n"
            "pipelines:\n  - {name: qwen, type: assistant, assistant_config: {llm: qwen}}\n"
        )
        with pytest.raises(SystemExit) as exc:
            launcher._validate_config(_args("start", "--config", str(config)))
        assert exc.value.code == 1
        assert "pipeline 'qwen' has the name of a model" in capsys.readouterr().err

    def test_an_invalid_cli_pipeline_names_its_flag(self, capsys):
        args = _args(
            "deploy",
            "--pipeline", "assistant", "--pipeline.type", "assistant", "--pipeline.assistant-config.llm", "q",
            "--pipeline.assistant-config.min-interruption-ms", "-1",
        )  # fmt: skip
        with pytest.raises(SystemExit):
            launcher._validate_config(args)
        err = capsys.readouterr().err
        assert "--pipeline.assistant-config.min-interruption-ms" in err
        assert "pipelines.0." not in err

    def test_a_cli_pipeline_without_a_type_names_the_flag(self, capsys):
        with pytest.raises(SystemExit):
            launcher._validate_config(
                _args("deploy", "--pipeline", "assistant", "--pipeline.assistant-config.llm", "q")
            )
        assert "--pipeline.type" in capsys.readouterr().err

    def test_models_alone_still_validate(self):
        config = launcher._validate_config(
            argparse.Namespace(config=None, models=[_QWEN], pipelines=None, config_from_job=False)
        )
        assert config is not None
        assert [m.name for m in config.models] == ["qwen"]


class TestAppName:
    def test_it_reads_back_as_a_pipeline_and_never_as_a_model(self):
        name = _assistant(name="voice", stt="whisper").deployment_name("modelship")
        assert name.startswith("modelship.voice-p")
        assert parse_pipeline_app_name(name) == ("modelship", "voice")
        assert parse_deployment_name(name) is None
        assert parse_gateway_app_name(name) == ("modelship", "voice")

    def test_a_model_app_never_reads_as_a_pipeline(self):
        assert parse_pipeline_app_name("modelship.qwen-p-3f9a0c12de") is None
        assert parse_pipeline_app_name("modelship.qwen-3f9a0c12de") is None

    def test_the_replica_count_keeps_the_name(self):
        base = _assistant(name="voice", stt="whisper")
        assert base.deployment_name("g") == _assistant(name="voice", stt="whisper", num_replicas=3).deployment_name("g")

    @pytest.mark.parametrize(
        "changed",
        [
            {"stt": "other"},
            {"llm": "qwen"},
            {"tts": "kokoro", "llm": "qwen"},
            {"duplex": "half"},
            {"min_interruption_ms": 800},
            {"session_defaults": {"instructions": "Be brief."}},
            {"node_selector": {"modelship/hostname": "head"}},
        ],
    )
    def test_any_other_setting_renames_the_app(self, changed):
        base = _assistant(name="voice", stt="whisper")
        other = _assistant(**{"name": "voice", "stt": "whisper", **changed})
        assert base.deployment_name("g") != other.deployment_name("g")


class TestPipelineNodeSelector:
    def test_unset_the_pipeline_goes_to_any_node_that_runs_sherpa_onnx(self):
        opts = build_pipeline_deployment_options(_assistant(name="voice", llm="qwen"), {})
        assert opts["ray_actor_options"]["label_selector"] == {"modelship/variant": "in(cuda,cpu,vulkan,metal)"}

    def test_it_is_added_and_replaces_a_computed_key(self):
        pipeline = _assistant(
            name="voice", llm="qwen", node_selector={"modelship/hostname": "head", "modelship/variant": "vulkan"}
        )
        opts = build_pipeline_deployment_options(pipeline, {})
        assert opts["ray_actor_options"]["label_selector"] == {
            "modelship/hostname": "head",
            "modelship/variant": "vulkan",
        }

    def test_key_order_leaves_the_name_alone(self):
        one = _assistant(name="voice", llm="qwen", node_selector={"rack": "r1", "gpu": "fast"})
        other = _assistant(name="voice", llm="qwen", node_selector={"gpu": "fast", "rack": "r1"})
        assert one.deployment_name("g") == other.deployment_name("g")
        assert list(one.node_selector) == ["gpu", "rack"]
