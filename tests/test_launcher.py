import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from modelship import launcher
from modelship.utils.cli import parse_args


class TestGuardPythonVersion:
    def test_matching_version_passes(self):
        with patch.object(launcher.sys, "version_info", (3, 12, 10, "final", 0)):
            launcher._guard_python_version()  # no raise

    def test_mismatched_version_exits(self):
        with (
            patch.object(launcher.sys, "version_info", (3, 11, 4, "final", 0)),
            pytest.raises(SystemExit) as exc,
        ):
            launcher._guard_python_version()
        assert exc.value.code == 1


class TestCheckLoaderCapabilities:
    def test_empty_set_is_noop(self):
        launcher._check_loader_capabilities(set())  # no raise

    def test_llama_server_loader_never_gated(self):
        with patch("modelship.launcher.importlib.util.find_spec") as mock_find:
            launcher._check_loader_capabilities({"llama_server"})
        mock_find.assert_not_called()

    def test_passes_when_module_importable(self):
        with patch("modelship.launcher.importlib.util.find_spec", return_value=MagicMock()):
            launcher._check_loader_capabilities({"vllm"})  # no raise

    def test_exits_when_module_missing(self):
        with (
            patch("modelship.launcher.importlib.util.find_spec", return_value=None),
            pytest.raises(SystemExit) as exc,
        ):
            launcher._check_loader_capabilities({"vllm"})
        assert exc.value.code == 1


class TestValidateConfig:
    def _write(self, tmp_path, body):
        path = tmp_path / "models.yaml"
        path.write_text(body)
        return str(path)

    def _args(self, *argv):
        return parse_args(list(argv))

    def test_absent_config_returns_none(self, tmp_path):
        with patch("modelship.deploy.config.default_config_path", return_value=tmp_path / "nope.yaml"):
            assert launcher._validate_config(self._args()) is None

    def test_missing_explicit_config_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            launcher._validate_config(self._args("--config", str(tmp_path / "nope.yaml")))
        assert exc.value.code == 1

    def test_unknown_loader_exits(self, tmp_path):
        config = self._write(tmp_path, "models:\n  - name: m\n    loader: nope\n    model: x\n")
        with pytest.raises(SystemExit) as exc:
            launcher._validate_config(self._args("--config", config))
        assert exc.value.code == 1

    def test_duplicate_name_exits(self, tmp_path):
        config = self._write(
            tmp_path,
            "models:\n"
            "  - name: m\n    loader: llama_server\n    model: a.gguf\n    usecase: generate\n"
            "  - name: m\n    loader: llama_server\n    model: b.gguf\n    usecase: generate\n",
        )
        with pytest.raises(SystemExit) as exc:
            launcher._validate_config(self._args("--config", config))
        assert exc.value.code == 1

    def test_valid_config_returns_parsed_models(self, tmp_path):
        config = self._write(
            tmp_path, "models:\n  - name: m\n    loader: llama_server\n    model: x.gguf\n    usecase: generate\n"
        )
        parsed = launcher._validate_config(self._args("--config", config))
        assert parsed is not None
        assert [m.loader.value for m in parsed.models] == ["llama_server"]

    def test_validation_does_not_import_ray(self, tmp_path):
        config = self._write(
            tmp_path, "models:\n  - name: m\n    loader: llama_server\n    model: x.gguf\n    usecase: generate\n"
        )
        with patch.dict(sys.modules):
            sys.modules.pop("ray", None)
            launcher._validate_config(self._args("--config", config))
            assert "ray" not in sys.modules


class TestValidateConfigFromModelFlag:
    def _args(self, *argv):
        return parse_args(list(argv))

    def test_model_flag_returns_parsed_model(self):
        parsed = launcher._validate_config(
            self._args("--model", "x/y-GGUF:*Q4_K_M.gguf", "--loader", "llama_server", "--usecase", "generate")
        )
        assert parsed is not None
        assert [(m.name, m.loader.value) for m in parsed.models] == [("y", "llama_server")]

    def test_model_with_config_exits_at_parse_time(self, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text("models: []\n")
        with pytest.raises(SystemExit) as exc:
            self._args("--model", "x/y", "--config", str(config))
        assert exc.value.code == 2

    def test_missing_loader_exits(self):
        with pytest.raises(SystemExit) as exc:
            launcher._validate_config(self._args("--model", "x/y", "--usecase", "generate"))
        assert exc.value.code == 1

    def test_fractional_num_gpus_with_whole_gpu_loader_exits(self):
        with pytest.raises(SystemExit) as exc:
            launcher._validate_config(
                self._args(
                    "--model", "x/y.gguf", "--loader", "llama_server", "--usecase", "generate", "--num-gpus", "1.5"
                )
            )
        assert exc.value.code == 1


class TestFlaggedErrors:
    """Pydantic reports a models.yaml path; a --model deploy has no file to point at."""

    def test_field_paths_become_flags(self, capsys):
        with pytest.raises(SystemExit):
            launcher._validate_config(parse_args(["--model", "x/y"]))
        err = capsys.readouterr().err
        assert "--usecase" in err and "--loader" in err
        assert "models.0." not in err

    def test_config_errors_keep_pydantic_paths(self, tmp_path, capsys):
        config = tmp_path / "models.yaml"
        config.write_text("models:\n  - name: m\n    model: x\n")
        with pytest.raises(SystemExit):
            launcher._validate_config(parse_args(["--config", str(config)]))
        assert "models.0." in capsys.readouterr().err


class TestIsOwnHeadDeploy:
    """The driver-local capability gate only applies to the own-head path —
    join and existing-cluster paths hand scheduling to Ray's own capability resources."""

    def test_bare_own_head_returns_true(self):
        with patch.dict(os.environ, {}, clear=True):
            assert launcher._is_own_head_deploy() is True

    def test_address_set_returns_false(self):
        with patch.dict(os.environ, {"MSHIP_ADDRESS": "10.0.0.1:6380"}, clear=True):
            assert launcher._is_own_head_deploy() is False

    def test_use_existing_ray_cluster_true_returns_false(self):
        with patch.dict(os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "true"}, clear=True):
            assert launcher._is_own_head_deploy() is False

    def test_use_existing_ray_cluster_false_returns_true(self):
        with patch.dict(os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, clear=True):
            assert launcher._is_own_head_deploy() is True

    def test_zero_capacity_coordinator_returns_false(self):
        """The thin head holds the config for models that only a joiner can run."""
        env = {"MSHIP_NODE_NUM_CPUS": "0", "MSHIP_NODE_NUM_GPUS": "0"}
        with patch.dict(os.environ, env, clear=True):
            assert launcher._is_own_head_deploy() is False

    @pytest.mark.parametrize(
        "env",
        [
            {"MSHIP_NODE_NUM_CPUS": "0"},
            {"MSHIP_NODE_NUM_CPUS": "0", "MSHIP_NODE_NUM_GPUS": "1"},
            {"MSHIP_NODE_NUM_CPUS": "4", "MSHIP_NODE_NUM_GPUS": "0"},
            {"MSHIP_NODE_NUM_CPUS": "", "MSHIP_NODE_NUM_GPUS": ""},
        ],
    )
    def test_any_reserved_capacity_still_gates(self, env):
        with patch.dict(os.environ, env, clear=True):
            assert launcher._is_own_head_deploy() is True


class TestCmdDeploy:
    def test_forwards_argv_to_driver_after_gates(self):
        argv = ["--config", "models.yaml", "--reconcile"]
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(launcher, "resolve_cache_root", return_value="/tmp/mship-test-cache"),
            patch.object(launcher, "detect_accelerator", return_value="cpu"),
            patch.object(launcher, "_validate_config", return_value=MagicMock(models=[])),
            patch.object(launcher, "_check_loader_capabilities") as mock_gate,
            patch.object(launcher, "_guard_python_version") as mock_guard,
            patch("modelship.driver.main") as mock_driver_main,
        ):
            launcher._cmd_deploy(argv)

        mock_guard.assert_called_once()
        mock_gate.assert_called_once()
        mock_driver_main.assert_called_once_with(argv)

    def test_gate_skipped_on_address_join(self):
        argv = ["--address", "10.0.0.1:6380", "--node-num-gpus", "0"]
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(launcher, "resolve_cache_root", return_value="/tmp/mship-test-cache"),
            patch.object(launcher, "detect_accelerator", return_value="cpu"),
            patch.object(launcher, "_validate_config", return_value=MagicMock(models=[])),
            patch.object(launcher, "_check_loader_capabilities") as mock_gate,
            patch.object(launcher, "_guard_python_version"),
            patch("modelship.driver.main"),
        ):
            launcher._cmd_deploy(argv)
        mock_gate.assert_not_called()

    def test_gate_skipped_on_use_existing_ray_cluster(self):
        argv = ["--use-existing-ray-cluster"]
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(launcher, "resolve_cache_root", return_value="/tmp/mship-test-cache"),
            patch.object(launcher, "detect_accelerator", return_value="cpu"),
            patch.object(launcher, "_validate_config", return_value=MagicMock(models=[])),
            patch.object(launcher, "_check_loader_capabilities") as mock_gate,
            patch.object(launcher, "_guard_python_version"),
            patch("modelship.driver.main"),
        ):
            launcher._cmd_deploy(argv)
        mock_gate.assert_not_called()

    def test_gate_runs_on_own_head(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(launcher, "resolve_cache_root", return_value="/tmp/mship-test-cache"),
            patch.object(launcher, "detect_accelerator", return_value="cpu"),
            patch.object(launcher, "_validate_config", return_value=MagicMock(models=[])),
            patch.object(launcher, "_check_loader_capabilities") as mock_gate,
            patch.object(launcher, "_guard_python_version"),
            patch("modelship.driver.main"),
        ):
            launcher._cmd_deploy([])
        mock_gate.assert_called_once()

    def test_driver_not_imported_when_guard_exits(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(launcher, "resolve_cache_root", return_value="/tmp/mship-test-cache"),
            patch.object(launcher, "_guard_python_version", side_effect=SystemExit(1)),
            patch("modelship.driver.main") as mock_driver_main,
            pytest.raises(SystemExit),
        ):
            launcher._cmd_deploy([])
        mock_driver_main.assert_not_called()


class TestMain:
    def test_no_args_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            launcher.main([])
        assert exc.value.code == 2

    def test_unknown_command_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            launcher.main(["bogus"])
        assert exc.value.code == 2

    def test_deploy_dispatches_to_cmd_deploy(self):
        with patch.object(launcher, "_cmd_deploy") as mock_cmd:
            launcher.main(["deploy", "--reconcile"])
        mock_cmd.assert_called_once_with(["--reconcile"])

    def test_info_dispatches_to_cmd_info(self):
        with patch.object(launcher, "_cmd_info") as mock_cmd:
            launcher.main(["info"])
        mock_cmd.assert_called_once()

    def test_defaults_to_sys_argv(self):
        with (
            patch.object(sys, "argv", ["mship", "info"]),
            patch.object(launcher, "_cmd_info") as mock_cmd,
        ):
            launcher.main()
        mock_cmd.assert_called_once()


class TestCmdInfo:
    def test_prints_cpu_details(self, capsys):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(launcher, "detect_accelerator", return_value="cpu"),
            patch.object(launcher, "resolve_cache_root", return_value="/tmp/cache"),
        ):
            launcher._cmd_info()
        out = capsys.readouterr().out
        assert "accelerator: cpu" in out
        assert "cache: /tmp/cache" in out
        assert "llama-server: unset" in out

    def test_prints_inherited_llama_server_bin(self, capsys):
        """The bootstrapper (or the image) sets this before the engine starts."""
        with (
            patch.dict(os.environ, {"MSHIP_LLAMA_SERVER_BIN": "/builds/cuda/llama-server.sh"}, clear=True),
            patch.object(launcher, "detect_accelerator", return_value="cuda"),
            patch.object(launcher, "resolve_cache_root", return_value="/tmp/cache"),
        ):
            launcher._cmd_info()
        assert "llama-server: /builds/cuda/llama-server.sh" in capsys.readouterr().out
