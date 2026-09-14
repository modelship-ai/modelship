import os
from unittest.mock import patch

import pytest

from mship_bootstrap import cli, paths, variants
from mship_bootstrap.variants import VARIANTS


@pytest.fixture
def provisioned(tmp_path, monkeypatch):
    """Everything past the gates stubbed, so tests exercise argv and env only."""
    monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
    monkeypatch.delenv("MSHIP_VARIANT", raising=False)
    monkeypatch.delenv("MSHIP_LLAMA_SERVER_BIN", raising=False)
    with (
        patch.object(cli.gates, "check_platform"),
        patch.object(cli.gates, "check_hardware"),
        patch.object(cli.gates, "check_toolchain"),
        patch.object(cli.uv_binary, "ensure_uv", return_value="/usr/bin/uv"),
        patch.object(cli.engine, "provision", return_value="/env/bin/python"),
        patch.object(cli.engine, "is_current", return_value=True),
        patch.object(cli.llama_cpp, "provision", return_value="/builds/llama-server.sh"),
        patch.object(cli.llama_cpp, "locate", return_value="/builds/llama-server.sh"),
        patch.object(cli.llama_cpp, "warn_if_no_cuda_device"),
        patch("os.execve") as execve,
    ):
        yield execve


class TestUsage:
    def test_no_args_exits_2(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main([])
        assert exc.value.code == 2

    def test_unknown_command_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["serve"])
        assert exc.value.code == 2


class TestVariantRequired:
    def test_deploy_without_a_variant_lists_the_options(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["deploy", "--config", "models.yaml"])
        assert "no variant selected" in str(exc.value)

    def test_two_variants_is_refused(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["deploy", "--cpu", "--cuda"])
        assert "pick one variant" in str(exc.value)


class TestExec:
    def test_execs_the_engine_via_module(self, provisioned, tmp_path):
        cli.main(["deploy", "--cpu", "--config", "models.yaml"])
        python, args, _env = provisioned.call_args[0]
        assert python == paths.venv_python("cpu")
        assert args == [python, "-m", "modelship.launcher", "deploy", "--config", "models.yaml"]

    def test_variant_flag_is_not_passed_to_the_engine(self, provisioned):
        cli.main(["deploy", "--cuda", "--reconcile"])
        assert "--cuda" not in provisioned.call_args[0][1]

    def test_env_var_variant_needs_no_flag(self, provisioned, monkeypatch):
        monkeypatch.setenv("MSHIP_VARIANT", "cpu")
        cli.main(["deploy", "--config", "x"])
        assert provisioned.called


class TestEngineEnvironment:
    def test_llama_server_bin_is_set_before_exec(self, provisioned):
        cli.main(["deploy", "--cpu"])
        env = provisioned.call_args[0][2]
        assert env["MSHIP_LLAMA_SERVER_BIN"] == "/builds/llama-server.sh"

    def test_thin_advertises_no_capacity(self, provisioned):
        cli.main(["deploy", "--thin"])
        env = provisioned.call_args[0][2]
        assert env["MSHIP_NODE_NUM_CPUS"] == "0"
        assert env["MSHIP_NODE_NUM_GPUS"] == "0"

    def test_other_variants_do_not_pin_capacity(self, provisioned):
        cli.main(["deploy", "--cpu"])
        env = provisioned.call_args[0][2]
        assert "MSHIP_NODE_NUM_CPUS" not in env

    def test_explicit_node_sizing_wins_over_thin_defaults(self, provisioned, monkeypatch):
        monkeypatch.setenv("MSHIP_NODE_NUM_CPUS", "4")
        cli.main(["deploy", "--thin"])
        assert provisioned.call_args[0][2]["MSHIP_NODE_NUM_CPUS"] == "4"

    def test_cache_dir_defaults_under_mship_home(self, provisioned, tmp_path, monkeypatch):
        # The dev container exports one, and setdefault would keep it.
        monkeypatch.delenv("MSHIP_CACHE_DIR", raising=False)
        cli.main(["deploy", "--cpu"])
        assert provisioned.call_args[0][2]["MSHIP_CACHE_DIR"] == os.path.join(str(tmp_path), "cache")

    def test_explicit_cache_dir_is_preserved(self, provisioned, monkeypatch):
        monkeypatch.setenv("MSHIP_CACHE_DIR", "/mnt/shared/models")
        cli.main(["deploy", "--cpu"])
        assert provisioned.call_args[0][2]["MSHIP_CACHE_DIR"] == "/mnt/shared/models"

    def test_node_cache_dir_defaults_under_mship_home(self, provisioned, tmp_path, monkeypatch):
        monkeypatch.delenv("MSHIP_NODE_CACHE_DIR", raising=False)
        cli.main(["deploy", "--cpu"])
        assert provisioned.call_args[0][2]["MSHIP_NODE_CACHE_DIR"] == os.path.join(str(tmp_path), "node-cache")

    def test_explicit_node_cache_dir_is_preserved(self, provisioned, monkeypatch):
        monkeypatch.setenv("MSHIP_NODE_CACHE_DIR", "/scratch/node-cache")
        cli.main(["deploy", "--cpu"])
        assert provisioned.call_args[0][2]["MSHIP_NODE_CACHE_DIR"] == "/scratch/node-cache"

    def test_cuda_checks_for_a_live_device(self, provisioned):
        cli.main(["deploy", "--cuda"])
        cli.llama_cpp.warn_if_no_cuda_device.assert_called_once()

    def test_non_cuda_skips_that_check(self, provisioned):
        cli.main(["deploy", "--cpu"])
        cli.llama_cpp.warn_if_no_cuda_device.assert_not_called()


class TestInfo:
    def test_info_without_a_variant_answers_locally(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
        monkeypatch.delenv("MSHIP_VARIANT", raising=False)
        cli.main(["info"])
        out = capsys.readouterr().out
        assert "MSHIP_HOME" in out
        assert "none provisioned" in out

    def test_info_with_a_variant_execs_the_engine(self, provisioned):
        cli.main(["info", "--cpu"])
        assert provisioned.call_args[0][1][-1] == "info"

    def test_whitespace_only_env_var_still_answers_locally(self, tmp_path, monkeypatch, capsys):
        """Matches resolve()'s own whitespace-as-unset handling."""
        monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
        monkeypatch.setenv("MSHIP_VARIANT", "  ")
        cli.main(["info"])
        out = capsys.readouterr().out
        assert "MSHIP_HOME" in out
        assert "none provisioned" in out


class TestPaths:
    def test_mship_home_relocates_everything(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
        assert paths.env_dir("cpu").startswith(str(tmp_path))
        assert paths.builds_dir("cpu").startswith(str(tmp_path))
        assert paths.bin_dir().startswith(str(tmp_path))

    def test_builds_are_scoped_per_variant(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
        assert paths.builds_dir("cpu") != paths.builds_dir("cuda")

    def test_cache_dir_is_not_the_bootstrappers_concern(self, monkeypatch, tmp_path):
        """MSHIP_CACHE_DIR may point at shared storage; MSHIP_HOME must not."""
        monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
        monkeypatch.setenv("MSHIP_CACHE_DIR", "/mnt/shared")
        for name in VARIANTS:
            assert not paths.env_dir(name).startswith("/mnt/shared")
            assert not paths.builds_dir(name).startswith("/mnt/shared")


class TestThinSkipsLlamaServer:
    def test_thin_does_not_provision_a_binary_it_cannot_use(self, provisioned):
        cli.main(["bootstrap", "--thin"])
        cli.llama_cpp.provision.assert_not_called()

    def test_serving_variants_still_provision(self, provisioned):
        cli.main(["bootstrap", "--cpu"])
        cli.llama_cpp.provision.assert_called_once()

    def test_thin_deploys_without_looking_for_one(self, provisioned):
        cli.main(["deploy", "--thin"])
        cli.llama_cpp.locate.assert_not_called()
        assert "MSHIP_LLAMA_SERVER_BIN" not in provisioned.call_args[0][2]


class TestBootstrap:
    def test_provisions_and_does_not_exec(self, provisioned):
        cli.main(["bootstrap", "--cpu"])
        cli.engine.provision.assert_called_once()
        assert not provisioned.called

    def test_records_the_variant(self, provisioned, tmp_path):
        cli.main(["bootstrap", "--metal"])
        assert variants.read_recorded(paths.env_file()) == "metal"

    def test_requires_a_variant(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["bootstrap"])
        assert "no variant selected" in str(exc.value)

    def test_rejects_trailing_arguments(self, provisioned):
        with pytest.raises(SystemExit) as exc:
            cli.main(["bootstrap", "--cpu", "--config", "models.yaml"])
        assert "--config models.yaml" in str(exc.value)
        cli.engine.provision.assert_not_called()

    def test_a_failed_bootstrap_records_nothing(self, provisioned, tmp_path):
        cli.engine.provision.side_effect = cli.engine.EngineError("error: boom")
        with pytest.raises(SystemExit):
            cli.main(["bootstrap", "--cpu"])
        assert not os.path.exists(paths.env_file())

    def test_never_checks_the_hardware(self, provisioned):
        cli.main(["bootstrap", "--cuda"])
        cli.gates.check_hardware.assert_not_called()
        cli.engine.provision.assert_called_once()

    def test_checks_the_toolchain(self, provisioned):
        cli.main(["bootstrap", "--cuda"])
        cli.gates.check_toolchain.assert_called_once_with("cuda")

    def test_a_missing_toolchain_provisions_nothing(self, provisioned):
        cli.gates.check_toolchain.side_effect = cli.gates.GateError("error: --cuda is missing ninja.")
        with pytest.raises(SystemExit) as exc:
            cli.main(["bootstrap", "--cuda"])
        assert "ninja" in str(exc.value)
        cli.engine.provision.assert_not_called()

    def test_an_unknown_flag_is_refused(self, provisioned):
        with pytest.raises(SystemExit) as exc:
            cli.main(["bootstrap", "--cpu", "--no-hardware-check"])
        assert "takes no arguments" in str(exc.value)


class TestDeployInstallsNothing:
    def test_deploy_still_gates_on_hardware(self, provisioned):
        cli.main(["deploy", "--cuda", "--config", "x"])
        cli.gates.check_hardware.assert_called_once_with("cuda")

    def test_deploy_never_provisions(self, provisioned):
        cli.main(["deploy", "--cpu", "--config", "x"])
        cli.uv_binary.ensure_uv.assert_not_called()
        cli.engine.provision.assert_not_called()

    def test_a_stale_environment_stops_the_deploy(self, provisioned):
        cli.engine.is_current.return_value = False
        with pytest.raises(SystemExit) as exc:
            cli.main(["deploy", "--cpu", "--config", "x"])
        assert "mship bootstrap --cpu" in str(exc.value)
        assert not provisioned.called

    def test_a_stale_environment_is_not_repaired(self, provisioned):
        cli.engine.is_current.return_value = False
        with pytest.raises(SystemExit):
            cli.main(["deploy", "--cpu"])
        cli.engine.provision.assert_not_called()


class TestRecordedVariant:
    def test_deploy_needs_no_flag_once_bootstrapped(self, provisioned):
        cli.main(["bootstrap", "--cuda"])
        cli.main(["deploy", "--config", "models.yaml"])
        assert provisioned.call_args[0][0] == paths.venv_python("cuda")

    def test_a_flag_overrides_the_record(self, provisioned):
        cli.main(["bootstrap", "--cuda"])
        cli.main(["deploy", "--thin"])
        assert provisioned.call_args[0][0] == paths.venv_python("thin")

    def test_bootstrapping_another_variant_moves_the_record(self, provisioned):
        cli.main(["bootstrap", "--cuda"])
        cli.main(["bootstrap", "--cpu"])
        assert variants.read_recorded(paths.env_file()) == "cpu"

    def test_an_unknown_recorded_value_is_an_error(self, provisioned, tmp_path):
        (tmp_path / "env").write_text("MSHIP_VARIANT=gpu\n")
        with pytest.raises(SystemExit) as exc:
            cli.main(["deploy", "--config", "x"])
        assert "not a variant" in str(exc.value)
