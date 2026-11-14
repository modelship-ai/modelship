import os
import stat

import pytest

from mship_bootstrap.variants import (
    VARIANTS,
    VariantError,
    engine_requirement,
    read_recorded,
    resolve,
    split_variant_flag,
    write_recorded,
)


class TestSplitVariantFlag:
    def test_extracts_the_flag_and_leaves_the_rest(self):
        flag, rest = split_variant_flag(["--cuda", "--config", "models.yaml"])
        assert flag == "cuda"
        assert rest == ["--config", "models.yaml"]

    def test_flag_position_does_not_matter(self):
        flag, rest = split_variant_flag(["--config", "models.yaml", "--cpu"])
        assert flag == "cpu"
        assert rest == ["--config", "models.yaml"]

    def test_no_flag(self):
        assert split_variant_flag(["--config", "x"]) == (None, ["--config", "x"])

    def test_two_variants_is_an_error_naming_both(self):
        with pytest.raises(VariantError, match="--cpu, --cuda"):
            split_variant_flag(["--cpu", "--cuda"])

    def test_leaves_unrelated_flags_untouched(self):
        _, rest = split_variant_flag(["--cuda", "--reconcile", "--gateway-name", "x"])
        assert rest == ["--reconcile", "--gateway-name", "x"]


class TestResolve:
    def test_flag_wins(self):
        assert resolve("cuda", env={}).name == "cuda"

    def test_env_var_when_no_flag(self):
        assert resolve(None, env={"MSHIP_VARIANT": "metal"}).name == "metal"

    def test_flag_and_matching_env_var_agree(self):
        assert resolve("cpu", env={"MSHIP_VARIANT": "cpu"}).name == "cpu"

    def test_conflict_is_an_error(self):
        with pytest.raises(VariantError, match="conflicts with"):
            resolve("cpu", env={"MSHIP_VARIANT": "cuda"})

    def test_no_variant_lists_every_option(self):
        with pytest.raises(VariantError) as exc:
            resolve(None, env={})
        message = str(exc.value)
        assert "no variant selected" in message
        for name in VARIANTS:
            assert f"--{name}" in message

    def test_unknown_env_var(self):
        with pytest.raises(VariantError, match="not a variant"):
            resolve(None, env={"MSHIP_VARIANT": "gpu"})

    def test_empty_env_var_is_treated_as_unset(self):
        with pytest.raises(VariantError, match="no variant selected"):
            resolve(None, env={"MSHIP_VARIANT": "  "})


class TestResolveRecorded:
    def test_recorded_is_used_when_nothing_else_is_given(self):
        assert resolve(None, env={}, recorded="metal").name == "metal"

    def test_flag_overrides_it_silently(self):
        assert resolve("cuda", env={}, recorded="thin").name == "cuda"

    def test_env_var_overrides_it_silently(self):
        assert resolve(None, env={"MSHIP_VARIANT": "cpu"}, recorded="thin").name == "cpu"

    def test_unknown_recorded_value_is_an_error(self):
        with pytest.raises(VariantError, match="not a variant"):
            resolve(None, env={}, recorded="gpu")

    def test_unknown_recorded_value_names_the_file(self):
        with pytest.raises(VariantError, match="env"):
            resolve(None, env={}, recorded="gpu")

    @pytest.mark.parametrize("recorded", [None, "", "  "])
    def test_absent_recorded_value_reads_as_unset(self, recorded):
        with pytest.raises(VariantError, match="no variant selected"):
            resolve(None, env={}, recorded=recorded)


class TestRecordedFile:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "env")
        write_recorded(path, "cuda")
        assert read_recorded(path) == "cuda"

    def test_carries_a_do_not_edit_header(self, tmp_path):
        path = str(tmp_path / "env")
        write_recorded(path, "cpu")
        body = open(path).read()
        assert body.startswith("# ")
        assert "do not edit" in body

    def test_is_read_only(self, tmp_path):
        path = str(tmp_path / "env")
        write_recorded(path, "cpu")
        assert not stat.S_IMODE(os.stat(path).st_mode) & stat.S_IWUSR

    def test_last_bootstrap_wins(self, tmp_path):
        path = str(tmp_path / "env")
        write_recorded(path, "cpu")
        write_recorded(path, "thin")
        assert read_recorded(path) == "thin"

    def test_missing_file(self, tmp_path):
        assert read_recorded(str(tmp_path / "nope")) is None

    def test_file_without_the_key(self, tmp_path):
        path = tmp_path / "env"
        path.write_text("# a comment\n\nHF_TOKEN=secret\n")
        assert read_recorded(str(path)) is None

    def test_other_keys_are_ignored(self, tmp_path):
        path = tmp_path / "env"
        path.write_text("MSHIP_RECORDED_TEST_KEY=secret\nMSHIP_VARIANT=metal\nMSHIP_HOME=/elsewhere\n")
        before = dict(os.environ)
        assert read_recorded(str(path)) == "metal"
        assert os.environ == before

    def test_empty_value_reads_as_unset(self, tmp_path):
        path = tmp_path / "env"
        path.write_text("MSHIP_VARIANT=\n")
        assert read_recorded(str(path)) is None

    def test_tolerates_quotes_and_whitespace(self, tmp_path):
        path = tmp_path / "env"
        path.write_text('  MSHIP_VARIANT = "cuda" \n')
        assert read_recorded(str(path)) == "cuda"

    def test_unreadable_file(self, tmp_path):
        assert read_recorded(str(tmp_path)) is None


class TestVariantDefinitions:
    def test_cpu_always_includes_vllm_cpu(self):
        assert VARIANTS["cpu"].extras == ("cpu", "vllm-cpu")

    def test_only_the_torch_variants_pass_extra_indexes(self):
        for name in ("cpu", "cuda", "vulkan"):
            assert VARIANTS[name].index_args
        for name in ("metal", "thin"):
            assert VARIANTS[name].index_args == ()

    def test_vulkan_installs_the_cpu_engine_environment(self):
        assert VARIANTS["vulkan"].extras == VARIANTS["cpu"].extras
        assert VARIANTS["vulkan"].index_args == VARIANTS["cpu"].index_args

    @pytest.mark.parametrize("name", ["cpu", "cuda", "vulkan"])
    def test_pypi_outranks_the_accelerator_indexes(self, name):
        """Both also serve triton/torchcodec as artifacts the lock never saw."""
        indexes = [a for a in VARIANTS[name].index_args if a.startswith("http")]
        assert indexes[0] == "https://pypi.org/simple"
        assert len(indexes) > 1

    def test_hardware_requirements(self):
        assert VARIANTS["cuda"].requires_accelerator == "cuda"
        assert VARIANTS["metal"].requires_accelerator == "metal"
        assert VARIANTS["vulkan"].requires_accelerator == "vulkan"
        assert VARIANTS["cpu"].requires_accelerator is None
        assert VARIANTS["thin"].requires_accelerator is None

    def test_engine_requirement_string(self):
        assert engine_requirement(VARIANTS["cpu"], "0.8.0") == "mship-engine[cpu,vllm-cpu]==0.8.0"
        assert engine_requirement(VARIANTS["cuda"], "0.8.0") == "mship-engine[cuda]==0.8.0"
