"""The deploy ledger's store layer: versions, merging, raw round trips and config loading."""

import pytest
from ray.serve.schema import ApplicationStatus

from modelship.deploy.config import load_raw_models
from modelship.deploy.ledger import Version, commit_version, merge, read_versions, to_config
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.state import MemoryStoreActor

# The plain class behind @ray.remote — a real store with no cluster, the same
# pattern test_state.py uses.
_MemoryStore = MemoryStoreActor.__ray_metadata__.modified_class


def _model(name: str, **overrides) -> dict:
    """A minimal raw model dict (the form the store holds)."""
    base = {"name": name, "model": f"org/{name}", "usecase": "generate", "loader": "llama_server"}
    base.update(overrides)
    return base


class TestMerge:
    def test_additive_union(self):
        merged = merge([_model("a")], [_model("b")], "g", "additive")
        assert [m["name"] for m in merged] == ["a", "b"]

    def test_additive_dedups_identical_config(self):
        # same name + identical config = same fingerprint = idempotent skip
        merged = merge([_model("a")], [_model("a")], "g", "additive")
        assert [m["name"] for m in merged] == ["a"]

    def test_additive_replaces_same_name_different_config(self):
        # same name, different config -> the new one replaces the old (one deployment per name)
        a1 = _model("a", num_cpus=1)
        a2 = _model("a", num_cpus=2)
        merged = merge([a1], [a2], "g", "additive")
        assert merged == [a2]

    def test_replacing_a_name_with_different_weights_warns(self, caplog):
        vllm = _model("qwen3-8b", model="Qwen/Qwen3-8B", loader="vllm")
        gguf = _model("qwen3-8b", model="org/Qwen3-8B-GGUF:*Q4_K_M.gguf")
        with caplog.at_level("WARNING"):
            merged = merge([vllm], [gguf], "g", "additive")
        assert merged == [gguf]
        assert "REPLACED" in caplog.text

    def test_replacing_a_name_with_the_same_weights_does_not_warn(self, caplog):
        with caplog.at_level("WARNING"):
            merge([_model("a", num_cpus=1)], [_model("a", num_cpus=2)], "g", "additive")
        assert caplog.text == ""

    def test_additive_rejects_duplicate_name_within_input(self):
        a1 = _model("a", num_cpus=1)
        a2 = _model("a", num_cpus=2)
        with pytest.raises(ValueError, match="duplicate model name"):
            merge([], [a1, a2], "g", "additive")

    def test_reconcile_replaces(self):
        merged = merge([_model("a"), _model("b")], [_model("c")], "g", "reconcile")
        assert [m["name"] for m in merged] == ["c"]

    def test_reconcile_rejects_duplicate_name_within_input(self):
        a1 = _model("a", num_cpus=1)
        a2 = _model("a", num_cpus=2)
        with pytest.raises(ValueError, match="duplicate model name"):
            merge([], [a1, a2], "g", "reconcile")


class TestVersions:
    def test_a_gateway_never_committed_has_no_versions(self):
        assert read_versions(_MemoryStore(), "g") == (None, None)

    def test_the_first_commit_is_version_1_with_no_previous(self):
        store = _MemoryStore()
        assert commit_version(store, "g", [_model("a")]) == Version(1, [_model("a")])
        assert read_versions(store, "g") == (Version(1, [_model("a")]), None)

    def test_a_commit_keeps_only_the_version_before_it(self):
        store = _MemoryStore()
        for name in ("a", "b", "c"):
            commit_version(store, "g", [_model(name)])
        assert read_versions(store, "g") == (Version(3, [_model("c")]), Version(2, [_model("b")]))
        assert "previous" not in store.get("effective/g")["previous"]

    def test_versions_are_per_gateway(self):
        store = _MemoryStore()
        commit_version(store, "g", [_model("a")])
        assert read_versions(store, "other") == (None, None)

    def test_an_emptied_gateway_has_a_committed_empty_version(self):
        store = _MemoryStore()
        commit_version(store, "g", [_model("a")])
        commit_version(store, "g", [])
        assert read_versions(store, "g")[0] == Version(2, [])

    def test_apps_maps_each_model_to_its_app(self):
        assert Version(1, [_model("a")]).apps("g") == {"a": _dep("a")}


class TestRawRoundTrip:
    """Store holds raw dicts because a normalized vLLM config does NOT round-trip
    (num_gpus=2 -> num_gpus=1.0/tp=2, which fails re-validation); raw dicts reload identically."""

    def test_multi_gpu_vllm_survives_store_roundtrip(self):
        raw = {"name": "x", "model": "org/x", "usecase": "generate", "loader": "vllm", "num_gpus": 2}
        store = _MemoryStore()
        commit_version(store, "g", [raw])

        committed, _ = read_versions(store, "g")
        assert committed is not None
        cfg = to_config(committed.models)  # must not raise on the normalized-but-reloaded config
        m = cfg.models[0]
        assert m.num_gpus == 1.0
        assert m.vllm_engine_kwargs.tensor_parallel_size == 2
        # identity preserved: same fingerprint as a fresh validate of the original
        assert m.fingerprint() == ModelshipModelConfig.model_validate(raw).fingerprint()

    def test_stored_value_is_raw_not_normalized(self):
        # The persisted value keeps the user's num_gpus=2, not the normalized 1.0.
        raw = {"name": "x", "model": "org/x", "usecase": "generate", "loader": "vllm", "num_gpus": 2}
        store = _MemoryStore()
        commit_version(store, "g", [raw])
        assert store.get("effective/g")["models"][0]["num_gpus"] == 2


def _dep(name: str, gw: str = "g", **overrides) -> str:
    return ModelshipModelConfig.model_validate(_model(name, **overrides)).deployment_name(gw)


def _running(*apps: str) -> dict[str, ApplicationStatus]:
    return dict.fromkeys(apps, ApplicationStatus.RUNNING)


class TestCase2AdditiveAccumulation:
    """Additive deploys accumulate beyond the last input, and that
    accumulation must survive in the effective config."""

    def test_additive_then_reconcile(self):
        a, b, c, d = _model("a"), _model("b"), _model("c"), _model("d")
        # deploy A,B,C additively
        eff = merge([], [a, b, c], "g", "additive")
        # later additive upgrade declaring only D -> effective keeps all four
        eff = merge(eff, [d], "g", "additive")
        assert sorted(m["name"] for m in eff) == ["a", "b", "c", "d"]
        # a reconcile declaring only D collapses the effective set to D
        eff = merge(eff, [d], "g", "reconcile")
        assert [m["name"] for m in eff] == ["d"]


class TestLoadRawModels:
    def _write(self, tmp_path, text: str) -> str:
        p = tmp_path / "models.yaml"
        p.write_text(text)
        return str(p)

    def test_reads_models_list(self, tmp_path):
        path = self._write(tmp_path, "models:\n  - name: a\n  - name: b\n")
        assert load_raw_models(path) == [{"name": "a"}, {"name": "b"}]

    def test_empty_file_is_empty_list(self, tmp_path):
        # yaml.safe_load("") -> None; `or {}` then `.get` yields no models.
        assert load_raw_models(self._write(tmp_path, "")) == []

    def test_missing_models_key_is_empty_list(self, tmp_path):
        assert load_raw_models(self._write(tmp_path, "other: 1\n")) == []

    def test_top_level_list_rejected(self, tmp_path):
        # A bare list at the top level has no .get(); must raise cleanly, not AttributeError.
        path = self._write(tmp_path, "- name: a\n- name: b\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_raw_models(path)

    def test_top_level_scalar_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must be a mapping"):
            load_raw_models(self._write(tmp_path, "just a string\n"))

    def test_models_not_a_list_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="'models' must be a list"):
            load_raw_models(self._write(tmp_path, "models:\n  a: 1\n"))

    def test_missing_file_names_the_flag(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="--config"):
            load_raw_models(str(tmp_path / "missing.yaml"))
