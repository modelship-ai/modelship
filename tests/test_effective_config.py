"""Tests for the deploy effective-config layer."""

import pytest
from ray.serve.schema import ApplicationStatus

from modelship.deploy.config import default_config_path, load_raw_models, resolve_config_path
from modelship.deploy.effective_config import (
    merge,
    read_effective,
    read_targets,
    resolve_mode,
    to_config,
    write_effective,
)
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


class TestResolveMode:
    def test_default_is_additive(self):
        assert resolve_mode(reconcile=False) == "additive"

    def test_reconcile(self):
        assert resolve_mode(reconcile=True) == "reconcile"


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


class TestReadWriteEffective:
    def test_write_then_read(self):
        store = _MemoryStore()
        models = [_model("a"), _model("b")]
        write_effective(store, "modelship api", models)
        assert read_effective(store, "modelship api") == models

    def test_read_absent_gateway_is_empty(self):
        assert read_effective(_MemoryStore(), "never-deployed") == []


class TestRawRoundTrip:
    """Store holds raw dicts because a normalized vLLM config does NOT round-trip
    (num_gpus=2 -> num_gpus=1.0/tp=2, which fails re-validation); raw dicts reload identically."""

    def test_multi_gpu_vllm_survives_store_roundtrip(self):
        raw = {"name": "x", "model": "org/x", "usecase": "generate", "loader": "vllm", "num_gpus": 2}
        store = _MemoryStore()
        write_effective(store, "g", [raw])

        back = read_effective(store, "g")
        cfg = to_config(back)  # must not raise on the normalized-but-reloaded config
        m = cfg.models[0]
        assert m.num_gpus == 1.0
        assert m.vllm_engine_kwargs.tensor_parallel_size == 2
        # identity preserved: same fingerprint as a fresh validate of the original
        assert m.fingerprint() == ModelshipModelConfig.model_validate(raw).fingerprint()

    def test_stored_value_is_raw_not_normalized(self):
        # The persisted value keeps the user's num_gpus=2, not the normalized 1.0.
        raw = {"name": "x", "model": "org/x", "usecase": "generate", "loader": "vllm", "num_gpus": 2}
        store = _MemoryStore()
        write_effective(store, "g", [raw])
        assert store.get("effective/g")["models"][0]["num_gpus"] == 2


def _dep(name: str, gw: str = "g", **overrides) -> str:
    return ModelshipModelConfig.model_validate(_model(name, **overrides)).deployment_name(gw)


def _running(*apps: str) -> dict[str, ApplicationStatus]:
    return dict.fromkeys(apps, ApplicationStatus.RUNNING)


class TestComputeDeployPlan:
    def test_a_model_with_no_app_is_added(self):
        from modelship.deploy.strategy import compute_deploy_plan

        plan = compute_deploy_plan(to_config([_model("a")]), _running("g"), "g")
        assert [c.name for c in plan.models_to_add] == ["a"]

    def test_idempotent_when_all_live(self):
        from modelship.deploy.strategy import compute_deploy_plan

        plan = compute_deploy_plan(to_config([_model("a")]), _running(_dep("a"), "g"), "g")
        assert plan.models_to_add == []
        assert plan.stale_apps == []

    def test_replaced_and_dropped_apps_are_stale(self):
        from modelship.deploy.strategy import compute_deploy_plan

        existing = _running(_dep("a", num_cpus=1), _dep("b"), "g")
        plan = compute_deploy_plan(to_config([_model("a", num_cpus=2)]), existing, "g")
        assert plan.stale_apps == sorted([_dep("a", num_cpus=1), _dep("b")])

    def test_other_gateways_and_unprefixed_apps_are_never_stale(self):
        from modelship.deploy.strategy import compute_deploy_plan

        existing = _running(_dep("b", gw="edge"), "b-0123456789", "g", "edge")
        plan = compute_deploy_plan(to_config([_model("a")]), existing, "g")
        assert plan.stale_apps == []


class TestComputeDeployPlanAppStatus:
    @pytest.mark.parametrize(
        ("status", "redeployed"),
        [
            (ApplicationStatus.DEPLOY_FAILED, True),
            (ApplicationStatus.DELETING, True),
            (ApplicationStatus.RUNNING, False),
            (ApplicationStatus.DEPLOYING, False),
            (ApplicationStatus.UNHEALTHY, False),
        ],
    )
    def test_only_a_failed_or_deleting_app_is_deployed_again(self, status, redeployed):
        from modelship.deploy.strategy import compute_deploy_plan

        plan = compute_deploy_plan(to_config([_model("a")]), {_dep("a"): status}, "g")
        assert [c.name for c in plan.models_to_add] == (["a"] if redeployed else [])

    def test_a_failed_target_is_not_stale(self):
        from modelship.deploy.strategy import compute_deploy_plan

        plan = compute_deploy_plan(to_config([_model("a")]), {_dep("a"): ApplicationStatus.DEPLOY_FAILED}, "g")
        assert plan.stale_apps == []


class TestReadTargets:
    @pytest.mark.asyncio
    async def test_maps_each_model_to_its_app(self):
        store = _MemoryStore()
        write_effective(store, "g", [_model("a"), _model("b")])
        assert await read_targets(store, "g") == {"a": _dep("a"), "b": _dep("b")}

    @pytest.mark.asyncio
    async def test_an_empty_config_targets_nothing(self):
        store = _MemoryStore()
        write_effective(store, "g", [])
        assert await read_targets(store, "g") == {}

    @pytest.mark.asyncio
    async def test_none_without_a_config(self):
        assert await read_targets(_MemoryStore(), "g") is None


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


class TestDefaultConfigPath:
    def test_points_at_models_yaml_under_config_dir(self, tmp_path):
        assert default_config_path(tmp_path) == tmp_path / "models.yaml"


class TestResolveConfigPath:
    def test_explicit_path_wins_even_if_default_exists(self, tmp_path):
        (tmp_path / "models.yaml").write_text("models: []\n")
        explicit = tmp_path / "other.yaml"
        explicit.write_text("models: []\n")
        assert resolve_config_path(str(explicit), tmp_path) == str(explicit)

    def test_explicit_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="--config"):
            resolve_config_path(str(tmp_path / "missing.yaml"), tmp_path)

    def test_default_path_used_when_no_explicit_arg(self, tmp_path):
        (tmp_path / "models.yaml").write_text("models: []\n")
        assert resolve_config_path(None, tmp_path) == str(tmp_path / "models.yaml")

    def test_default_missing_raises_with_pointer_to_examples(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config/examples"):
            resolve_config_path(None, tmp_path)
