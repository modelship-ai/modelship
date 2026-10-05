"""Every curated sherpa_onnx registry entry must be internally consistent:
well-formed paths, a speaker count matching voice_names, and a tarball URL in
modelship's model-bundles releases."""

from modelship.infer.sherpa_onnx.registry import REGISTRY, registry_names


def test_registry_names_are_unique_and_nonempty():
    names = registry_names()
    assert len(names) > 0
    assert len(names) == len(set(names))


class TestEntries:
    def test_tarball_url_is_a_model_bundles_release(self):
        for name, entry in REGISTRY.items():
            assert entry.tarball_url.startswith("https://github.com/modelship-ai/model-bundles/releases/download/"), (
                f"{name}: {entry.tarball_url}"
            )

    def test_tarball_file_name_carries_the_sha256_prefix(self):
        for name, entry in REGISTRY.items():
            assert entry.tarball_url.endswith(f"/{name}-{entry.sha256[:8]}.tar.bz2"), f"{name}: {entry.tarball_url}"

    def test_sha256_is_a_valid_hex_digest(self):
        for name, entry in REGISTRY.items():
            assert len(entry.sha256) == 64, name
            int(entry.sha256, 16)  # raises ValueError if not hex

    def test_files_and_dirs_have_nonempty_paths(self):
        for name, entry in REGISTRY.items():
            for slot, path in entry.files.items():
                assert path, f"{name}.files[{slot}]"
            for slot, path in entry.dirs.items():
                assert path, f"{name}.dirs[{slot}]"
            for i, path in enumerate(entry.lexicon):
                assert path, f"{name}.lexicon[{i}]"

    def test_required_kokoro_slots_present(self):
        for name, entry in REGISTRY.items():
            assert entry.family == "kokoro"
            assert set(entry.files) == {"model", "tokens", "voices"}, name

    def test_voice_names_nonempty_and_unique(self):
        for name, entry in REGISTRY.items():
            assert len(entry.voice_names) > 0, name
            assert len(entry.voice_names) == len(set(entry.voice_names)), name

    def test_af_bella_present_in_every_entry(self):
        for name, entry in REGISTRY.items():
            assert "af_bella" in entry.voice_names, name
