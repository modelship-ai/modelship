import fnmatch
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from huggingface_hub import hf_hub_download, model_info, snapshot_download
from tqdm.asyncio import tqdm_asyncio

from modelship.infer.sources.progress import DownloadProgress
from modelship.logging import get_logger

logger = get_logger("startup")

# Read by every worker thread hf_hub_download/snapshot_download spawns, so
# one model's files all report into a single aggregate.
_active_download: DownloadProgress | None = None


class _DownloadProgressLogger(tqdm_asyncio):
    """`tqdm_class` for HF downloads: one instance per file, so it feeds the
    model-wide `DownloadProgress` instead of logging itself."""

    def update(self, n: float | None = 1):
        result = super().update(n)
        if _active_download is not None and n is not None and n > 0:
            _active_download.add(int(n))
        return result

    def display(self, msg=None, pos=None):
        pass


class HfSource(NamedTuple):
    """An HF repo pinned to a commit. `filename` XOR `patterns` picks `hf_hub_download`
    or `snapshot_download`; `first_shard` is the entry-point file of a snapshot."""

    repo: str
    revision: str
    filename: str | None
    patterns: list[str] | None
    first_shard: str | None
    total_bytes: int | None

    @property
    def resolves_to_gguf(self) -> bool:
        filename = self.filename or self.first_shard
        return bool(filename and filename.lower().endswith(".gguf"))


def _matches_any_pattern(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p + "*" if p.endswith("/") else p) for p in patterns)


def _sum_sizes(files: Iterable[str], sizes_by_file: dict[str, int | None]) -> int | None:
    """Total bytes for `files`, or None if any of them is missing size metadata."""
    total = 0
    for f in files:
        size = sizes_by_file.get(f)
        if size is None:
            return None
        total += size
    return total


def _select_patterns(repo_files: list[str], trust_remote_code: bool = False) -> list[str]:
    """Universal filter: prefer safetensors over bin/h5/onnx if present."""
    has_safetensors = any(f.endswith(".safetensors") or ".safetensors.index.json" in f for f in repo_files)

    patterns = [
        "*.json",
        "*.txt",
        "*.model",
        "tokenizer*",
        "vocab*",
        "merges*",
        "*.jinja",
        "chat_template*",
        "preprocessor_config.json",
        "generation_config.json",
        "image_processor_config.json",
        "processor_config.json",
    ]

    if trust_remote_code:
        patterns.append("*.py")
        patterns.append("**/*.py")

    if has_safetensors:
        patterns.extend(["*.safetensors", "*.safetensors.index.json", "**/*.safetensors"])
    else:
        # Fallback to bin if no safetensors
        patterns.extend(["*.bin", "*.bin.index.json", "**/*.bin"])

    return patterns


def _format_gguf_variants(repo_files: list[str]) -> str:
    """Format the GGUF files in a repo as a bullet list for error messages."""
    ggufs = sorted(f for f in repo_files if f.endswith(".gguf"))
    return "\n".join(f"  - {f}" for f in ggufs)


def check_hf_source(repo: str, selector: str | None, trust_remote_code: bool = False) -> HfSource:
    """`model_info` gives both the file listing (surfacing auth/missing-repo/
    selector-no-match) and the commit SHA every node later downloads."""
    try:
        info = model_info(repo, files_metadata=True)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch info for HF repo {repo!r}: {e}") from e

    if info.siblings is None:
        raise RuntimeError(f"HF repo {repo!r} returned no file listing")
    if info.sha is None:
        raise RuntimeError(f"HF repo {repo!r} returned no commit SHA")

    repo_files = [s.rfilename for s in info.siblings]
    sizes_by_file = {s.rfilename: s.size for s in info.siblings}
    revision = info.sha

    if selector:
        matches = sorted(fnmatch.filter(repo_files, selector))
        if not matches:
            raise FileNotFoundError(f"Selector {selector!r} matched no files in HF repo {repo!r}")

        if len(matches) > 1:
            # Sharded weights (e.g. model-00001-of-00003.gguf): pull every shard,
            # resolve to the first so file-path loaders auto-load the rest.
            logger.info(
                "Selector %r matched %d files in HF repo %r; will download all shards, resolving to first %s",
                selector,
                len(matches),
                repo,
                matches[0],
            )
            return HfSource(repo, revision, None, [selector], matches[0], _sum_sizes(matches, sizes_by_file))

        return HfSource(repo, revision, matches[0], None, None, sizes_by_file.get(matches[0]))

    # No selector: a multi-variant GGUF repo needs an explicit pick, or the
    # loader would silently resolve to the wrong quant.
    ggufs = [f for f in repo_files if f.endswith(".gguf")]
    if len(ggufs) > 1:
        raise ValueError(
            f"HF repo {repo!r} contains {len(ggufs)} GGUF variants — pick one with the `:filename` "
            f"syntax (glob supported, must match exactly one file):\n"
            f"{_format_gguf_variants(repo_files)}\n"
            f"Example: model: {repo}:*Q4_K_M.gguf"
        )

    # Single GGUF: resolve to its file path, since llama_server needs a file, not a directory.
    if len(ggufs) == 1:
        logger.info("HF repo %r has a single GGUF (%s); will resolve to its file path", repo, ggufs[0])
        return HfSource(repo, revision, ggufs[0], None, None, sizes_by_file.get(ggufs[0]))

    patterns = _select_patterns(repo_files, trust_remote_code=trust_remote_code)
    matched_files = [f for f in repo_files if _matches_any_pattern(f, patterns)]
    return HfSource(repo, revision, None, patterns, None, _sum_sizes(matched_files, sizes_by_file))


def download_hf_source(source: HfSource) -> str:
    """HF checks its own cache first, so a cached source costs no transfer and
    logs nothing — `DownloadProgress` starts on the first byte."""
    global _active_download
    progress = DownloadProgress(source.repo, source.total_bytes)
    _active_download = progress
    success = False
    try:
        if source.filename is not None:
            path = hf_hub_download(
                source.repo, source.filename, revision=source.revision, tqdm_class=_DownloadProgressLogger
            )
            success = True
            return path

        assert source.patterns is not None
        snapshot_dir = snapshot_download(
            source.repo,
            revision=source.revision,
            allow_patterns=source.patterns,
            tqdm_class=_DownloadProgressLogger,
        )
        success = True
        if source.first_shard is not None:
            return str(Path(snapshot_dir, source.first_shard).absolute())
        return snapshot_dir
    finally:
        progress.finish(success)
        _active_download = None
