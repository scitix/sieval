"""HF scheme handler: mirror a repo via ``huggingface_hub.snapshot_download``.

File-level mirror rather than ``datasets.load_dataset`` so multi-config repos
(``opencompass/AIME2025``) and script-based repos (``livecodebench/...``) that
``load_dataset`` refuses to open config-blind still download end-to-end.

Lands at ``{dest_root}/<org>/<name>/`` as plain files
(``snapshot_download(local_dir=...)`` mode, not ``cache_dir=...``). Matches the
url-scheme convention ``{dest_root}/<dataset_name>/`` so the runtime side can
resolve a bare HF repo_id to its on-disk path by string concat — no hub-cache
indirection, no offline/online split inside ``datasets.load_dataset``.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

# Payload suffixes found across pilot datasets; extend as other formats surface.
_DATA_SUFFIXES = (".arrow", ".parquet", ".jsonl", ".json", ".csv")


@dataclass(frozen=True)
class HFSource:
    repo_id: str
    revision: str | None


def parse_hf_source(source: str) -> HFSource:
    if not source.startswith("hf:"):
        raise ValueError(f"Expected hf: scheme, got {source!r}")
    repo_ref = source[len("hf:") :]
    if not repo_ref:
        raise ValueError(f"Invalid hf source: {source!r}")
    repo_id, separator, revision = repo_ref.rpartition("@")
    if not separator:
        return HFSource(repo_id=repo_ref, revision=None)
    if not repo_id or not revision:
        raise ValueError(f"Invalid hf source revision pin: {source!r}")
    return HFSource(repo_id=repo_id, revision=revision)


class HFHandler:
    scheme = "hf"

    def download(
        self,
        source: str,
        dest_root: Path,
        dataset_name: str,
        force: bool,
    ) -> None:
        from huggingface_hub import snapshot_download

        parsed = parse_hf_source(source)
        local_dir = dest_root / parsed.repo_id
        if force and local_dir.exists():
            # `snapshot_download` only (re)writes the files it fetches, so a
            # file from another revision or a hand-placed copy would survive
            # next to the pinned ones and `load_dataset(dir)` would read both.
            # A forced refetch starts from an empty directory.
            shutil.rmtree(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=parsed.repo_id,
            repo_type="dataset",
            revision=parsed.revision,
            local_dir=str(local_dir),
            force_download=force,
        )

    def is_downloaded(
        self,
        source: str,
        dest_root: Path,
        dataset_name: str,
    ) -> bool:
        parsed = parse_hf_source(source)
        local_dir = dest_root / parsed.repo_id
        if not local_dir.exists():
            return False
        # `huggingface_hub` writes `.incomplete` sidecars under
        # `<local_dir>/.cache/huggingface/download/` during active transfers
        # and removes them on success. Any present = aborted mid-stream.
        if any(local_dir.rglob("*.incomplete")):
            return False
        # Require at least one actual data payload outside the internal
        # `.cache/` bookkeeping; metadata stubs land early and aren't proof
        # the download finished.
        cache_dir = local_dir / ".cache"
        for p in local_dir.rglob("*"):
            if not p.is_file():
                continue
            if cache_dir in p.parents:
                continue
            if p.suffix in _DATA_SUFFIXES:
                return True
        return False


def pin_mismatch(source: str, dest_root: Path) -> str | None:
    """Why the staged repo is not the pinned revision; ``None`` if it is.

    ``snapshot_download(local_dir=...)`` records each file it writes as
    ``.cache/huggingface/download/<path>.metadata``, whose first line is the
    commit hash it came from. Every file outside ``.cache/`` must have such a
    record naming the pinned commit: a file with no record was not written by a
    download (hand-placed, or left behind by an older layout), and
    ``load_dataset(dir)`` reads it all the same.

    This proves which revision each file was fetched from, not that its bytes
    are untouched since — editing a downloaded file in place goes unnoticed.

    Parsed here rather than via ``huggingface_hub._local_folder``, whose reader
    creates lock files and deletes records it cannot parse — side effects a
    read-only probe must not have.
    """
    parsed = parse_hf_source(source)
    local_dir = dest_root / parsed.repo_id
    cache_dir = local_dir / ".cache"
    record_dir = cache_dir / "huggingface" / "download"
    files = sorted(
        p for p in local_dir.rglob("*") if p.is_file() and cache_dir not in p.parents
    )
    for p in files:
        rel = p.relative_to(local_dir).as_posix()
        try:
            commit = (record_dir / f"{rel}.metadata").read_text().partition("\n")[0]
        except OSError:
            commit = ""
        if not commit:
            return f"{rel} has no download record, so its revision is unknown"
        if parsed.revision is not None and commit != parsed.revision:
            return f"{rel} is from revision {commit}, expected {parsed.revision}"
    return None
