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

import os
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
            # `snapshot_download` keeps files it did not write.
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

    ``snapshot_download(local_dir=...)`` records each file as
    ``.cache/huggingface/download/<path>.metadata``, first line the commit.
    Files and records must pair up, each naming the pin. An in-place edit of a
    downloaded file goes unnoticed.

    Parsed here, not via ``huggingface_hub._local_folder``: its reader takes
    locks and deletes records it cannot parse.
    """
    parsed = parse_hf_source(source)
    local_dir = dest_root / parsed.repo_id
    record_dir = local_dir / ".cache" / "huggingface" / "download"
    files = _relative_files(local_dir, skip=".cache")
    records = {
        rel.removesuffix(".metadata")
        for rel in _relative_files(record_dir)
        if rel.endswith(".metadata")
    }
    if deleted := records - files:
        return f"{min(deleted)} was downloaded but is missing"
    for rel in sorted(files):
        try:
            commit = (record_dir / f"{rel}.metadata").read_text().partition("\n")[0]
        except OSError:
            commit = ""
        if not commit:
            return f"{rel} has no download record, so its revision is unknown"
        if parsed.revision is not None and commit != parsed.revision:
            return f"{rel} is from revision {commit}, expected {parsed.revision}"
    return None


def _relative_files(root: Path, skip: str | None = None) -> set[str]:
    """POSIX paths of the files under *root*, pruning the top-level *skip* dir.

    ``os.walk`` rather than ``rglob``: a 1k-file repo walks in ~1 ms, not ~90.
    """
    out: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        if skip is not None and dirpath == str(root):
            dirnames[:] = [d for d in dirnames if d != skip]
        rel = Path(dirpath).relative_to(root).as_posix()
        out.update(f if rel == "." else f"{rel}/{f}" for f in filenames)
    return out
