"""Whether a source's staged bytes are the ones its dataset pins.

``is_downloaded`` only says a download is resident. This adds the pin check —
``hf:`` against the recorded revision, ``url:`` against the declared checksum —
so ``download``, readiness and ``eval`` share one verdict.

A ``url:`` archive is checked, not what a loader extracts from it.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sieval.core.datasets.meta import DatasetMeta, url_path_basename
from sieval.datasets.downloaders.base import resolve
from sieval.datasets.downloaders.hf import parse_hf_source, pin_mismatch
from sieval.datasets.downloaders.verify import compute_sha256

SourceStatus = Literal["absent", "pinned", "stale"]


@dataclass(frozen=True, slots=True)
class SourceProbe:
    """``stale`` = resident but not provably the pinned bytes; ``reason`` says why."""

    status: SourceStatus
    reason: str | None = None


def probe_source(meta: DatasetMeta, source: str, dest_root: Path) -> SourceProbe:
    """Classify *source* of *meta* as staged under *dest_root*.

    Raises ``NotImplementedError`` for an unregistered scheme, like ``resolve``.
    """
    if resolve(source).is_downloaded(source, dest_root, meta.name):
        reason = _pin_mismatch_reason(meta, source, dest_root)
        return SourceProbe("stale", reason) if reason else SourceProbe("pinned")
    # `is_downloaded` looks for data suffixes only; a dir holding other files
    # is still occupied, and a fetch into it would keep them.
    if source.startswith("hf:") and (reason := pin_mismatch(source, dest_root)):
        return SourceProbe("stale", reason)
    return SourceProbe("absent")


def _pin_mismatch_reason(meta: DatasetMeta, source: str, dest_root: Path) -> str | None:
    if source.startswith("hf:"):
        return pin_mismatch(source, dest_root)
    if source.startswith("url:"):
        return _checksum_mismatch(meta, source, dest_root)
    return None  # `local:` is bring-your-own and carries no pin


def staged_dir(meta: DatasetMeta, source: str, dest_root: Path) -> Path:
    """``<root>/<org>/<repo>`` for ``hf:``, ``<root>/<dataset name>`` otherwise."""
    if source.startswith("hf:"):
        return dest_root / parse_hf_source(source).repo_id
    return dest_root / meta.name


def _checksum_mismatch(meta: DatasetMeta, source: str, dest_root: Path) -> str | None:
    basename = url_path_basename(source[len("url:") :]) or "download"
    expected = dict(meta.checksums).get(basename)
    if expected is None:
        return None  # `check_datasets` requires one for every registered source
    actual = _cached_sha256(dest_root / meta.name / basename)
    if actual != expected:
        return f"{basename} has {actual}, expected {expected}"
    return None


def _cached_sha256(path: Path) -> str:
    """``compute_sha256``, memoized in a ``.<name>.sha256`` sidecar.

    Keyed on size + mtime, so readiness does not reread a multi-GB archive per
    CLI call and a rewrite still rehashes. Best effort: a read-only data dir
    just hashes every time.
    """
    stat = path.stat()
    key = f"{stat.st_size} {stat.st_mtime_ns}"
    sidecar = path.with_name(f".{path.name}.sha256")
    try:
        cached_key, _, digest = sidecar.read_text().rpartition(" ")
        if cached_key == key and digest:
            return digest
    except OSError:
        pass
    digest = compute_sha256(path)
    tmp = sidecar.with_name(f"{sidecar.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(f"{key} {digest}")
        tmp.replace(sidecar)
    except OSError:
        tmp.unlink(missing_ok=True)
    return digest
