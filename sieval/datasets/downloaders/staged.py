"""Whether a source's staged bytes are the ones its dataset pins.

``is_downloaded`` only says a download is resident. This adds the pin check —
``hf:`` against the recorded revision, ``url:`` against the declared checksum —
so ``download``, readiness and ``eval`` share one verdict.

A ``url:`` archive is checked, not what a loader extracts from it. Probes never
write (the data dir may be read-only or shared); ``download`` records each
verified digest so they can skip the hash.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import contextlib
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
    actual = _staged_sha256(dest_root / meta.name / basename)
    if actual != expected:
        return f"{basename} has {actual}, expected {expected}"
    return None


def stat_checksummed(meta: DatasetMeta, dest_root: Path) -> dict[str, str]:
    """Size + mtime of each declared file present. Take it before verifying, so
    a file rewritten after verification is not recorded."""
    keys: dict[str, str] = {}
    for basename, _ in meta.checksums:
        with contextlib.suppress(OSError):
            keys[basename] = _stat_key(dest_root / meta.name / basename)
    return keys


def record_checksums(
    meta: DatasetMeta, dest_root: Path, verified: dict[str, str]
) -> None:
    """Record the digest of each file in *verified* (``stat_checksummed``) for
    probes to trust. Best effort: an unwritable record only costs a rehash."""
    for basename, expected in meta.checksums:
        if basename not in verified:
            continue
        record = _record_path(dest_root / meta.name / basename)
        tmp = record.with_name(f"{record.name}.{os.getpid()}.tmp")
        try:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(f"{verified[basename]} {expected}\n")
            tmp.replace(record)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)


def _staged_sha256(path: Path) -> str:
    """*path*'s digest from its download record if size + mtime still match,
    else hashed once per process. An edit keeping both goes unnoticed, as with
    ``hf:`` records."""
    key = _stat_key(path)
    try:
        recorded_key, _, digest = _record_path(path).read_text().strip().rpartition(" ")
    except OSError:
        recorded_key, digest = "", ""
    if recorded_key == key and digest.startswith("sha256:"):
        return digest
    memo_key = (str(path.resolve()), key)
    if (digest := _HASHED.get(memo_key)) is None:
        digest = _HASHED[memo_key] = compute_sha256(path)
    return digest


_HASHED: dict[tuple[str, str], str] = {}


def _stat_key(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size} {stat.st_mtime_ns}"


def _record_path(path: Path) -> Path:
    """Like ``hf:``'s ``.cache/huggingface/download/<file>.metadata``."""
    return path.parent / ".sieval" / "download" / f"{path.name}.sha256"
