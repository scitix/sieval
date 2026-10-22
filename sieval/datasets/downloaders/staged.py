"""Whether a source's staged bytes are the ones its dataset pins.

A handler's ``is_downloaded`` answers "is a finished download resident"; it
cannot tell a pinned snapshot from a directory someone else filled. This module
adds the pin comparison on top — ``hf:`` against the recorded revision,
``url:`` against the declared checksum — so ``sieval dataset download``,
readiness and ``sieval eval`` share one verdict on what "present" means.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from dataclasses import dataclass
from functools import lru_cache
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
    if not resolve(source).is_downloaded(source, dest_root, meta.name):
        return SourceProbe("absent")
    reason = _pin_mismatch_reason(meta, source, dest_root)
    return SourceProbe("stale", reason) if reason else SourceProbe("pinned")


def _pin_mismatch_reason(meta: DatasetMeta, source: str, dest_root: Path) -> str | None:
    if source.startswith("hf:"):
        return pin_mismatch(source, dest_root)
    if source.startswith("url:"):
        return _checksum_mismatch(meta, source, dest_root)
    # `local:` corpora are bring-your-own and carry no pin; a custom scheme's
    # handler owns its own notion of residency.
    return None


def staged_dir(meta: DatasetMeta, source: str, dest_root: Path) -> Path:
    """Directory *source* stages into: ``<root>/<org>/<repo>`` for ``hf:``,
    ``<root>/<dataset name>`` for every other scheme."""
    if source.startswith("hf:"):
        return dest_root / parse_hf_source(source).repo_id
    return dest_root / meta.name


def _checksum_mismatch(meta: DatasetMeta, source: str, dest_root: Path) -> str | None:
    basename = url_path_basename(source[len("url:") :]) or "download"
    expected = dict(meta.checksums).get(basename)
    if expected is None:
        # `check_datasets` requires a checksum for every registered url:
        # source; an undeclared one has nothing to compare against.
        return None
    path = dest_root / meta.name / basename
    stat = path.stat()
    actual = _sha256((path, stat.st_size, stat.st_mtime_ns))
    if actual != expected:
        return f"{basename} has {actual}, expected {expected}"
    return None


@lru_cache(maxsize=256)
def _sha256(key: tuple[Path, int, int]) -> str:
    # Keyed on (path, size, mtime) so `task list` — one probe per task,
    # several tasks per dataset — hashes a large archive once per process,
    # while a rewrite between probes still rehashes.
    return compute_sha256(key[0])
