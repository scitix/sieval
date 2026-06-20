"""Content-checksum verification for staged dataset files.

Pure: hashes files under ``dest_root/<dataset_name>/`` and compares to
``DatasetMeta.checksums``. Side effects (deleting a bad file, raising) are the
caller's responsibility, so this stays reusable from read-only contexts.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from sieval.core.datasets.meta import DatasetMeta

_CHUNK = 1 << 20


def compute_sha256(path: Path) -> str:
    """``sha256:<hex>`` of *path*'s bytes, streamed in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


@dataclass(frozen=True)
class ChecksumMismatch:
    basename: str
    expected: str
    actual: str | None  # None => declared file is missing


def verify_checksums(meta: DatasetMeta, dest_root: Path) -> list[ChecksumMismatch]:
    """Compare each declared checksum to the staged file; return mismatches.

    Empty ``meta.checksums`` (all-``hf:`` datasets) returns ``[]``.
    """
    mismatches: list[ChecksumMismatch] = []
    for basename, expected in meta.checksums:
        path = dest_root / meta.name / basename
        if not path.is_file():
            mismatches.append(ChecksumMismatch(basename, expected, None))
            continue
        actual = compute_sha256(path)
        if actual != expected:
            mismatches.append(ChecksumMismatch(basename, expected, actual))
    return mismatches
