"""Tests for sieval.datasets.downloaders.staged.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import errno
import hashlib
import os
from dataclasses import replace
from unittest.mock import patch

import pytest

from sieval.core.datasets.meta import Category, DatasetMeta, Level1Category
from sieval.datasets.downloaders import staged
from sieval.datasets.downloaders.staged import (
    SourceProbe,
    probe_source,
    record_checksums,
    staged_dir,
    stat_checksummed,
)
from sieval.datasets.downloaders.verify import compute_sha256, verify_checksums

_PIN = "a" * 40
_URL = "url:https://example.com/f.csv"
_GOOD = b"pinned-bytes"


def _meta(*sources: str) -> DatasetMeta:
    return DatasetMeta(
        name="ds",
        display_name="ds",
        description="d",
        source=sources,
        categories=(Category(Level1Category.CODE, "CodeGeneration"),),
        checksums=(("f.csv", "sha256:" + hashlib.sha256(_GOOD).hexdigest()),),
    )


def _stage_hf(dest_root, commit: str | None) -> None:
    local_dir = dest_root / "org" / "foo"
    (local_dir / "data").mkdir(parents=True)
    (local_dir / "data" / "train.parquet").write_bytes(b"x")
    if commit is not None:
        record = local_dir / ".cache/huggingface/download/data/train.parquet.metadata"
        record.parent.mkdir(parents=True)
        record.write_text(f"{commit}\netag\n1786191938.5\n")


def test_absent_when_nothing_is_staged(tmp_path):
    meta = _meta(f"hf:org/foo@{_PIN}", _URL)
    assert probe_source(meta, f"hf:org/foo@{_PIN}", tmp_path) == SourceProbe("absent")
    assert probe_source(meta, _URL, tmp_path) == SourceProbe("absent")


def test_hf_pinned_when_the_download_record_names_the_pin(tmp_path):
    _stage_hf(tmp_path, _PIN)
    src = f"hf:org/foo@{_PIN}"
    assert probe_source(_meta(src), src, tmp_path) == SourceProbe("pinned")


def test_hf_stale_when_the_directory_was_filled_by_hand(tmp_path):
    _stage_hf(tmp_path, None)
    src = f"hf:org/foo@{_PIN}"
    probe = probe_source(_meta(src), src, tmp_path)
    assert probe.status == "stale"
    assert probe.reason == (
        "data/train.parquet has no download record, so its revision is unknown"
    )


def test_url_pinned_when_the_checksum_matches(tmp_path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "f.csv").write_bytes(_GOOD)
    assert probe_source(_meta(_URL), _URL, tmp_path) == SourceProbe("pinned")


def test_url_stale_when_the_checksum_differs(tmp_path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "f.csv").write_bytes(b"other")
    probe = probe_source(_meta(_URL), _URL, tmp_path)
    assert probe.status == "stale"
    other = "sha256:" + hashlib.sha256(b"other").hexdigest()
    assert probe.reason is not None and probe.reason.startswith(f"f.csv has {other}")


def test_url_without_a_declared_checksum_is_pinned_on_presence(tmp_path):
    src = "url:https://example.com/g.csv"
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "g.csv").write_bytes(b"anything")
    assert probe_source(_meta(src), src, tmp_path) == SourceProbe("pinned")


def _stage_url(tmp_path, data: bytes = _GOOD):
    path = tmp_path / "ds" / "f.csv"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    return path


def _tree(root) -> list[tuple[str, int, int]]:
    return sorted(
        (str(p.relative_to(root)), p.lstat().st_size, p.lstat().st_mtime_ns)
        for p in root.rglob("*")
    )


@pytest.fixture(autouse=True)
def _fresh_memo():
    staged._HASHED.clear()
    yield
    staged._HASHED.clear()


@pytest.fixture
def hashes():
    with patch(f"{staged.__name__}.compute_sha256", wraps=compute_sha256) as mock:
        yield mock


def test_url_rehashes_after_the_file_is_rewritten(tmp_path, hashes):
    """The memo is keyed on size+mtime, so a replaced file is re-read."""
    path = _stage_url(tmp_path)
    assert probe_source(_meta(_URL), _URL, tmp_path).status == "pinned"
    path.write_bytes(b"replaced-bytes")
    assert probe_source(_meta(_URL), _URL, tmp_path).status == "stale"
    assert hashes.call_count == 2


def test_url_hashes_once_per_process(tmp_path, hashes):
    _stage_url(tmp_path)
    for _ in range(3):
        assert probe_source(_meta(_URL), _URL, tmp_path).status == "pinned"
    assert hashes.call_count == 1


def test_probing_writes_nothing_into_the_data_dir(tmp_path):
    _stage_url(tmp_path)
    before = _tree(tmp_path)
    assert probe_source(_meta(_URL), _URL, tmp_path).status == "pinned"
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("errno_", [errno.EROFS, errno.EACCES])
def test_url_probes_a_dir_that_refuses_every_write(tmp_path, hashes, errno_):
    """A read-only mount fails `unlink` with EROFS too, not ENOENT (#187)."""
    _stage_url(tmp_path)
    refuse = OSError(errno_, os.strerror(errno_))
    with (
        patch("pathlib.Path.mkdir", side_effect=refuse),
        patch("pathlib.Path.write_text", side_effect=refuse),
        patch("pathlib.Path.replace", side_effect=refuse),
        patch("pathlib.Path.unlink", side_effect=refuse),
    ):
        for _ in range(2):
            assert probe_source(_meta(_URL), _URL, tmp_path).status == "pinned"
    assert hashes.call_count == 1


@pytest.mark.parametrize("errno_", [errno.EROFS, errno.EACCES, errno.ENOSPC])
def test_record_checksums_never_raises_when_the_dir_refuses_writes(
    tmp_path, hashes, errno_
):
    _stage_url(tmp_path)
    refuse = OSError(errno_, os.strerror(errno_))
    with (
        patch("pathlib.Path.mkdir", side_effect=refuse),
        patch("pathlib.Path.write_text", side_effect=refuse),
        patch("pathlib.Path.unlink", side_effect=refuse),
    ):
        record_checksums(_meta(_URL), tmp_path, {"f.csv": "0 0"})
    assert list((tmp_path / "ds").iterdir()) == [tmp_path / "ds" / "f.csv"]
    assert probe_source(_meta(_URL), _URL, tmp_path).status == "pinned"
    assert hashes.call_count == 1


def _download(tmp_path, meta=None):
    """`_download_one`'s verify-and-record step."""
    meta = meta or _meta(_URL)
    verified = stat_checksummed(meta, tmp_path)
    assert verify_checksums(meta, tmp_path) == []
    record_checksums(meta, tmp_path, verified)


def test_record_checksums_skips_a_file_absent_before_verification(tmp_path):
    (tmp_path / "ds").mkdir()
    record_checksums(_meta(_URL), tmp_path, stat_checksummed(_meta(_URL), tmp_path))
    assert list((tmp_path / "ds").iterdir()) == []


def test_record_checksums_does_not_vouch_for_a_file_rewritten_after_verification(
    tmp_path,
):
    path = _stage_url(tmp_path)
    verified = stat_checksummed(_meta(_URL), tmp_path)
    path.write_bytes(b"rewritten-after-verify")
    record_checksums(_meta(_URL), tmp_path, verified)
    assert probe_source(_meta(_URL), _URL, tmp_path).status == "stale"


def test_url_trusts_its_download_record_without_hashing(tmp_path, hashes):
    _stage_url(tmp_path)
    _download(tmp_path)
    assert (tmp_path / "ds/.sieval/download/f.csv.sha256").is_file()
    assert probe_source(_meta(_URL), _URL, tmp_path).status == "pinned"
    assert hashes.call_count == 0


def test_url_record_of_an_older_pin_reads_stale(tmp_path, hashes):
    """The record holds a digest, not a verdict, so a re-pin is caught."""
    _stage_url(tmp_path)
    _download(tmp_path)
    repinned = replace(_meta(_URL), checksums=(("f.csv", "sha256:" + "0" * 64),))
    assert probe_source(repinned, _URL, tmp_path).status == "stale"
    assert hashes.call_count == 0


def test_url_ignores_a_record_with_a_stale_key(tmp_path, hashes):
    path = _stage_url(tmp_path)
    _download(tmp_path)
    path.write_bytes(b"other")
    assert probe_source(_meta(_URL), _URL, tmp_path).status == "stale"
    assert hashes.call_count == 1


def test_hf_stale_when_the_dir_holds_only_non_data_files(tmp_path):
    """`is_downloaded` misses them, but a fetch into the dir would keep them."""
    local_dir = tmp_path / "org" / "foo"
    local_dir.mkdir(parents=True)
    (local_dir / "old_copy.jsonl.gz").write_bytes(b"x")
    src = f"hf:org/foo@{_PIN}"
    probe = probe_source(_meta(src), src, tmp_path)
    assert probe == SourceProbe(
        "stale", "old_copy.jsonl.gz has no download record, so its revision is unknown"
    )


def test_hf_absent_for_an_empty_repo_dir(tmp_path):
    (tmp_path / "org" / "foo").mkdir(parents=True)
    src = f"hf:org/foo@{_PIN}"
    assert probe_source(_meta(src), src, tmp_path) == SourceProbe("absent")


def test_local_source_is_pinned_on_presence(tmp_path):
    src = "local:pg/essays.json"
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "essays.json").write_text("{}")
    assert probe_source(_meta(src), src, tmp_path) == SourceProbe("pinned")


def test_unknown_scheme_raises(tmp_path):
    with pytest.raises(NotImplementedError):
        probe_source(_meta("hf:org/foo"), "s3://bucket/x", tmp_path)


def test_staged_dir_per_scheme(tmp_path):
    meta = _meta(f"hf:org/foo@{_PIN}", _URL)
    assert staged_dir(meta, f"hf:org/foo@{_PIN}", tmp_path) == tmp_path / "org/foo"
    assert staged_dir(meta, _URL, tmp_path) == tmp_path / "ds"
