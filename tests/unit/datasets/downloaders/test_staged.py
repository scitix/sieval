"""Tests for sieval.datasets.downloaders.staged.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import hashlib
from unittest.mock import patch

import pytest

from sieval.core.datasets.meta import Category, DatasetMeta, Level1Category
from sieval.datasets.downloaders.staged import (
    SourceProbe,
    probe_source,
    staged_dir,
)
from sieval.datasets.downloaders.verify import compute_sha256

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


def test_url_rehashes_after_the_file_is_rewritten(tmp_path):
    """The sidecar is keyed on size+mtime, so a replaced file is re-read."""
    path = tmp_path / "ds" / "f.csv"
    path.parent.mkdir()
    path.write_bytes(_GOOD)
    assert probe_source(_meta(_URL), _URL, tmp_path).status == "pinned"
    path.write_bytes(b"replaced-bytes")
    assert probe_source(_meta(_URL), _URL, tmp_path).status == "stale"


def test_url_hashes_once_across_probes(tmp_path):
    """The sidecar outlives the process, unlike an in-memory cache."""
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "f.csv").write_bytes(_GOOD)
    with patch(
        "sieval.datasets.downloaders.staged.compute_sha256", wraps=compute_sha256
    ) as mock_hash:
        for _ in range(3):
            assert probe_source(_meta(_URL), _URL, tmp_path).status == "pinned"
    assert mock_hash.call_count == 1
    assert (tmp_path / "ds" / ".f.csv.sha256").is_file()


def test_url_hashes_every_time_in_a_read_only_dir(tmp_path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "f.csv").write_bytes(_GOOD)
    with (
        patch("pathlib.Path.write_text", side_effect=PermissionError),
        patch(
            "sieval.datasets.downloaders.staged.compute_sha256", wraps=compute_sha256
        ) as mock_hash,
    ):
        for _ in range(2):
            assert probe_source(_meta(_URL), _URL, tmp_path).status == "pinned"
    assert mock_hash.call_count == 2
    assert list((tmp_path / "ds").iterdir()) == [tmp_path / "ds" / "f.csv"]


def test_url_ignores_a_sidecar_with_a_stale_key(tmp_path):
    """A sidecar copied alongside other bytes is not trusted."""
    path = tmp_path / "ds" / "f.csv"
    path.parent.mkdir()
    path.write_bytes(b"other")
    good = "sha256:" + hashlib.sha256(_GOOD).hexdigest()
    (tmp_path / "ds" / ".f.csv.sha256").write_text(f"12 0 {good}")
    assert probe_source(_meta(_URL), _URL, tmp_path).status == "stale"


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
