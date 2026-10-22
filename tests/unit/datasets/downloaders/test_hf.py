from unittest.mock import patch

import pytest

from sieval.datasets.downloaders.hf import (
    HFHandler,
    HFSource,
    parse_hf_source,
    pin_mismatch,
)


def test_scheme():
    assert HFHandler().scheme == "hf"


def test_parse_hf_source_without_revision():
    assert parse_hf_source("hf:org/foo") == HFSource(repo_id="org/foo", revision=None)


def test_parse_hf_source_with_revision():
    assert parse_hf_source("hf:org/foo@abc123") == HFSource(
        repo_id="org/foo", revision="abc123"
    )


@pytest.mark.parametrize(
    "source",
    [
        "url:https://example.com/x",
        "hf:",
        "hf:@abc123",
        "hf:org/foo@",
    ],
)
def test_parse_hf_source_rejects_invalid_source(source):
    with pytest.raises(ValueError):
        parse_hf_source(source)


def test_download_invokes_snapshot_download_with_local_dir(tmp_path):
    """The HF handler mirrors the repo via huggingface_hub.snapshot_download
    (config-agnostic file copy), not datasets.load_dataset — this is what
    fixes multi-config repos (e.g. opencompass/AIME2025) and script-based
    repos (e.g. livecodebench/code_generation_lite) that load_dataset
    refused without `name=` / `trust_remote_code=True`. Lands at
    `{dest_root}/<org>/<name>/` so the runtime side can resolve the bare
    repo_id to an on-disk path by string concat."""
    h = HFHandler()
    with patch("huggingface_hub.snapshot_download") as mock_snap:
        mock_snap.return_value = str(tmp_path / "org" / "foo")
        h.download("hf:org/foo", dest_root=tmp_path, dataset_name="foo", force=False)
    call = mock_snap.call_args
    assert call.kwargs["repo_id"] == "org/foo"
    assert call.kwargs["repo_type"] == "dataset"
    assert call.kwargs["revision"] is None
    assert call.kwargs["local_dir"] == str(tmp_path / "org" / "foo")
    assert call.kwargs["force_download"] is False
    # `cache_dir=` must not be passed — local_dir mode is the contract.
    assert "cache_dir" not in call.kwargs


def test_download_invokes_snapshot_download_with_revision(tmp_path):
    h = HFHandler()
    with patch("huggingface_hub.snapshot_download") as mock_snap:
        mock_snap.return_value = str(tmp_path / "org" / "foo")
        h.download(
            "hf:org/foo@abc123", dest_root=tmp_path, dataset_name="foo", force=False
        )
    call = mock_snap.call_args
    assert call.kwargs["repo_id"] == "org/foo"
    assert call.kwargs["revision"] == "abc123"
    assert call.kwargs["local_dir"] == str(tmp_path / "org" / "foo")


def test_force_maps_to_force_download(tmp_path):
    h = HFHandler()
    with patch("huggingface_hub.snapshot_download") as mock_snap:
        mock_snap.return_value = str(tmp_path / "org" / "foo")
        h.download("hf:org/foo", tmp_path, "foo", force=True)
    assert mock_snap.call_args.kwargs["force_download"] is True


def test_is_downloaded_false_when_local_dir_missing(tmp_path):
    h = HFHandler()
    assert not h.is_downloaded("hf:org/foo", tmp_path, "foo")


def test_is_downloaded_false_when_dir_empty(tmp_path):
    """Partial-download regression: huggingface_hub creates the outer dir
    very early. The naive `dir.exists()` probe we used to do returned
    True even when no data had been written."""
    h = HFHandler()
    (tmp_path / "org" / "foo").mkdir(parents=True)
    assert not h.is_downloaded("hf:org/foo", tmp_path, "foo")


def test_is_downloaded_false_when_incomplete_sidecar_present(tmp_path):
    """huggingface_hub writes `<file>.incomplete` under `.cache/` during
    active transfers and removes them on success. Presence = aborted
    mid-stream — treat as not downloaded so a retry re-fetches."""
    h = HFHandler()
    local_dir = tmp_path / "org" / "foo"
    local_dir.mkdir(parents=True)
    (local_dir / "data.parquet").write_bytes(b"partial")
    download_dir = local_dir / ".cache" / "huggingface" / "download"
    download_dir.mkdir(parents=True)
    (download_dir / "data.parquet.incomplete").write_bytes(b"")
    assert not h.is_downloaded("hf:org/foo", tmp_path, "foo")


def test_is_downloaded_true_with_parquet_payload(tmp_path):
    h = HFHandler()
    local_dir = tmp_path / "org" / "foo"
    local_dir.mkdir(parents=True)
    (local_dir / "data.parquet").write_bytes(b"x")
    assert h.is_downloaded("hf:org/foo", tmp_path, "foo")


def test_is_downloaded_strips_revision_from_local_dir(tmp_path):
    h = HFHandler()
    local_dir = tmp_path / "org" / "foo"
    local_dir.mkdir(parents=True)
    (local_dir / "data.parquet").write_bytes(b"x")
    assert h.is_downloaded("hf:org/foo@abc123", tmp_path, "foo")


def test_is_downloaded_true_with_arrow_payload(tmp_path):
    h = HFHandler()
    local_dir = tmp_path / "org" / "foo"
    local_dir.mkdir(parents=True)
    (local_dir / "shard.arrow").write_bytes(b"x")
    assert h.is_downloaded("hf:org/foo", tmp_path, "foo")


def test_is_downloaded_true_with_jsonl_payload(tmp_path):
    """livecodebench-shaped repos ship .jsonl; the probe must accept them
    the same as parquet."""
    h = HFHandler()
    local_dir = tmp_path / "org" / "foo"
    local_dir.mkdir(parents=True)
    (local_dir / "test.jsonl").write_bytes(b"{}")
    assert h.is_downloaded("hf:org/foo", tmp_path, "foo")


def test_is_downloaded_ignores_cache_subtree(tmp_path):
    """`.cache/huggingface/` may contain `.metadata` files that match
    `_DATA_SUFFIXES` lexically; they don't count as data payloads."""
    h = HFHandler()
    local_dir = tmp_path / "org" / "foo"
    (local_dir / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (local_dir / ".cache" / "huggingface" / "download" / "stub.json").write_bytes(b"{}")
    assert not h.is_downloaded("hf:org/foo", tmp_path, "foo")


def test_is_downloaded_handles_multi_level_repo_id(tmp_path):
    """Multi-level repo paths (e.g. HuggingFaceH4/aime_2024) land at the
    same `<org>/<name>/` layout."""
    h = HFHandler()
    local_dir = tmp_path / "HuggingFaceH4" / "aime_2024"
    local_dir.mkdir(parents=True)
    (local_dir / "train.parquet").write_bytes(b"x")
    assert h.is_downloaded("hf:HuggingFaceH4/aime_2024", tmp_path, "aime_2024")


def test_download_creates_local_dir(tmp_path):
    h = HFHandler()
    local_dir = tmp_path / "org" / "foo"
    assert not local_dir.exists()
    with patch("huggingface_hub.snapshot_download") as mock_snap:
        mock_snap.return_value = str(local_dir)
        h.download("hf:org/foo", tmp_path, "foo", force=False)
    assert local_dir.is_dir()


def test_download_rejects_non_hf_scheme(tmp_path):
    h = HFHandler()
    with pytest.raises(ValueError, match="Expected hf: scheme"):
        h.download("url:https://example.com/x", tmp_path, "x", force=False)


def test_is_downloaded_rejects_non_hf_scheme(tmp_path):
    h = HFHandler()
    with pytest.raises(ValueError, match="Expected hf: scheme"):
        h.is_downloaded("url:https://example.com/x", tmp_path, "x")


# ── pin_mismatch ──────────────────────────────────────────────────────

_PIN = "a" * 40


def _stage(local_dir, rel, commit):
    """Stage *rel* with the download record `snapshot_download` writes."""
    (local_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (local_dir / rel).write_bytes(b"x")
    if commit is not None:
        record = local_dir / ".cache" / "huggingface" / "download" / f"{rel}.metadata"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(f"{commit}\netag\n1786191938.5\n")


def test_pin_mismatch_none_when_every_file_records_the_pin(tmp_path):
    local_dir = tmp_path / "org" / "foo"
    _stage(local_dir, "data/train-00000-of-00001.parquet", _PIN)
    _stage(local_dir, "README.md", _PIN)
    assert pin_mismatch(f"hf:org/foo@{_PIN}", tmp_path) is None


def test_pin_mismatch_flags_a_file_with_no_download_record(tmp_path):
    """The #179 shape: a hand-placed `train.jsonl` that `is_downloaded`
    accepts because `.jsonl` is a data suffix."""
    local_dir = tmp_path / "org" / "foo"
    _stage(local_dir, "train.jsonl", None)
    assert h_is_downloaded(tmp_path)
    assert pin_mismatch(f"hf:org/foo@{_PIN}", tmp_path) == (
        "train.jsonl has no download record, so its revision is unknown"
    )


def test_pin_mismatch_flags_a_stray_file_beside_a_pinned_snapshot(tmp_path):
    """A leftover next to pinned files: `load_dataset(dir)` reads both."""
    local_dir = tmp_path / "org" / "foo"
    _stage(local_dir, "data/train.parquet", _PIN)
    _stage(local_dir, "train.jsonl", None)
    reason = pin_mismatch(f"hf:org/foo@{_PIN}", tmp_path)
    assert reason is not None and reason.startswith("train.jsonl ")


def test_pin_mismatch_flags_a_snapshot_of_another_revision(tmp_path):
    local_dir = tmp_path / "org" / "foo"
    _stage(local_dir, "data/train.parquet", "b" * 40)
    assert pin_mismatch(f"hf:org/foo@{_PIN}", tmp_path) == (
        f"data/train.parquet is from revision {'b' * 40}, expected {_PIN}"
    )


def test_pin_mismatch_flags_an_empty_download_record(tmp_path):
    local_dir = tmp_path / "org" / "foo"
    _stage(local_dir, "data/train.parquet", "")
    reason = pin_mismatch(f"hf:org/foo@{_PIN}", tmp_path)
    assert reason is not None and "no download record" in reason


def test_pin_mismatch_unpinned_source_accepts_any_recorded_revision(tmp_path):
    local_dir = tmp_path / "org" / "foo"
    _stage(local_dir, "data/train.parquet", "b" * 40)
    assert pin_mismatch("hf:org/foo", tmp_path) is None


def test_pin_mismatch_ignores_files_under_cache(tmp_path):
    """`.cache/` holds the records themselves, which have no records."""
    local_dir = tmp_path / "org" / "foo"
    _stage(local_dir, "data/train.parquet", _PIN)
    (local_dir / ".cache" / "huggingface" / ".gitignore").write_text("*")
    assert pin_mismatch(f"hf:org/foo@{_PIN}", tmp_path) is None


def test_pin_mismatch_flags_a_downloaded_file_since_deleted(tmp_path):
    """A record with no file: `load_dataset(dir)` would load fewer rows."""
    local_dir = tmp_path / "org" / "foo"
    _stage(local_dir, "data/train-00000.parquet", _PIN)
    _stage(local_dir, "data/train-00001.parquet", _PIN)
    (local_dir / "data/train-00001.parquet").unlink()
    assert pin_mismatch(f"hf:org/foo@{_PIN}", tmp_path) == (
        "data/train-00001.parquet was downloaded but is missing"
    )


def test_pin_mismatch_none_for_an_absent_repo(tmp_path):
    assert pin_mismatch(f"hf:org/foo@{_PIN}", tmp_path) is None


def h_is_downloaded(dest_root):
    return HFHandler().is_downloaded(f"hf:org/foo@{_PIN}", dest_root, "foo")


def test_force_clears_files_snapshot_download_would_leave_behind(tmp_path):
    """`snapshot_download` keeps files it does not write, so a forced refetch
    must start from an empty directory."""
    local_dir = tmp_path / "org" / "foo"
    _stage(local_dir, "train.jsonl", None)
    with patch("huggingface_hub.snapshot_download"):
        HFHandler().download(f"hf:org/foo@{_PIN}", tmp_path, "foo", force=True)
    assert local_dir.is_dir()
    assert not (local_dir / "train.jsonl").exists()


def test_download_without_force_keeps_existing_files(tmp_path):
    local_dir = tmp_path / "org" / "foo"
    _stage(local_dir, "train.jsonl", None)
    with patch("huggingface_hub.snapshot_download"):
        HFHandler().download(f"hf:org/foo@{_PIN}", tmp_path, "foo", force=False)
    assert (local_dir / "train.jsonl").exists()
