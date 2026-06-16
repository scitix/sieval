from unittest.mock import patch

import pytest

from sieval.datasets.downloaders.hf import HFHandler, HFSource, parse_hf_source


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
