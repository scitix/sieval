"""Tests for the local: source handler (bring-your-own corpus).

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import pytest

from sieval.datasets.downloaders.local import (
    LocalHandler,
    LocalSourceUnavailable,
    _basename,
)


def test_scheme():
    assert LocalHandler().scheme == "local"


def test_strip_scheme_rejects_wrong_scheme():
    with pytest.raises(ValueError, match="Expected local: scheme"):
        LocalHandler._strip_scheme("url:https://example.com/foo.json")


def test_basename():
    assert _basename("pg/PaulGrahamEssays.json.gz") == "PaulGrahamEssays.json.gz"
    assert _basename("trailing/") == "download"


def test_download_noop_when_present(tmp_path):
    """BYO: an already-staged corpus is a no-op, even with force (nothing to
    re-fetch), and never touches the bytes."""
    target_dir = tmp_path / "ruler"
    target_dir.mkdir()
    (target_dir / "PaulGrahamEssays.json.gz").write_bytes(b"corpus")
    h = LocalHandler()
    for force in (False, True):
        h.download(
            "local:pg/PaulGrahamEssays.json.gz",
            dest_root=tmp_path,
            dataset_name="ruler",
            force=force,
        )
    assert (target_dir / "PaulGrahamEssays.json.gz").read_bytes() == b"corpus"


def test_download_raises_with_instructions_when_missing(tmp_path):
    """BYO: an absent corpus is not silently fetched — it raises with the
    expected staging path so the caller can tell the user how to produce it."""
    h = LocalHandler()
    with pytest.raises(LocalSourceUnavailable, match="bring-your-own") as exc:
        h.download(
            "local:pg/PaulGrahamEssays.json.gz",
            dest_root=tmp_path,
            dataset_name="ruler",
            force=False,
        )
    assert str(tmp_path / "ruler" / "PaulGrahamEssays.json.gz") in str(exc.value)


def test_is_downloaded(tmp_path):
    h = LocalHandler()
    assert not h.is_downloaded("local:pg/PaulGrahamEssays.json.gz", tmp_path, "ruler")
    target_dir = tmp_path / "ruler"
    target_dir.mkdir()
    (target_dir / "PaulGrahamEssays.json.gz").write_text("x")
    assert h.is_downloaded("local:pg/PaulGrahamEssays.json.gz", tmp_path, "ruler")
