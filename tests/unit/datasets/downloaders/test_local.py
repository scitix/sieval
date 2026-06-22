"""Tests for the local: source handler.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

from unittest.mock import patch

import pytest

from sieval.datasets.downloaders.local import LocalHandler, _basename


def test_scheme():
    assert LocalHandler().scheme == "local"


def test_strip_scheme_rejects_wrong_scheme():
    with pytest.raises(ValueError, match="Expected local: scheme"):
        LocalHandler._strip_scheme("url:https://example.com/foo.json")


def test_basename():
    assert _basename("pg/PaulGrahamEssays.json.gz") == "PaulGrahamEssays.json.gz"
    assert _basename("trailing/") == "download"


@pytest.mark.parametrize(
    "bad",
    ["", "/abs/path.json", "../escape.json", "a/../../b.json", "a/./b.json"],
)
def test_bundled_path_rejects_traversal(bad):
    """`local:` may only read normalized, package-relative paths — an absolute
    path or a `..` segment that escapes the bundled `_data/` root is a hard
    error, never a silently-resolved path."""
    with pytest.raises(ValueError, match="package-relative"):
        LocalHandler._bundled_path(bad)


def test_download_copies_to_basename(tmp_path):
    """Layout: <dest>/<dataset_name>/<basename>, copied from the bundled file."""
    src = tmp_path / "bundled.json"
    src.write_text("payload")
    h = LocalHandler()
    with patch.object(LocalHandler, "_bundled_path", return_value=src):
        h.download(
            "local:pg/bundled.json",
            dest_root=tmp_path,
            dataset_name="pg",
            force=False,
        )
    target = tmp_path / "pg" / "bundled.json"
    assert target.read_text() == "payload"


def test_download_skips_when_target_exists(tmp_path):
    src = tmp_path / "bundled.json"
    src.write_text("fresh")
    target_dir = tmp_path / "pg"
    target_dir.mkdir()
    (target_dir / "bundled.json").write_text("cached")
    h = LocalHandler()
    with patch.object(LocalHandler, "_bundled_path", return_value=src):
        h.download(
            "local:pg/bundled.json",
            dest_root=tmp_path,
            dataset_name="pg",
            force=False,
        )
    assert (target_dir / "bundled.json").read_text() == "cached"


def test_download_force_recopies(tmp_path):
    src = tmp_path / "bundled.json"
    src.write_text("fresh")
    target_dir = tmp_path / "pg"
    target_dir.mkdir()
    (target_dir / "bundled.json").write_text("cached")
    h = LocalHandler()
    with patch.object(LocalHandler, "_bundled_path", return_value=src):
        h.download(
            "local:pg/bundled.json",
            dest_root=tmp_path,
            dataset_name="pg",
            force=True,
        )
    assert (target_dir / "bundled.json").read_text() == "fresh"


def test_is_downloaded(tmp_path):
    h = LocalHandler()
    assert not h.is_downloaded("local:pg/bundled.json", tmp_path, "pg")
    target_dir = tmp_path / "pg"
    target_dir.mkdir()
    (target_dir / "bundled.json").write_text("x")
    assert h.is_downloaded("local:pg/bundled.json", tmp_path, "pg")
