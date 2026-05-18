import contextlib
from unittest.mock import MagicMock, patch

import httpx

from sieval.datasets.downloaders.url import URLHandler, _basename


def test_scheme():
    assert URLHandler().scheme == "url"


def test_download_uses_per_phase_timeout(tmp_path):
    """`read` is time-between-bytes, not a total transfer cap — so a slow
    multi-GB download doesn't get killed mid-stream. Regression: earlier
    revisions used ``timeout=60`` (a total cap), which truncated large files."""
    h = URLHandler()
    with patch("sieval.datasets.downloaders.url.httpx.stream") as mock_stream:
        mock_resp = MagicMock()
        mock_resp.iter_bytes.return_value = [b"x"]
        mock_resp.raise_for_status.return_value = None
        mock_stream.return_value.__enter__.return_value = mock_resp
        h.download("url:https://example.com/foo.csv", tmp_path, "foo", force=False)
    timeout_arg = mock_stream.call_args.kwargs.get("timeout")
    assert isinstance(timeout_arg, httpx.Timeout)
    assert timeout_arg.read == 60.0
    assert timeout_arg.connect == 30.0


def test_basename_strips_query_and_keeps_extension():
    assert _basename("https://example.com/foo.csv") == "foo.csv"
    assert _basename("https://example.com/a/b/train.jsonl.gz") == "train.jsonl.gz"


def test_basename_falls_back_for_path_only_slash():
    assert _basename("https://example.com/") == "download"


def test_download_writes_file_at_basename(tmp_path):
    """Layout: <dest>/<dataset_name>/<basename>. No hash prefix."""
    h = URLHandler()
    with patch("sieval.datasets.downloaders.url.httpx.stream") as mock_stream:
        mock_resp = MagicMock()
        mock_resp.iter_bytes.return_value = [b"hello"]
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {"content-length": "5"}
        mock_stream.return_value.__enter__.return_value = mock_resp
        h.download(
            "url:https://example.com/foo.csv",
            dest_root=tmp_path,
            dataset_name="foo",
            force=False,
        )
    target = tmp_path / "foo" / "foo.csv"
    assert target.exists()
    assert target.read_bytes() == b"hello"


def test_download_skips_when_target_exists(tmp_path):
    h = URLHandler()
    target_dir = tmp_path / "foo"
    target_dir.mkdir()
    (target_dir / "foo.csv").write_text("cached")
    with patch("sieval.datasets.downloaders.url.httpx.stream") as mock_stream:
        h.download(
            "url:https://example.com/foo.csv",
            dest_root=tmp_path,
            dataset_name="foo",
            force=False,
        )
        mock_stream.assert_not_called()


def test_download_force_redownloads(tmp_path):
    h = URLHandler()
    target_dir = tmp_path / "foo"
    target_dir.mkdir()
    (target_dir / "foo.csv").write_text("cached")
    with patch("sieval.datasets.downloaders.url.httpx.stream") as mock_stream:
        mock_resp = MagicMock()
        mock_resp.iter_bytes.return_value = [b"fresh"]
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {"content-length": "5"}
        mock_stream.return_value.__enter__.return_value = mock_resp
        h.download(
            "url:https://example.com/foo.csv",
            dest_root=tmp_path,
            dataset_name="foo",
            force=True,
        )
    assert (target_dir / "foo.csv").read_bytes() == b"fresh"


def test_download_rejects_truncated_stream(tmp_path):
    """Server returned 200 + Content-Length=100 but only streamed 3 bytes
    before the connection died. Without verification, we'd commit the
    truncated file and `is_downloaded` would claim success forever."""
    import pytest

    h = URLHandler()
    with patch("sieval.datasets.downloaders.url.httpx.stream") as mock_stream:
        mock_resp = MagicMock()
        mock_resp.iter_bytes.return_value = [b"abc"]
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {"content-length": "100"}
        mock_stream.return_value.__enter__.return_value = mock_resp
        with pytest.raises(RuntimeError, match="truncated download"):
            h.download(
                "url:https://example.com/foo.csv",
                dest_root=tmp_path,
                dataset_name="foo",
                force=False,
            )
    # Target must NOT have been renamed from .partial on truncation.
    assert not (tmp_path / "foo" / "foo.csv").exists()
    assert not (tmp_path / "foo" / "foo.csv.partial").exists()


def test_download_accepts_missing_content_length(tmp_path):
    """Chunked transfer-encoded responses often omit Content-Length. The
    truncation check must degrade gracefully rather than reject every
    chunked download."""
    h = URLHandler()
    with patch("sieval.datasets.downloaders.url.httpx.stream") as mock_stream:
        mock_resp = MagicMock()
        mock_resp.iter_bytes.return_value = [b"chunked-body"]
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {}  # no Content-Length header at all
        mock_stream.return_value.__enter__.return_value = mock_resp
        h.download(
            "url:https://example.com/foo.csv",
            dest_root=tmp_path,
            dataset_name="foo",
            force=False,
        )
    assert (tmp_path / "foo" / "foo.csv").read_bytes() == b"chunked-body"


def test_download_accepts_nonint_content_length(tmp_path):
    """A malformed Content-Length (e.g. from a broken proxy) must not crash
    the download — the check degrades to "skip" just like a missing header."""
    h = URLHandler()
    with patch("sieval.datasets.downloaders.url.httpx.stream") as mock_stream:
        mock_resp = MagicMock()
        mock_resp.iter_bytes.return_value = [b"body"]
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {"content-length": "not-a-number"}
        mock_stream.return_value.__enter__.return_value = mock_resp
        h.download(
            "url:https://example.com/foo.csv",
            dest_root=tmp_path,
            dataset_name="foo",
            force=False,
        )
    assert (tmp_path / "foo" / "foo.csv").read_bytes() == b"body"


def test_is_downloaded_false_when_payload_absent(tmp_path):
    """After manual payload deletion, is_downloaded must report false — the
    probe is the file itself, not a detached receipt."""
    h = URLHandler()
    assert not h.is_downloaded("url:https://example.com/foo.csv", tmp_path, "foo")


def test_is_downloaded_true_when_payload_present(tmp_path):
    h = URLHandler()
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "foo.csv").write_text("x")
    assert h.is_downloaded("url:https://example.com/foo.csv", tmp_path, "foo")


def test_download_failure_leaves_no_partial(tmp_path):
    """Atomic semantics: mid-stream failure removes the .partial; target file
    does not appear."""
    h = URLHandler()
    with patch("sieval.datasets.downloaders.url.httpx.stream") as mock_stream:
        mock_stream.side_effect = RuntimeError("network down")
        with contextlib.suppress(RuntimeError):
            h.download("url:https://example.com/foo.csv", tmp_path, "foo", force=False)
    assert not (tmp_path / "foo" / "foo.csv").exists()
    assert not (tmp_path / "foo" / "foo.csv.partial").exists()
