"""URL scheme handler: HTTP(S) download into ``dest_root/<dataset_name>/``.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

from pathlib import Path

import httpx

from sieval.core.datasets.meta import url_path_basename

# `read` is time-between-bytes, not total transfer time — slow-but-steady
# multi-GB downloads don't trip it; stalled connections do.
_TIMEOUT = httpx.Timeout(connect=30.0, read=60.0, write=60.0, pool=30.0)


class URLHandler:
    scheme = "url"

    def download(
        self,
        source: str,
        dest_root: Path,
        dataset_name: str,
        force: bool,
    ) -> None:
        url = self._strip_scheme(source)
        target_dir = dest_root / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / _basename(url)
        if target.exists() and not force:
            return
        tmp = target.with_name(target.name + ".partial")
        try:
            with httpx.stream("GET", url, timeout=_TIMEOUT, follow_redirects=True) as r:
                r.raise_for_status()
                # Catches 2xx responses that dropped the connection mid-stream.
                # Compare against on-the-wire bytes, not bytes written: when the
                # server sends Content-Encoding (e.g. gzip), Content-Length is the
                # compressed size while iter_bytes() yields the larger decompressed
                # body, so comparing written bytes would falsely flag truncation.
                expected = _parse_content_length(r.headers.get("content-length"))
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1 << 16):
                        f.write(chunk)
                received = r.num_bytes_downloaded
                if expected is not None and received != expected:
                    raise RuntimeError(
                        f"size mismatch on download from {url}: "
                        f"Content-Length={expected} but received {received} bytes"
                    )
            tmp.replace(target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def is_downloaded(
        self,
        source: str,
        dest_root: Path,
        dataset_name: str,
    ) -> bool:
        url = self._strip_scheme(source)
        return (dest_root / dataset_name / _basename(url)).exists()

    @staticmethod
    def _strip_scheme(source: str) -> str:
        if not source.startswith("url:"):
            raise ValueError(f"Expected url: scheme, got {source!r}")
        return source[len("url:") :]


def _basename(url: str) -> str:
    """Filename from *url*; fallback ``"download"`` for trailing-slash paths.

    Intra-dataset basename collisions are rejected at ``@sieval_dataset``
    registration, so two URLs in one dataset never share a basename here.
    """
    return url_path_basename(url) or "download"


def _parse_content_length(raw: str | None) -> int | None:
    """Parse ``Content-Length``; ``None`` disables the truncation check.

    Chunked transfer encoding + some proxies omit the header entirely.
    """
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None
