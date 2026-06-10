"""SourceHandler Protocol + scheme registry. v0.2 ships ``hf:``, ``url:``, ``local:``.

AI-Generated Code - Claude Haiku 4.5 (Anthropic)
"""

from pathlib import Path
from typing import Protocol


class SourceHandler(Protocol):
    """Downloader for a single source scheme (e.g. hf:, url:)."""

    scheme: str

    def download(
        self,
        source: str,
        dest_root: Path,
        dataset_name: str,
        force: bool,
    ) -> None:
        """Fetch *source* into *dest_root*; raise on failure. Layout is
        scheme-specific — locate files via ``is_downloaded`` if needed."""

    def is_downloaded(
        self,
        source: str,
        dest_root: Path,
        dataset_name: str,
    ) -> bool:
        """Probe whether *source* is already resident under *dest_root*."""


_HANDLERS: dict[str, SourceHandler] = {}
_builtin_registered = False


def _ensure_builtin_handlers() -> None:
    global _builtin_registered
    if _builtin_registered:
        return
    from sieval.datasets.downloaders.hf import HFHandler
    from sieval.datasets.downloaders.local import LocalHandler
    from sieval.datasets.downloaders.url import URLHandler

    for handler in (HFHandler(), URLHandler(), LocalHandler()):
        if handler.scheme not in _HANDLERS:
            register_handler(handler)
    _builtin_registered = True


def register_handler(h: SourceHandler) -> None:
    if h.scheme in _HANDLERS:
        raise ValueError(f"Handler for scheme {h.scheme!r} is already registered")
    _HANDLERS[h.scheme] = h


def resolve(source: str) -> SourceHandler:
    _ensure_builtin_handlers()
    scheme = source.split(":", 1)[0]
    if scheme not in _HANDLERS:
        raise NotImplementedError(
            f"Unknown source scheme: {scheme!r}. Registered: {sorted(_HANDLERS)}."
        )
    return _HANDLERS[scheme]
