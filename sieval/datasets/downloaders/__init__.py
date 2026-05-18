"""Dataset source-scheme handlers. Builtins register lazily on first
``resolve()`` so package import stays dep-light.

AI-Generated Code - Claude Opus 4.7 (Anthropic)
"""

from sieval.datasets.downloaders.base import (
    SourceHandler,
    register_handler,
    resolve,
)

__all__ = ["SourceHandler", "register_handler", "resolve"]
