"""The SGLang legacy executor.

``SglangTransport`` is the native ``/generate`` executor behind
``SglangGenModel``'s explicit ``sglang_legacy`` bypass.  It is not a canonical
Dialect and must not be registered as ``sglang_native`` before PR 5.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from .sglang import SglangTransport

__all__ = ["SglangTransport"]
