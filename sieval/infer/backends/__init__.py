"""
Backend registry: lookup infer backends and translators by name.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from sieval.infer.backends.sglang_translator import SglangTranslator
from sieval.infer.backends.translator import BackendTranslator
from sieval.infer.backends.vllm_translator import VllmTranslator

__all__ = [
    "BackendTranslator",
    "SglangTranslator",
    "VllmTranslator",
    "get_translator",
]

_TRANSLATORS: dict[str, type[BackendTranslator]] = {
    "vllm": VllmTranslator,
    "sglang": SglangTranslator,
}


def get_translator(name: str) -> BackendTranslator:
    """Get a translator instance by backend name."""
    cls = _TRANSLATORS.get(name)
    if cls is None:
        available = ", ".join(sorted(_TRANSLATORS)) or "(none)"
        raise KeyError(f"Unknown backend {name!r}. Available: {available}")
    return cls()
