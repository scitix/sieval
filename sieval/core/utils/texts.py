"""Text normalization and choice extraction for answer matching."""

import re
import string

_ARTICLES = {"a", "an", "the"}
_PUNCT = set(string.punctuation)


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy answer matching.

    Applies: lowercase → strip punctuation → remove articles (a, an, the) →
    collapse whitespace.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    tokens = [tok for tok in text.split() if tok and tok not in _ARTICLES]
    return " ".join(tokens).strip()


def general_postprocess(text: str) -> str:
    """Normalize short-form answers for exact/substring matching."""
    return normalize_text(text.strip().lower())


def extract_choice(text: str) -> str:
    """Extract the first A-E option label from a model response."""
    match = re.search(r"([A-E])", text.strip().upper())
    return match.group(1) if match else ""
