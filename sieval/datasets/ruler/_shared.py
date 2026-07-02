"""Shared constants and helpers used across all RULER subtask modules."""

import gzip
import json
import os
import re
from typing import TypedDict, cast

import numpy as np

from sieval.community.ruler.datasets.constants import TASKS

_NOISE_HAYSTACK = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)

# Qwen3 thinking tag overhead: <think>\n\n</think>\n\n (4 tokens)
QWEN3_THINKING_TAG_OVERHEAD = 4
_CORPUS_FILE = "PaulGrahamEssays.json.gz"
_NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
_SQUAD_FILE = "dev-v2.0.json"
_DOCUMENT_PROMPT = "Document {i}:\n{document}"

# Pin the HotpotQA snapshot for reproducibility across downloads.
_HOTPOTQA_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"

# Pin english_words.json to the same RULER commit vendored into community/ruler/.
_RULER_DATA_SHA = "ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13"

# NIAH insertion depths (percentages).
_NIAH_DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))

# VT insertion depths (percentages).
_VT_DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))


class RulerTaskSpec(TypedDict):
    tokens_to_generate: int
    template: str
    answer_prefix: str


def ruler_task(name: str) -> RulerTaskSpec:
    """Return the RULER spec for *name* with precise field types."""
    return cast(RulerTaskSpec, TASKS[name])


def tokens_to_generate(
    task_name: str,
    *,
    enable_thinking: bool,
    think_budget: int,
    model_name: str = "",
) -> int:
    """Compute the total generation budget for a RULER task.

    Args:
        task_name: Name of the RULER task (e.g., "niah", "qa")
        enable_thinking: Whether thinking mode is enabled
        think_budget: Token budget for thinking content (used only when
                    enable_thinking=True)
        model_name: Model identifier (default "qwen3"). Only Qwen3-family models
                    have thinking tag overhead. Other models (e.g., "gpt-4", "llama")
                    should pass their own model_name for correct token calculation.

    Returns:
        Total tokens needed for generation, accounting for:
        - Thinking tag overhead (Qwen3 only): 4 tokens for <think>\n\n</think>\n\n
        - Thinking content (if enabled): think_budget tokens
        - Final answer: base tokens from task spec
    """
    base = ruler_task(task_name)["tokens_to_generate"]

    # Only Qwen3-family models have thinking tag overhead
    is_qwen3 = model_name.lower().startswith("qwen3")

    if not is_qwen3:
        # Other models: no thinking tag overhead
        return think_budget + base if enable_thinking else base

    # Qwen3: always includes thinking tag overhead
    if enable_thinking:
        return QWEN3_THINKING_TAG_OVERHEAD + think_budget + base
    else:
        return QWEN3_THINKING_TAG_OVERHEAD + 1 + base


def thinking_prefill(model_name: str, enable_thinking: bool) -> str:
    """Placeholder text a reasoning model prefills into the assistant turn.

    Compatibility layer supporting both assistant-message and user-message patterns.

    Qwen3 specifics:
    - When thinking is enabled: Returns empty string (model continues in the
      existing <think> block)
    - When thinking is disabled: Returns "<think>\n\n</think>\n\n" (empty block
      to skip reasoning)

    Other models: Always returns empty string (no special handling needed)

    This maintains backward compatibility with feat/ruler branch while supporting
    feat/ruler_exp's message pattern (appending answer_prefix to user message).
    """
    if "qwen3" in model_name.lower() and not enable_thinking:
        return "<think>\n\n</think>\n\n"  # Empty block; skip to answer
    return ""


def len_tag(length: int) -> str:
    """Convert a context length to a short tag: 4096 → '4k', 131072 → '128k'."""
    return f"{length // 1024}k" if length % 1024 == 0 else str(length)


def _build_haystack(name_or_path: str, type_haystack: str):
    if type_haystack == "essay":
        path = os.path.join(name_or_path, _CORPUS_FILE)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            text = json.load(f)["text"]
        return re.sub(r"\s+", " ", text).split(" ")
    if type_haystack == "noise":
        return _NOISE_HAYSTACK
    if type_haystack == "needle":
        return _NEEDLE
    raise NotImplementedError(f"{type_haystack} is not implemented.")


def _ensure_punkt() -> None:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")
