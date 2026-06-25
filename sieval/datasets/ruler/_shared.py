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


def thinking_prefill(model_name: str, enable_thinking: bool) -> str:
    """Placeholder text a reasoning model prefills into the assistant turn when
    thinking is disabled; empty string for non-reasoning models.

    Qwen3 keeps the ``<think>...</think>`` framing even with thinking off — the
    empty block is injected so the model continues from the answer cue instead
    of reopening a reasoning span. Other models (and ``enable_thinking=True``)
    get nothing, so input-budget accounting and assistant prefill are no-ops in
    the general case. This is the single source of truth for the placeholder:
    both the dataset loaders (to reserve token budget) and the task base (to
    prefill the assistant turn) consume it, so the two can never disagree.
    """
    if not enable_thinking and "qwen3" in model_name.lower():
        return "<think>\n\n</think>\n\n"
    return ""


def _len_tag(length: int) -> str:
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
