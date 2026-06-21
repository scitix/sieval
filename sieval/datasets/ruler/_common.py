import gzip
import json
import os
import re

_NOISE_HAYSTACK = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)

_CORPUS_FILE = "PaulGrahamEssays.json.gz"

_NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."


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
    else:
        raise NotImplementedError(f"{type_haystack} is not implemented.")

def _ensure_punkt() -> None:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")
