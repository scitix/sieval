"""Shared constants and helpers used across all RULER subtask modules."""

import gzip
import json
import os
import re
from typing import TypedDict, cast

import numpy as np

from sieval.community.ruler.datasets.constants import TASKS
from sieval.community.ruler.datasets.template import Templates

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

# HotpotQA HuggingFace repo id. `sieval dataset download` mirrors it to
# `<data_dir>/hotpotqa/hotpot_qa` (downloaders/hf.py: `dest_root / repo_id`), so
# the loader resolves the staged copy by joining this onto the base data dir.
_HOTPOTQA_REPO_ID = "hotpotqa/hotpot_qa"

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
    """Compute the generation budget (``max_tokens``) for a RULER task.

    This is what ``infer()`` passes as ``max_tokens``, so it must cover
    everything the model *generates*:

    - Thinking content (``think_budget``) when thinking is enabled — the model
      generates the reasoning, so it needs room for it.
    - The generated ``<think>...</think>`` tags (Qwen3, thinking mode only).
    - The final answer (``base`` from the task spec).

    The message-template overhead (role markers, and the *prefilled* empty
    ``<think></think>`` block in Qwen3 non-thinking mode) belongs to the prompt,
    not to generation, and is counted by :func:`calculate_prompt_tokens`.

    Args:
        task_name: Name of the RULER task (e.g., "niah", "qa")
        enable_thinking: Whether thinking mode is enabled
        think_budget: Token budget for generated thinking content
        model_name: Model identifier. Only Qwen3-family models generate the
            ``<think>`` tags that add the tag overhead.

    Returns:
        Total tokens the model may generate (thinking + tags + answer).
    """
    base = ruler_task(task_name)["tokens_to_generate"]
    is_qwen3 = model_name.lower().startswith("qwen3")

    if not enable_thinking:
        # Non-thinking: the empty <think></think> block (Qwen3) is prefilled in
        # the prompt template, so only the answer is generated.
        return base

    # Thinking: the model generates think_budget tokens of reasoning + answer.
    # Qwen3 also generates the <think>...</think> tags.
    if is_qwen3:
        return QWEN3_THINKING_TAG_OVERHEAD + think_budget + base
    return think_budget + base


def thinking_prefill(model_name: str, enable_thinking: bool) -> str:
    """Placeholder text a reasoning model prefills into the assistant turn.

    Compatibility layer supporting both assistant-message and user-message patterns.

    Qwen3 specifics:
    - When thinking is enabled: Returns empty string (model continues in the
      existing <think> block)
    - When thinking is disabled: Returns "<think>\n\n</think>\n\n" (empty block
      to skip reasoning)

    Other models: Always returns empty string (no special handling needed)
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


def get_template(model_name: str, enable_thinking: bool) -> str:
    """Get the message template for a model and thinking mode.

    Returns the formatting template that will be used during inference.
    This ensures data generation uses the same format as actual inference.

    Args:
        model_name: Model identifier (e.g., "qwen3", "gpt-4", "llama")
        enable_thinking: Whether thinking mode is enabled

    Returns:
        Template string with {task_template} placeholder
    """
    # Case-insensitive lookup: template.py capitalizes model names inconsistently
    # (e.g. "Qwen3-nonthinking", "Phi3", "meta-llama3"). Matching on lowercase
    # avoids the silent base-fallback bug where a case mismatch made the template
    # overhead count as zero and prompts overflowed max_seq_length.
    by_lower = {k.lower(): k for k in Templates}

    if model_name.lower().startswith("qwen3"):
        want = "qwen3-thinking" if enable_thinking else "qwen3-nonthinking"
        if want not in by_lower:
            # Fail loud: a missing Qwen3 template must not silently degrade to
            # base (zero overhead) — that under-sizes prompts.
            raise KeyError(
                f"RULER template {want!r} not found in Templates "
                f"(available: {sorted(Templates)}). Check "
                f"sieval/community/ruler/datasets/template.py."
            )
        return Templates[by_lower[want]]

    # Other models: use the model's named template if present, else base.
    key = by_lower.get(model_name.lower())
    return Templates[key] if key is not None else "{task_template}"


def calculate_prompt_tokens(
    tokenizer,
    prompt: str,
    *,
    model_name: str = "qwen3",
    enable_thinking: bool = False,
) -> int:
    """Count prompt tokens the way NVIDIA/RULER's prepare.py + niah.py do.

    Mirrors upstream exactly (commit ab17b78): content tokens plus the
    ``model_template_token`` reserve, where::

        model_template_token = len(text_to_tokens(model_template))

    and ``model_template`` is the RAW template string *including* the literal
    ``{task_template}`` placeholder (it is NOT formatted with the content first).
    Upstream then does ``max_seq_length -= model_template_token`` and fits with
    ``content + tokens_to_generate <= max_seq_length``; folding the reserve into
    the returned count here is algebraically identical.

    Counting the unformatted template means the placeholder's own tokens are
    reserved but never emitted at inference (they are replaced by real content),
    leaving a few tokens of headroom — this is why upstream never fills the
    context to exactly ``max_seq_length``, and why sizing on the *formatted*
    template (exact overhead, zero headroom) let requests hit the limit exactly
    and get rejected by the serving engine.

    The generation budget — including ``think_budget`` when thinking is enabled —
    is NOT counted here; it is returned separately by :func:`tokens_to_generate`.

    Args:
        tokenizer: RULER tokenizer wrapper exposing ``text_to_tokens``
        prompt: The prompt/task text (bare content, unwrapped)
        model_name: Model identifier for template selection
        enable_thinking: Whether thinking mode is enabled (selects the template)

    Returns:
        content tokens + model_template_token (the upstream reserve).
    """
    template = get_template(model_name, enable_thinking)
    # Upstream reserve: tokenize the RAW template (placeholder unreplaced).
    model_template_token = len(tokenizer.text_to_tokens(template))
    # RULER tokenizers (HFTokenizer / OpenAITokenizer) expose ``text_to_tokens``.
    return len(tokenizer.text_to_tokens(prompt)) + model_template_token
