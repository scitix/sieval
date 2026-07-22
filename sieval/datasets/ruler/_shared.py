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
    context_length: int | None = None,
    for_dataset: bool = False,
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
    not to generation, and is reserved via :func:`model_template_token`.

    Qwen3 Think Adaptation:
        This implementation diverges from upstream RULER (@NVIDIA/RULER ab17b78)
        to support Qwen3's extended thinking mode (per Qwen3 technical report).

        Upstream RULER: Computes a single ``tokens_to_generate`` value that accounts
        for the model's generation capacity without distinguishing between dataset
        fitting (context window calculation) and inference (max_tokens constraint).

        Qwen3 Adaptation: Splits the budget calculation based on context_length:
        - Small contexts (!128k): During dataset generation, assume thinking
          content uses the native context window space (don't reserve think_budget
          for fitting). During inference, max_tokens = gen_budget + think_budget.
        - Large contexts (128k): During dataset generation, reserve think_budget
          space (model can't reuse thinking tokens). During inference, max_tokens
          already includes think_budget.

        This ensures generated samples fit within their target context window
        while allocating sufficient max_tokens for thinking during inference.

    Args:
        task_name: Name of the RULER task (e.g., "niah", "qa")
        enable_thinking: Whether thinking mode is enabled
        think_budget: Token budget for thinking (Qwen3: typically 8192)
        model_name: Model identifier. Only Qwen3-family models generate
            ``<think>`` tags that add the tag overhead.
        context_length: Context length in tokens (4096, 8192, 32768, 131072, etc.).
            Determines think_budget allocation strategy during dataset generation.
        for_dataset: Whether this is called during dataset generation (True) or
            inference (False). Affects think_budget inclusion based on context_length.

    Returns:
        Total tokens the model may generate (thinking + tags + answer).
    """
    base = ruler_task(task_name)["tokens_to_generate"]
    is_qwen3 = model_name.lower().startswith("qwen3")

    if not enable_thinking:
        # Non-thinking: the empty <think></think> block (Qwen3) is prefilled in
        # the prompt template, so only the answer is generated.
        return base

    # Thinking mode: Qwen3-adapted budget allocation based on context_length
    # (Diverges from upstream RULER which always includes think_budget)
    should_skip_think_budget = (
        for_dataset
        and context_length is not None
        and context_length != 131072  # 128k
    )

    if should_skip_think_budget:
        # SMALL CONTEXT (4k, 8k, etc.) DATASET GENERATION:
        # Assume thinking content reuses the native context window space.
        # Don't reserve think_budget in gen_budget — this makes the generated
        # samples shorter, fitting within the small context window.
        # Inference will later add think_budget via:
        # max_tokens = gen_budget + think_budget.
        #
        # Rationale: In Qwen3's thinking mode, models can perform inference over
        # the thinking tokens within the same context pass for small windows.
        if is_qwen3:
            return QWEN3_THINKING_TAG_OVERHEAD + base
        return base

    # LARGE CONTEXT (128k) OR INFERENCE:
    # For large contexts during dataset generation: reserve think_budget upfront
    # (model cannot reuse thinking tokens efficiently across context boundaries).
    # For inference regardless of context_length: always include think_budget
    # (inference max_tokens must cover all generated output including thinking).
    #
    # Thinking budget = tags (Qwen3 only) + think_budget + answer base
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


# Qwen3 message templates. These are a sieval addition — upstream RULER's
# vendored template.py (community/ruler/datasets, @ab17b78) has no Qwen3 entry —
# so they live here to keep the vendored file byte-faithful to upstream.
# Non-thinking prefills an empty <think></think> block to suppress reasoning;
# thinking opens the assistant turn and lets the model generate the block.
_QWEN3_TEMPLATES = {
    False: (
        "<|im_start|>user\n{task_template} <|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
    True: "<|im_start|>user\n{task_template} <|im_end|>\n<|im_start|>assistant\n",
}


def get_template(model_name: str, enable_thinking: bool) -> str:
    """Get the message template for a model and thinking mode.

    Qwen3 templates are sieval-owned (see ``_QWEN3_TEMPLATES``); other models use
    the vendored upstream ``Templates`` (case-insensitive), falling back to the
    passthrough ``base`` template for unknown models.

    Args:
        model_name: Model identifier (e.g., "qwen3", "gpt-4", "llama")
        enable_thinking: Whether thinking mode is enabled

    Returns:
        Template string with a ``{task_template}`` placeholder.
    """
    if model_name.lower().startswith("qwen3"):
        return _QWEN3_TEMPLATES[enable_thinking]

    # Other models: vendored template by name (case-insensitive), else base.
    by_lower = {k.lower(): k for k in Templates}
    key = by_lower.get(model_name.lower())
    return Templates[key] if key is not None else "{task_template}"


def model_template_token(
    tokenizer,
    model_name: str = "qwen3",
    enable_thinking: bool = False,
) -> int:
    """Per-config template-token reserve, as NVIDIA/RULER computes it (@ab17b78).

    Mirrors upstream ``prepare.py``::

        model_template_token = len(text_to_tokens(model_template))

    where ``model_template`` is the RAW template string — the literal
    ``{task_template}`` placeholder is NOT replaced. Upstream then does
    ``max_seq_length -= model_template_token`` before fitting; callers here add
    this reserve to the content token count, which is algebraically identical.

    Counting the unformatted template reserves the placeholder's own tokens,
    which are replaced by real content at inference and never emitted — that is
    the headroom that keeps prompts off the exact ``max_seq_length`` boundary.

    This value is invariant per ``(model_name, enable_thinking)``, so compute it
    once per dataset build and reuse it across the fitting loop and per-sample
    checks rather than recomputing it for every candidate.
    """
    return len(tokenizer.text_to_tokens(get_template(model_name, enable_thinking)))
