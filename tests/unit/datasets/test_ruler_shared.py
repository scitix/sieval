"""Tests for sieval/datasets/ruler/_shared.py — token calculation and templates.

Tests the unified token calculation system:
- get_template(): Message format template selection
- model_template_token(): upstream per-config template reserve
- tokens_to_generate(): generation budget (answer + any think_budget)

model_template_token() consumes the RULER tokenizer wrappers
(HFTokenizer / OpenAITokenizer) via their ``text_to_tokens`` interface, so
tests build one through ``select_tokenizer`` — the same path the loaders use.
Loaders size prompts as ``len(text_to_tokens(content)) + model_template_token``.
"""

import pytest

try:
    import tiktoken as _tiktoken  # noqa: F401

    _ruler_deps = True
except ImportError:
    _ruler_deps = False

_needs_ruler_deps = pytest.mark.skipif(
    not _ruler_deps, reason="ruler deps group not installed"
)

if _ruler_deps:
    from sieval.community.ruler.scripts.tokenizer import select_tokenizer
    from sieval.datasets.ruler._shared import (
        _is_qwen3,
        get_template,
        model_template_token,
        thinking_prefill,
        tokens_to_generate,
    )


@pytest.fixture
def tokenizer():
    """RULER OpenAI tokenizer wrapper (cl100k_base) — no model download needed."""
    return select_tokenizer("openai", "cl100k_base")


# ---------------------------------------------------------------------------
# get_template() — Template selection by model and thinking mode
# ---------------------------------------------------------------------------


@_needs_ruler_deps
def test_get_template_qwen3_nonthinking():
    """qwen3 non-thinking mode uses nonthinking template with empty think block."""
    template = get_template("qwen3", enable_thinking=False)
    assert "<|im_start|>user" in template
    assert "<think>" in template  # Empty think block to skip reasoning
    assert "<|im_end|>" in template
    assert "{task_template}" in template


@_needs_ruler_deps
def test_get_template_qwen3_thinking():
    """qwen3 thinking mode opens the assistant turn WITHOUT a prefilled block.

    In thinking mode the model generates its own <think>...</think>, so the
    template must NOT prefill an empty block (that is the non-thinking behavior).
    """
    template = get_template("qwen3", enable_thinking=True)
    assert "<|im_start|>user" in template
    assert "<|im_start|>assistant" in template
    assert "{task_template}" in template
    # Distinguishing feature vs non-thinking: no prefilled closed think block.
    assert "</think>" not in template
    # Non-thinking, by contrast, DOES prefill the empty block.
    assert "</think>" in get_template("qwen3", enable_thinking=False)


@_needs_ruler_deps
def test_get_template_other_models():
    """Other models use their named template from Templates if available."""
    template = get_template("meta-llama3", enable_thinking=False)
    assert "<|begin_of_text|>" in template
    assert "{task_template}" in template


@_needs_ruler_deps
def test_get_template_unknown_model_uses_base():
    """Unknown models default to the passthrough base template."""
    template = get_template("completely-unknown-model", enable_thinking=False)
    assert template == "{task_template}"


@_needs_ruler_deps
def test_get_template_qwen3_case_insensitive_and_not_base():
    """Regression: qwen3 must resolve to the real template regardless of the

    capitalization used for the key in template.py ("Qwen3-nonthinking"). A
    case mismatch previously fell back to base, counting ZERO template overhead
    and overflowing max_seq_length. Resolution is case-insensitive and never base.
    """
    for name in ("qwen3", "Qwen3-8b", "QWEN3-8B"):
        for thinking in (False, True):
            tmpl = get_template(name, thinking)
            assert tmpl != "{task_template}"
            assert "<|im_start|>" in tmpl


@_needs_ruler_deps
def test_get_template_qwen3_is_sieval_owned_not_vendored(monkeypatch):
    """Qwen3 templates are sieval-owned, independent of the vendored Templates.

    Regression for the community/ vendor-fidelity fix: even if the vendored
    upstream Templates dict has no Qwen3 entry (as upstream indeed doesn't),
    get_template still resolves Qwen3 from the local _QWEN3_TEMPLATES.
    """
    import sieval.datasets.ruler._shared as shared_mod

    monkeypatch.setattr(shared_mod, "Templates", {"base": "{task_template}"})
    for thinking in (False, True):
        tmpl = get_template("qwen3", enable_thinking=thinking)
        assert "<|im_start|>" in tmpl and "{task_template}" in tmpl


@_needs_ruler_deps
def test_template_has_task_template_placeholder():
    """Every resolved template must carry the {task_template} placeholder."""
    for model in ["qwen3", "meta-llama3", "unknown-model"]:
        for thinking in [True, False]:
            template = get_template(model, thinking)
            assert "{task_template}" in template, (
                f"Template for {model} (thinking={thinking}) missing placeholder"
            )


# ---------------------------------------------------------------------------
# _is_qwen3() — basename-normalized qwen3 detection (case table)
# ---------------------------------------------------------------------------


@_needs_ruler_deps
@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("qwen3", True),
        ("Qwen3-8b", True),
        ("QWEN3-8B", True),
        ("Qwen/Qwen3-8B", True),  # HF repo id (served-model-name shape)
        ("/models/Qwen3-8B", True),  # local checkpoint path (served-model-name shape)
        ("gpt-4", False),
        ("meta-llama3", False),
        ("", False),
    ],
)
def test_is_qwen3_case_table(model_name, expected):
    assert _is_qwen3(model_name) is expected


@_needs_ruler_deps
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-8B", "/models/Qwen3-8B"])
def test_thinking_prefill_recognizes_served_model_ids(model_name):
    """Served ids (not just the YAML's bare 'qwen3') must still get the empty

    <think></think> prefill in non-thinking mode — preprocess feeds the served
    id here, not model_name: qwen3 from the dataset config.
    """
    prefill = thinking_prefill(model_name, enable_thinking=False)
    assert prefill == "<think>\n\n</think>\n\n"
    assert thinking_prefill(model_name, enable_thinking=True) == ""


@_needs_ruler_deps
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-8B", "/models/Qwen3-8B"])
def test_get_template_recognizes_served_model_ids(model_name):
    for thinking in (False, True):
        assert get_template(model_name, thinking) == get_template("qwen3", thinking)


# ---------------------------------------------------------------------------
# tokens_to_generate() — Base answer generation budget only
# ---------------------------------------------------------------------------


@_needs_ruler_deps
@pytest.mark.parametrize(
    ("task_name", "expected_base"),
    [
        ("niah", 128),
        ("qa", 32),
        ("variable_tracking", 30),
        ("common_words_extraction", 120),
        ("freq_words_extraction", 50),
    ],
)
def test_tokens_to_generate_nonthinking_is_base(task_name, expected_base):
    """Non-thinking generation budget is just the answer base."""
    assert (
        tokens_to_generate(
            task_name, enable_thinking=False, think_budget=0, model_name="qwen3"
        )
        == expected_base
    )


@_needs_ruler_deps
def test_tokens_to_generate_qwen3_thinking_includes_budget_and_tags():
    """qwen3 thinking budget = tag overhead + think_budget + base (the max_tokens)."""
    budget = tokens_to_generate(
        "niah", enable_thinking=True, think_budget=8192, model_name="qwen3"
    )
    assert budget == 4 + 8192 + 128


@_needs_ruler_deps
def test_tokens_to_generate_non_qwen3_thinking_no_tag_overhead():
    """Non-qwen3 thinking = think_budget + base (no <think> tag generation)."""
    budget = tokens_to_generate(
        "niah", enable_thinking=True, think_budget=8192, model_name="gpt-4"
    )
    assert budget == 8192 + 128


# ---------------------------------------------------------------------------
# tokens_to_generate() — Qwen3 extended thinking: dataset vs inference split
# ---------------------------------------------------------------------------


@_needs_ruler_deps
def test_tokens_to_generate_qwen3_small_context_dataset_no_think_budget():
    """Small context (4k) dataset generation: skip think_budget in gen_budget."""
    budget = tokens_to_generate(
        "niah",
        enable_thinking=True,
        think_budget=8192,
        model_name="qwen3",
        context_length=4096,
        for_dataset=True,
    )
    # Only tag overhead + base, not think_budget
    assert budget == 4 + 128


@_needs_ruler_deps
def test_tokens_to_generate_qwen3_32k_context_dataset_no_think_budget():
    """32k dataset generation: under the 128k-only rule 32k is treated as small,
    so think_budget is NOT reserved in gen_budget (added later at inference)."""
    budget = tokens_to_generate(
        "niah",
        enable_thinking=True,
        think_budget=8192,
        model_name="qwen3",
        context_length=32768,
        for_dataset=True,
    )
    # Only tag overhead + base, not think_budget
    assert budget == 4 + 128


@_needs_ruler_deps
def test_tokens_to_generate_qwen3_128k_context_dataset_with_think_budget():
    """Large context (128k) dataset generation: reserve think_budget upfront."""
    budget = tokens_to_generate(
        "niah",
        enable_thinking=True,
        think_budget=8192,
        model_name="qwen3",
        context_length=131072,
        for_dataset=True,
    )
    assert budget == 4 + 8192 + 128


@_needs_ruler_deps
def test_tokens_to_generate_qwen3_small_context_inference_with_think_budget():
    """Small context (4k) inference: add think_budget (omitted in dataset gen)."""
    budget = tokens_to_generate(
        "niah",
        enable_thinking=True,
        think_budget=8192,
        model_name="qwen3",
        context_length=4096,
        for_dataset=False,
    )
    # Tag overhead + think_budget + base (same as 32k case, think_budget is included)
    assert budget == 4 + 8192 + 128


@_needs_ruler_deps
def test_tokens_to_generate_qwen3_32k_context_inference_with_think_budget():
    """32k inference: think_budget is always included at inference — 32k gen
    omitted it under the 128k-only rule, so it is added on top here."""
    budget = tokens_to_generate(
        "niah",
        enable_thinking=True,
        think_budget=8192,
        model_name="qwen3",
        context_length=32768,
        for_dataset=False,
    )
    # Tag overhead + think_budget + base
    assert budget == 4 + 8192 + 128


@_needs_ruler_deps
def test_tokens_to_generate_gpt_small_context_dataset():
    """Non-Qwen3 small context dataset: no tag overhead, no think_budget."""
    budget = tokens_to_generate(
        "niah",
        enable_thinking=True,
        think_budget=8192,
        model_name="gpt-4",
        context_length=4096,
        for_dataset=True,
    )
    # Only base, no tags, no think_budget
    assert budget == 128


@_needs_ruler_deps
def test_tokens_to_generate_reserve_think_budget_true_overrides_legacy_rule():
    """Explicit reserve_think_budget=True reserves think_budget even at a length
    the legacy (context_length == 131072) heuristic would treat as having
    headroom — the native-serving-without-headroom case from the review."""
    budget = tokens_to_generate(
        "niah",
        enable_thinking=True,
        think_budget=8192,
        model_name="qwen3",
        context_length=32768,
        for_dataset=True,
        reserve_think_budget=True,
    )
    assert budget == 4 + 8192 + 128


@_needs_ruler_deps
def test_tokens_to_generate_reserve_think_budget_false_overrides_legacy_rule():
    """Explicit reserve_think_budget=False skips the reserve even at 128k."""
    budget = tokens_to_generate(
        "niah",
        enable_thinking=True,
        think_budget=8192,
        model_name="qwen3",
        context_length=131072,
        for_dataset=True,
        reserve_think_budget=False,
    )
    assert budget == 4 + 128


@_needs_ruler_deps
def test_tokens_to_generate_gpt_32k_context_dataset():
    """Non-Qwen3 32k dataset generation: under the 128k-only rule 32k is treated
    as small — no tag overhead, no think_budget (added later at inference)."""
    budget = tokens_to_generate(
        "niah",
        enable_thinking=True,
        think_budget=8192,
        model_name="gpt-4",
        context_length=32768,
        for_dataset=True,
    )
    # Only base, no tags, no think_budget
    assert budget == 128


# ---------------------------------------------------------------------------
# model_template_token() — upstream per-config template reserve
# ---------------------------------------------------------------------------


@_needs_ruler_deps
def test_model_template_token_qwen3_nonzero(tokenizer):
    """Regression: the qwen3 template reserve must be non-zero.

    The overflow bug was a silent base fallback (empty template) that made this
    reserve 0, so prompts were sized ~13 tokens too large and overran context.
    """
    reserve = model_template_token(tokenizer, model_name="qwen3", enable_thinking=False)
    # role markers + prefilled <think></think> block
    assert reserve >= 10


@_needs_ruler_deps
def test_model_template_token_equals_raw_template_count(tokenizer):
    """Matches upstream: len(text_to_tokens(RAW template incl {task_template}))."""
    for thinking in (False, True):
        reserve = model_template_token(
            tokenizer, model_name="qwen3", enable_thinking=thinking
        )
        raw = get_template("qwen3", thinking)  # placeholder NOT replaced
        assert reserve == len(tokenizer.text_to_tokens(raw))


@_needs_ruler_deps
def test_model_template_token_unknown_model_is_base(tokenizer):
    """Unknown model → base template reserve = len(tokenize("{task_template}"))."""
    reserve = model_template_token(
        tokenizer, model_name="unknown", enable_thinking=False
    )
    assert reserve == len(tokenizer.text_to_tokens("{task_template}"))


# ---------------------------------------------------------------------------
# Integration: content + template reserve + generation budget fits max_seq_length
# ---------------------------------------------------------------------------


def _sizing(tokenizer, prompt, *, model_name, enable_thinking, think_budget):
    """Loader sizing: content + template reserve + generation budget."""
    reserve = model_template_token(
        tokenizer, model_name=model_name, enable_thinking=enable_thinking
    )
    gen = tokens_to_generate(
        "niah",
        enable_thinking=enable_thinking,
        think_budget=think_budget,
        model_name=model_name,
    )
    return len(tokenizer.text_to_tokens(prompt)) + reserve + gen


@_needs_ruler_deps
def test_full_budget_nonthinking_fits(tokenizer):
    """content + reserve + answer budget stays within a modest max_seq_length."""
    max_seq_length = 4096
    prompt = "Context sentence. " * 100
    total = _sizing(
        tokenizer, prompt, model_name="qwen3", enable_thinking=False, think_budget=0
    )
    assert total <= max_seq_length


@_needs_ruler_deps
def test_full_budget_thinking_reserves_think_budget(tokenizer):
    """The think_budget enters the sizing total via tokens_to_generate, not the

    template reserve — enabling thinking must add think_budget (+ qwen3 tag
    overhead) of room beyond the non-thinking sizing.
    """
    think_budget = 2048
    prompt = "Context sentence. " * 100
    thinking = _sizing(
        tokenizer,
        prompt,
        model_name="qwen3",
        enable_thinking=True,
        think_budget=think_budget,
    )
    nonthinking = _sizing(
        tokenizer, prompt, model_name="qwen3", enable_thinking=False, think_budget=0
    )
    # Difference is think_budget + the 4-token generated <think></think> tags,
    # adjusted by the small template delta between the two qwen3 templates.
    template_delta = model_template_token(
        tokenizer, model_name="qwen3", enable_thinking=True
    ) - model_template_token(tokenizer, model_name="qwen3", enable_thinking=False)
    assert thinking - nonthinking == think_budget + 4 + template_delta
