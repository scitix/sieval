"""Tests for sieval/datasets/ruler/_shared.py — token calculation and templates.

Tests the unified token calculation system:
- get_template(): Message format template selection
- calculate_prompt_tokens(): Full token count including message format
- tokens_to_generate(): Base answer generation budget

calculate_prompt_tokens() consumes the RULER tokenizer wrappers
(HFTokenizer / OpenAITokenizer) via their ``text_to_tokens`` interface, so
tests build one through ``select_tokenizer`` — the same path the loaders use.
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
        calculate_prompt_tokens,
        get_template,
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
    """qwen3 thinking mode uses thinking template with /think marker."""
    template = get_template("qwen3", enable_thinking=True)
    assert "<|im_start|>user" in template
    assert "/think" in template
    assert "{task_template}" in template


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
def test_template_has_task_template_placeholder():
    """Every resolved template must carry the {task_template} placeholder."""
    for model in ["qwen3", "meta-llama3", "unknown-model"]:
        for thinking in [True, False]:
            template = get_template(model, thinking)
            assert "{task_template}" in template, (
                f"Template for {model} (thinking={thinking}) missing placeholder"
            )


# ---------------------------------------------------------------------------
# tokens_to_generate() — Base answer generation budget only
# ---------------------------------------------------------------------------


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


def test_tokens_to_generate_qwen3_thinking_includes_budget_and_tags():
    """qwen3 thinking budget = tag overhead + think_budget + base (the max_tokens)."""
    budget = tokens_to_generate(
        "niah", enable_thinking=True, think_budget=8192, model_name="qwen3"
    )
    assert budget == 4 + 8192 + 128


def test_tokens_to_generate_non_qwen3_thinking_no_tag_overhead():
    """Non-qwen3 thinking = think_budget + base (no <think> tag generation)."""
    budget = tokens_to_generate(
        "niah", enable_thinking=True, think_budget=8192, model_name="gpt-4"
    )
    assert budget == 8192 + 128


# ---------------------------------------------------------------------------
# calculate_prompt_tokens() — Full token count with message template
# ---------------------------------------------------------------------------


@_needs_ruler_deps
def test_calculate_prompt_tokens_nonthinking_is_template_plus_content(tokenizer):
    """Non-thinking count == empty-template overhead + raw content tokens."""
    prompt = "What is 2+2?"

    total = calculate_prompt_tokens(
        tokenizer, prompt, model_name="qwen3", enable_thinking=False
    )
    template_overhead = calculate_prompt_tokens(
        tokenizer, "", model_name="qwen3", enable_thinking=False
    )
    content = len(tokenizer.text_to_tokens(prompt))

    # RULER prompts tokenize additively across the template boundary.
    assert total == template_overhead + content


@_needs_ruler_deps
def test_calculate_prompt_tokens_excludes_generation_budget(tokenizer):
    """calculate_prompt_tokens is prompt-only: it must NOT add any think budget.

    The thinking budget lives in tokens_to_generate (max_tokens), so switching
    thinking on/off changes only the template overhead, never by a think budget.
    """
    prompt = "What is 2+2?"
    thinking = calculate_prompt_tokens(
        tokenizer, prompt, model_name="qwen3", enable_thinking=True
    )
    nonthinking = calculate_prompt_tokens(
        tokenizer, prompt, model_name="qwen3", enable_thinking=False
    )
    # Both are prompt-side only; the difference is the small template delta
    # (/think marker vs prefilled <think></think> block), never thousands.
    assert abs(thinking - nonthinking) < 20


@_needs_ruler_deps
def test_calculate_prompt_tokens_monotonic_in_content(tokenizer):
    """More content yields more tokens."""
    short = calculate_prompt_tokens(
        tokenizer, "a", model_name="qwen3", enable_thinking=False
    )
    long = calculate_prompt_tokens(
        tokenizer, "a " * 100, model_name="qwen3", enable_thinking=False
    )
    assert long > short


@_needs_ruler_deps
def test_calculate_prompt_tokens_unknown_model_base_template(tokenizer):
    """Unknown model uses base template — count equals raw content tokens."""
    prompt = "hello world"
    total = calculate_prompt_tokens(
        tokenizer, prompt, model_name="unknown", enable_thinking=False
    )
    assert total == len(tokenizer.text_to_tokens(prompt))


# ---------------------------------------------------------------------------
# Integration: prompt tokens + generation budget fits within max_seq_length
# ---------------------------------------------------------------------------


@_needs_ruler_deps
def test_full_budget_nonthinking_fits(tokenizer):
    """prompt tokens + answer budget stays within a modest max_seq_length."""
    max_seq_length = 4096
    prompt = "Context sentence. " * 100

    total = calculate_prompt_tokens(
        tokenizer, prompt, model_name="qwen3", enable_thinking=False
    ) + tokens_to_generate(
        "niah", enable_thinking=False, think_budget=0, model_name="qwen3"
    )
    assert total <= max_seq_length


@_needs_ruler_deps
def test_full_budget_thinking_includes_think_budget_via_gen(tokenizer):
    """The think_budget enters the fitting total through tokens_to_generate."""
    think_budget = 2048
    prompt = "Context sentence. " * 100

    prompt_tokens = calculate_prompt_tokens(
        tokenizer, prompt, model_name="qwen3", enable_thinking=True
    )
    gen_thinking = tokens_to_generate(
        "niah", enable_thinking=True, think_budget=think_budget, model_name="qwen3"
    )
    gen_nonthinking = tokens_to_generate(
        "niah", enable_thinking=False, think_budget=0, model_name="qwen3"
    )
    # The thinking fitting total reserves think_budget more room, and it comes
    # from the generation budget — not from calculate_prompt_tokens.
    assert (prompt_tokens + gen_thinking) - (prompt_tokens + gen_nonthinking) == (
        think_budget + 4  # + qwen3 <think></think> tag overhead
    )
