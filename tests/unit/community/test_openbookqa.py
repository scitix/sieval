"""Tests for the OpenCompass-vendored OBQA prompt template and extractor.

AI-Generated Code - Opus 4.8 (Anthropic)
"""

from sieval.community.openbookqa import (
    OBQA_OPTIONS,
    OBQA_PROMPT_TEMPLATE,
    first_option_postprocess,
)


def test_prompt_template_matches_opencompass_main_variant():
    # Pinned byte-for-byte against obqa_gen_9069e4.py _template[0]. Any drift
    # here silently de-aligns us from the reference, so assert the exact string.
    assert OBQA_PROMPT_TEMPLATE == (
        "Question: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"
    )
    assert OBQA_OPTIONS == "ABCD"


def test_extracts_english_answer_phrasings():
    assert first_option_postprocess("The answer is B.", OBQA_OPTIONS) == "B"
    assert first_option_postprocess("ANSWER: C", OBQA_OPTIONS) == "C"
    assert (
        first_option_postprocess("The correct answer is option (D).", OBQA_OPTIONS)
        == "D"
    )


def test_cushion_fallback_returns_first_bare_option():
    # No answer phrasing — cushion patterns pick the first option-letter present.
    assert first_option_postprocess("A", OBQA_OPTIONS) == "A"


def test_returns_empty_when_no_option_present():
    assert first_option_postprocess("none of these apply", OBQA_OPTIONS) == ""


def test_options_arg_restricts_alphabet():
    # The letter must be drawn from `options`; "E" is not a valid OBQA option.
    assert first_option_postprocess("The answer is E.", OBQA_OPTIONS) == ""
