"""Unit tests for the vendored SciTaRC prompt + grading assets.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest

from sieval.community.scitarc import (
    CORRECT_SCORE,
    EVAL_PROMPT,
    NO_REASONING,
    create_language_prompt,
    exact_match,
    extract_answer_language,
    get_table_text,
    normalize_text,
    parse_response,
)

# --- table flattening: inner lines carry their own newlines ---


def test_get_table_text_joins_lines_bare_and_tables_blank_line():
    tables = [["\\begin{table}\n", "row\n", "\\end{table}\n"], ["second\n"]]
    assert get_table_text(tables) == ("\\begin{table}\nrow\n\\end{table}\n\n\nsecond\n")


def test_get_table_text_empty():
    assert get_table_text([]) == ""


def test_create_language_prompt_carries_persona_tables_and_question():
    prompt = create_language_prompt("How many?", [["T\n"]])
    # The persona lives in the same string as the table block and the answer
    # instruction — the task sends all of it as ONE user turn, so a split here
    # would change the rendered text.
    assert prompt.startswith(
        "You are a helpful science assistant who answers questions about "
        "information in tables."
    )
    assert "Here is the relevant tabular data:\n\nT\n" in prompt
    assert 'Your final response should be "Answer:" followed by the answer.' in prompt
    assert prompt.endswith("Question: How many?\n")


# --- answer extraction ---


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Answer: 42", "42"),
        ("answer:42", "42"),
        ("Final Answer: NLLB-200-1.3B. 64.71", "NLLB-200-1.3B. 64.71"),
        ("  final answer :  x  ", "x"),
        ("\nAnswer: leading newline", "leading newline"),
        # No marker: the whole stripped reply IS the answer.
        ("no answer marker here", "no answer marker here"),
        ("", ""),
        ("   ", ""),
        # `Answer:` with nothing after it extracts to blank, which is what the
        # task turns into a `None` prediction and never sends to the grader.
        ("Answer:", ""),
    ],
)
def test_extract_answer_language(reply, expected):
    assert extract_answer_language(reply) == expected


def test_extract_answer_language_is_dotall_greedy_to_end():
    """Everything after the FIRST marker is the answer, second marker included.

    Upstream searches with ``DOTALL`` and captures ``(.*)$``, so trailing
    commentary lands inside the answer. That is what its published exact-match
    column was computed against, so it is preserved rather than tightened.
    """
    assert (
        extract_answer_language("Line1\nAnswer: a\nmore text\nAnswer: b")
        == "a\nmore text\nAnswer: b"
    )


# --- grader reply parsing: upstream's three-method ladder ---


def test_parse_response_json_path():
    score, reasoning, parsed = parse_response('{"reasoning": "ok", "score": 1.0}')
    assert (score, reasoning, parsed) == (1.0, "ok", True)


def test_parse_response_truncates_at_evaluation_end():
    reply = '{"reasoning": "ok", "score": 1.0}\n[Evaluation End]\n{"score": 0.0}'
    score, _, parsed = parse_response(reply)
    assert (score, parsed) == (1.0, True)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("score: 1.0", 1.0),
        ("SCORE: 0.5", 0.5),
        ("**Final Score**: 1.0", 1.0),
        ("the score = 1.0", 1.0),
        ("Scoring: 0.0", 0.0),
    ],
)
def test_parse_response_loose_score_patterns(reply, expected):
    score, _, parsed = parse_response(reply)
    assert (score, parsed) == (expected, True)


@pytest.mark.parametrize("reply", ["", "   \n  ", "I cannot evaluate this."])
def test_parse_response_unparsed_is_flagged_not_a_verdict(reply):
    """The flag is the whole point: upstream's 0.0 conflates two different facts.

    Without ``parsed`` a grader that returned nothing readable is indis-
    tinguishable from one that read the answer and scored it wrong, and
    ``n_grader_unparsed`` could not be counted at all.
    """
    score, reasoning, parsed = parse_response(reply)
    assert (score, parsed) == (0.0, False)
    assert reasoning == NO_REASONING


def test_parse_response_null_score_reads_as_unparsed():
    """The one deliberate divergence from upstream, which raises here.

    ``json.loads('{"score": null}')`` succeeds and ``float(None)`` then raises
    ``TypeError`` — not in the tuple upstream catches, so upstream kills the
    whole batch. Here the reply is what it is: unreadable.
    """
    assert parse_response('{"score": null}') == (0.0, NO_REASONING, False)


def test_parse_response_reads_reasoning_without_json():
    _, reasoning, _ = parse_response("reasoning: partly right\nscore: 0.5")
    assert reasoning == "partly right"


def test_partial_credit_is_not_correct():
    score, _, _ = parse_response('{"reasoning": "half", "score": 0.5}')
    assert score != CORRECT_SCORE


def test_rendered_grader_prompt_self_parses_as_a_perfect_score():
    """PINNED, not endorsed: the prompt's own JSON example reads as 1.0.

    ``{"reasoning": "Brief explanation", "score": 1.0}`` sits ahead of the
    ``[Evaluation End]`` truncation point and matches the very pattern
    :func:`parse_response` looks for. Upstream never trips on it — its
    completion endpoint returns only the continuation, so the example is never
    in the parsed text — but a chat grader that quotes the template back reads
    high, and this is the only misread that inflates rather than deflates.

    The assertion exists so nobody "repairs" the parser without deciding what
    to do about the prompt: tightening the regex here would silently change
    every published-column comparison. If this test starts failing, the parser
    diverged from upstream — check that first.
    """
    rendered = EVAL_PROMPT.format(question="q", ground_truth="g", prediction="p")
    assert parse_response(rendered) == (1.0, "Brief explanation", True)


def test_eval_prompt_renders_all_three_placeholders():
    rendered = EVAL_PROMPT.format(question="Q?", ground_truth="GOLD", prediction="PRED")
    assert rendered.endswith(
        "Question: Q?\nGround Truth: GOLD\nPrediction: PRED\n\n[Evaluation Start]"
    )
    # Upstream escapes the JSON example against its own `.format` call; if the
    # doubling were lost, `format` would raise on the `reasoning` key instead.
    assert '{"reasoning": "Brief explanation", "score": 1.0}' in rendered


# --- exact match: strict and case-sensitive ---


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        ("4", "4", True),
        ("  4  ", "4", True),
        (None, "4", False),
        ("", "4", False),
        ("4.0", "4", False),
        # Upstream leaves the `.lower()` commented out. This is most of why its
        # published EM column runs so far below the grader column.
        ("neural translation", "Neural Translation", False),
    ],
)
def test_exact_match(prediction, gold, expected):
    assert exact_match(prediction, gold) is expected


def test_normalize_text_stringifies_and_strips_only():
    assert normalize_text(None) == ""
    assert normalize_text("  x \n") == "x"
    assert normalize_text("MiXeD") == "MiXeD"
