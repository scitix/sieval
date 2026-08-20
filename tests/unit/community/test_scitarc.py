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


def test_echoed_grader_prompt_is_not_read_as_a_verdict():
    """The template's own JSON example must not become the score.

    ``{"reasoning": "Brief explanation", "score": 1.0}`` sits ahead of the
    ``[Evaluation End]`` truncation point and matches the very pattern
    :func:`parse_response` looks for, so a chat grader that quotes the template
    back would otherwise read as a perfect score — the one misread that
    inflates rather than deflates. Upstream cannot reach it (its completion
    endpoint returns only the continuation); the parser reconstructs that
    boundary by cutting to the last ``[Evaluation Start]``.

    Nothing follows the marker in a bare echo, so there is no verdict to read
    and the reply is a read failure — counted by ``n_grader_unparsed`` instead
    of scoring a silent 1.0.
    """
    rendered = EVAL_PROMPT.format(question="q", ground_truth="g", prediction="p")
    assert parse_response(rendered) == (0.0, NO_REASONING, False)


def test_echoed_grader_prompt_does_not_outrank_the_real_verdict():
    """An echo followed by a real grade must yield the grade, not the example."""
    rendered = EVAL_PROMPT.format(question="q", ground_truth="g", prediction="p")
    verdict = '{"reasoning": "missing a part", "score": 0.5}'
    assert parse_response(f"{rendered}\n{verdict}\n[Evaluation End]") == (
        0.5,
        "missing a part",
        True,
    )


def test_first_scoring_json_wins():
    """Upstream's ``re.search`` takes the FIRST brace-run; a later one must not win.

    Pinned because switching to the last match is the natural-looking "repair"
    for the two echo cases above, and it would silently change every
    published-column comparison. The echo is handled at the prompt/reply
    boundary; the score patterns stay upstream's.
    """
    assert parse_response('{"score": 1.0} then {"score": 0.0}')[0] == 1.0


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
