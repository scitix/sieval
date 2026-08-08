"""Unit tests for UGMathBench prompting, extraction, and per-type grading.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest

from sieval.community.ugmathbench import (
    SUBJECTS,
    VERSIONS,
    build_prompt,
    describe_answer_type,
    extract_answer,
    extract_predictions,
    judge_answer,
    judge_answers,
    math_equal,
    split_answers,
)

# --- prompt ----------------------------------------------------------------


def test_single_answer_prompt_uses_singular_wording():
    prompt = build_prompt("Algebra", "Solve $x+1=2$. [ANS]", 1, ["NV"])
    assert "This problem involves only one placeholders [ANS]" in prompt
    assert "The answer type is a numerical value without units." in prompt
    assert 'end your response with: "The final answer is \\boxed{ANSWER}"' in prompt
    assert "Problem:\nSolve $x+1=2$. [ANS]" in prompt


def test_multi_answer_prompt_lists_every_type_in_order():
    prompt = build_prompt(
        "Trigonometry", "a) [ANS] b) [ANS]", 2, ["NV", "TF"], [[], []]
    )
    assert "This problem involves 2 placeholders [ANS]" in prompt
    assert (
        "Their answer types are, in order, a numerical value without units, "
        "either True or False." in prompt
    )
    assert 'end your response with: "The final answers are \\boxed{ANSWER}"' in prompt


def test_multiple_choice_description_embeds_the_option_list():
    # Upstream interpolates the Python list repr; the model sees the brackets.
    assert describe_answer_type("MCS", ["A", "B"]) == (
        "one option of a multiple choice question with options ['A', 'B']"
    )


def test_unknown_answer_type_is_rejected():
    # LookupError, not KeyError: the latter's `__str__` is `repr(args[0])`,
    # so the sentence would reach the CLI wrapped in a second set of quotes.
    with pytest.raises(LookupError) as exc_info:
        describe_answer_type("XX")

    assert not isinstance(exc_info.value, KeyError)
    assert str(exc_info.value).startswith("unknown UGMathBench answer type 'XX'")


def test_answer_count_drives_the_wording_not_the_declared_type_count():
    # A handful of pinned rows declare fewer types than answers; upstream takes
    # the count from the answers and describes only the declared types.
    prompt = build_prompt("Linear_algebra", "p", 5, ["NV", "NV"])
    assert "This problem involves 5 placeholders [ANS]" in prompt
    assert prompt.count("a numerical value without units") == 2


def test_missing_option_entries_are_padded_not_fatal():
    prompt = build_prompt("Algebra", "p", 2, ["MCS", "MCS"], [["A", "B"]])
    assert "options ['A', 'B']" in prompt
    assert "options []" in prompt


def test_benchmark_shape_constants():
    assert len(SUBJECTS) == 16
    assert VERSIONS == 3


# --- extraction ------------------------------------------------------------


def test_extracts_the_last_box():
    response = "First \\boxed{1}. On reflection, \\boxed{2}."
    assert extract_answer(response) == "2"


def test_boxed_extraction_is_brace_balanced():
    assert extract_answer("So \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_falls_back_to_an_explicit_answer_handoff():
    assert extract_answer("blah blah. The final answer is 42.") == "42"


def test_no_answer_at_all_is_none_not_a_guess():
    # Strict extraction: a response full of numbers but no box and no hand-off
    # must not be scored on the last number it happened to mention.
    assert extract_answer("We compute 17, then 3, and stop.") is None
    assert extract_predictions("We compute 17, then 3, and stop.") is None


def test_normalization_strips_decoration_but_not_content():
    assert extract_answer("\\boxed{\\left(\\frac{1}{2}\\right)}") == "(\\frac{1}{2})"
    assert extract_answer("\\boxed{\\text{none}}") == "none"


def test_split_keeps_bracketed_commas_together():
    assert split_answers("1, (2, 3), [4, 5]") == ["1", "(2, 3)", "[4, 5]"]


def test_split_keeps_latex_group_commas_together():
    assert split_answers("\\frac{a,b}{c}, 2") == ["\\frac{a,b}{c}", "2"]


def test_split_folds_latex_set_delimiters():
    assert split_answers("\\{1, 2\\}, 3") == ["(1, 2)", "3"]


def test_split_treats_angle_brackets_as_operators_not_grouping():
    # Upstream counts `<` and `>` as brackets. Here they are the relational
    # operators the dataset actually uses -- a slot whose whole answer is `<` --
    # so counting them swallows the following comma and the row comes out short
    # by a slot, which grades every slot in it wrong however good the answer.
    assert split_answers("<, 55000") == ["<", "55000"]
    assert split_answers("10/[5^{2*n+1}], <, monotone decreasing") == [
        "10/[5^{2*n+1}]",
        "<",
        "monotone decreasing",
    ]


def test_split_does_not_let_an_unmatched_closer_swallow_the_rest():
    # Upstream's depth goes negative here, and below zero no later comma splits
    # -- so one stray `>` costs every remaining slot in the row, not just its
    # own. `Arithmetic_0071` is exactly this shape.
    assert split_answers("12+20/4, >, (12+20)/4") == ["12+20/4", ">", "(12+20)/4"]
    assert split_answers(") , a, b") == [")", "a", "b"]


def test_split_still_groups_latex_angle_delimiters():
    # Dropping `<`/`>` costs no real grouping: the inner-product form is
    # `\langle ... \rangle`, which is folded to parentheses before the scan.
    # The fold is textual, so `\langle `'s trailing space rides along -- what
    # matters is that the comma inside stays inside.
    assert split_answers("\\langle 1, 2\\rangle, 3") == ["( 1, 2)", "3"]
    assert split_answers("\\langle1,2\\rangle, 3") == ["(1,2)", "3"]


# --- per-type grading ------------------------------------------------------


def test_numerical_value_accepts_latex_equivalents():
    assert judge_answer("\\frac{\\sqrt{3}}{3}", "1/sqrt(3)", "NV")
    assert not judge_answer("2", "1/sqrt(3)", "NV")


def test_numerical_value_uses_relative_tolerance():
    assert judge_answer("1000.5", "1000.0", "NV", precision=1e-2)
    assert not judge_answer("1000.5", "1000.0", "NV", precision=1e-6)


def test_numerical_value_normalizes_a_percent_marked_reference():
    # The dataset stores a few "numerical value without units" answers with a
    # percent sign; a plain-number prediction still matches.
    assert judge_answer("0.66", "0.66%", "NV")


def test_expression_equivalence_is_symbolic():
    assert judge_answer("x^3+2x^2+6", "x^3+2*x^2+6", "EX")
    assert not judge_answer("x^3+2x^2+7", "x^3+2*x^2+6", "EX")


def test_equation_equivalence():
    assert judge_answer("T = 1.02x + 10", "T = 1.02*x+10", "EQ")


def test_interval_equivalence():
    assert judge_answer("(-\\infty, 5)", "(-infinity,5)", "INT")
    assert not judge_answer("(-\\infty, 5]", "(-infinity,5)", "INT")


def test_true_false_accepts_the_datasets_yn_spelling():
    assert judge_answer("True", "Y", "TF")
    assert judge_answer("yes", "Y", "TF")
    assert judge_answer("False", "N", "TF")
    assert not judge_answer("True", "N", "TF")
    # Not a boolean at all -> wrong, never silently true.
    assert not judge_answer("maybe", "Y", "TF")


def test_single_choice_letter_forms():
    options = ["A", "B", "C", "D"]
    assert judge_answer("C", "C", "MCS", options)
    assert judge_answer("c", "C", "MCS", options)
    assert judge_answer("C. the third one", "C", "MCS", options)
    assert not judge_answer("D", "C", "MCS", options)


def test_single_choice_supports_non_letter_option_labels():
    assert judge_answer("Q(X)", "Q(X)", "MCS", ["P(X)", "Q(X)"])


def test_multiple_choice_ignores_order_but_not_membership():
    options = ["A", "B", "C", "D", "E"]
    assert judge_answer("DCA", "ACD", "MCM", options)
    assert not judge_answer("ACDE", "ACD", "MCM", options)
    assert not judge_answer("AC", "ACD", "MCM", options)


def test_multiple_choice_supports_non_letter_option_labels():
    # Option letters are matched a character at a time, so word-labelled choices
    # yield no letters at all and the slot would be unwinnable without the
    # whole-answer comparison the single-choice rule already does.
    options = ["even", "odd", "neither"]
    assert judge_answer("even, odd", "even, odd", "MCM", options)
    assert not judge_answer("even", "even, odd", "MCM", options)


def test_open_ended_is_text_not_mathematics():
    assert judge_answer("None", "none", "OE")
    assert not judge_answer("0", "none", "OE")


def test_ordered_list_respects_order():
    assert judge_answer("(1, 2, 3, 6)", "(1, 2, 3, 6)", "OL")
    assert not judge_answer("(2, 1, 3, 6)", "(1, 2, 3, 6)", "OL")


def test_unordered_list_ignores_order():
    assert judge_answer("(x3, e)", "(e, x3)", "UOL")
    assert not judge_answer("(x3, x4)", "(e, x3)", "UOL")


def test_unordered_list_matches_multiset_not_set():
    assert not judge_answer("(1, 1)", "(1, 2)", "UOL")


def test_list_elements_are_not_read_as_booleans():
    # Upstream converts booleans only for a slot the dataset typed TF —
    # `norm_ans_str` gates `norm_str2bool` on `ans_type == "TF"` — and leaves
    # list elements alone under a standing `TODO: deal with OL with boolean`.
    # No OL/UOL reference on the pinned revision holds a boolean: all 1,244 are
    # values or parameter names, so there is nothing here to win by coercing.
    assert not judge_answer("(True, 1)", "(Y, 1)", "OL")


@pytest.mark.parametrize(
    ("pred", "gold", "kind"),
    [
        ("(x, t)", "(x, y)", "OL"),
        ("(1, f)", "(1, n)", "OL"),
        ("(y, 2)", "(t, 2)", "UOL"),
        ("(s, t)", "(s, y)", "UOL"),
    ],
)
def test_parameter_names_are_not_truth_values(pred, gold, kind):
    # `t`, `y`, `f` and `n` are the parameter names this dataset actually uses —
    # 13 OL/UOL references on the pinned revision carry a bare `t` or `y`. Read
    # as booleans they collapse into each other and a wrong answer scores right,
    # which is the one direction a grader must never fail in.
    assert not judge_answer(pred, gold, kind)


def test_math_equal_survives_unparseable_input():
    # Malformed LaTeX must grade wrong, not raise into the runner.
    assert not math_equal("\\frac{{{", "1")


# --- sample-level grading --------------------------------------------------


def test_every_slot_must_be_right():
    golds = ["1", "2"]
    types = ["NV", "NV"]
    assert judge_answers(["1", "2"], golds, types) == [True, True]
    assert judge_answers(["1", "3"], golds, types) == [True, False]


def test_slot_count_mismatch_grades_everything_wrong():
    assert judge_answers(["1"], ["1", "2"], ["NV", "NV"]) == [False, False]
    assert judge_answers(["1", "2", "3"], ["1", "2"], ["NV", "NV"]) == [False, False]


def test_missing_extraction_grades_everything_wrong():
    assert judge_answers(None, ["1", "2"], ["NV", "NV"]) == [False, False]


def test_undeclared_trailing_types_are_still_graded():
    # 3 answers, 2 declared types: the trailing slot falls through to the
    # mathematical rule instead of being dropped.
    assert judge_answers(["1", "2", "3"], ["1", "2", "3"], ["NV", "NV"]) == [
        True,
        True,
        True,
    ]
    assert judge_answers(["1", "2", "9"], ["1", "2", "3"], ["NV", "NV"]) == [
        True,
        True,
        False,
    ]


def test_end_to_end_boxed_response():
    response = (
        "Working through it... The final answers are "
        "\\boxed{-\\sqrt{3}, -1, \\sqrt{3}, \\frac{2\\sqrt{3}}{3}, \\sqrt{2}}"
    )
    golds = ["-sqrt(3)", "-1", "sqrt(3)", "2/sqrt(3)", "sqrt(2)"]
    predictions = extract_predictions(response)
    assert predictions is not None and len(predictions) == 5
    assert all(judge_answers(predictions, golds, ["NV"] * 5))


def test_a_reference_with_no_answers_is_never_correct():
    # Two pinned rows are upstream-corrupt: the problem text is an error message
    # and the answer sequence is empty. `all([])` is True, so an empty verdict
    # list must not read as a correct sample.
    assert judge_answers(["1"], [], []) == []
    assert judge_answers(None, [], []) == []


# --- gold parsed as sympy source, not as LaTeX -----------------------------
# UGMathBench stores gold answers in a plain-sympy dialect while models answer
# in LaTeX. `math_verify.parse` reads everything as LaTeX, where an unescaped
# `sin` is the product s*i*n and `pi` is p*i, so `7*sin(pi*x/5)+1` came back as
# `7*s*i*n*(i*p*x)/5 + 1` and could only ever match by exact string equality.
# The first live run measured the cost at 34.9% of wrong slots in the
# slot-aligned bucket, entirely in the free-form answer types.


@pytest.mark.parametrize(
    ("pred", "gold"),
    [
        # the canonical shape: LaTeX function names against sympy ones
        ("7 \\sin(\\frac{\\pi}{5}x) + 1", "7*sin(pi*x/5)+1"),
        ("a^x \\ln a", "ln(a)*a^x"),
        ("7z - \\cos z + 11", "7*z-cos(z)+11"),
        # equal only after simplification, which no normalizer reaches
        ("3\\cos(2\\sqrt{35}t)", "3*cos(sqrt(980/7)*t)"),
        ("\\dfrac{8}{5 - 4\\cos(q^2)}", "2*8/(2 + 1*8 - 1*8*cos(q^2))"),
        ("-162\\pi", "-pi*2*9^2"),
        # the dataset's square brackets are grouping, and x^0 is a literal 1
        ("e^{-8x} \\cos(9x)", "x^0*e^(-8*x)*cos(9*x)"),
        ("\\frac{18.02}{0.013} ( e^{0.013t} - 1 )", "1386.15*[e^{0.013*t}-1]"),
        # implicit multiplication on the prediction side
        ("5 - 5c", "-5*c+5*1"),
        # same line, different parameterization
        ("\\frac{23 - x}{54}", "0.333333+(-0.0185185)*(x-5)"),
        # overflows to infinity at a large probe -- the ladder has to stay
        # small enough to leave usable points, or a correct exponential answer
        # is marked wrong for want of evidence
        ("4 e^{\\cosh(4x)} \\sinh(4x)", "2.71828182845905^{cosh(4*x)} * sinh(4*x) * 4"),
    ],
)
def test_latex_prediction_matches_plain_sympy_gold(pred, gold):
    assert math_equal(pred, gold)


@pytest.mark.parametrize(
    ("pred", "gold"),
    [
        # rounded more coarsely than the benchmark's own relative tolerance:
        # still wrong, and substitution must not rescue it
        ("2.3", "2.2892"),
        ("1.97", "1.97423402868049"),
        ("0.16", "0.160257708151404"),
        ("25/37", "0.657894736842105"),
        # plainly different
        ("x^2", "x^3"),
        ("\\sin(x)", "cos(x)"),
        ("10\\sqrt{2}", "5*[1+sqrt(2)]"),
        ("162", "54"),
        # same shape, different variable: not the same answer
        ("x+1", "t+1"),
        # exponentials that differ only in rate: the small probe ladder must
        # still separate them
        ("e^{x}", "e^{2x}"),
        ("\\frac{1}{x}", "\\frac{1}{x^2}"),
    ],
)
def test_substitution_does_not_credit_wrong_answers(pred, gold):
    assert not math_equal(pred, gold)


def test_substitution_reaches_list_elements_too():
    # OL/UOL compare element-wise through `math_equal`, so the fix has to show
    # up there as well -- that is where the multi-answer rows live.
    assert judge_answer("(0, 2\\cos x \\sin x)", "(0,2*cos(x)*sin(x))", "OL")
    assert judge_answer(
        "(e^{-8x} \\cos(9x), x e^{-8x})",
        "(x^1*e^(-8*x), x^0*e^(-8*x)*cos(9*x))",
        "UOL",
    )


def test_structured_types_are_untouched_by_the_substitution_pass():
    # TF / MCS / MCM / OE never route through `math_equal`, and the live-run
    # audit found 0 false negatives in all of them. Pin that they stay strict.
    assert not judge_answer("True", "False", "TF")
    assert not judge_answer("B", "A", "MCS", ["A", "B", "C"])
    assert not judge_answer("AB", "AC", "MCM", ["A", "B", "C"])
    assert not judge_answer("prime", "composite", "OE")


def test_substitution_verdicts_are_deterministic():
    # The probe ladder is fixed rather than a seeded RNG on purpose: a grader
    # has to return the same verdict for the same pair regardless of how many
    # comparisons ran before it, and in every process.
    pairs = [("7 \\sin(\\frac{\\pi}{5}x) + 1", "7*sin(pi*x/5)+1"), ("x^2", "x^3")]
    first = [math_equal(p, g) for p, g in pairs]
    for _ in range(3):
        assert [math_equal(p, g) for p, g in pairs] == first
    assert first == [True, False]


# --- untrusted-input guards ------------------------------------------------


def test_parsing_a_prediction_cannot_reach_the_interpreter(tmp_path):
    # `parse_expr` evaluates what it parses, and the string reaching it is model
    # output -- on the ordinary path, since every wrong free-form answer falls
    # through to the substitution pass. Its default global namespace is built by
    # `exec("from sympy import *", ...)`, which also injects `__builtins__`.
    marker = tmp_path / "executed"
    payload = f"__import__('os').system('touch {marker}')"
    assert judge_answers([payload], ["42"], ["EX"]) == [False]
    assert not marker.exists()

    payload = f"open('{marker}', 'w')"
    assert judge_answers([payload], ["42"], ["EX"]) == [False]
    assert not marker.exists()


#: Every way a call can hand a *string* back to sympy, which re-sympifies it
#: with sympy's own default namespace -- builtins included. Clearing
#: `__builtins__` for the top-level parse does not reach into that nested one,
#: so each of these executes despite the sanitized namespace. `eval` is the
#: shape that survives the sanitizing most surprisingly: it is not a sympy name,
#: so `auto_symbol` rewrites it to `Function('eval')`, and calling a sympy
#: Function on a `str` sympifies the argument.
_NESTED_SYMPIFY_CARRIERS = ["eval", "sympify", "S", "N", "Function('f')"]


@pytest.mark.parametrize("carrier", _NESTED_SYMPIFY_CARRIERS)
def test_no_call_shape_can_smuggle_a_string_back_into_the_interpreter(
    tmp_path, carrier
):
    # The namespace restriction covers one parse. These carriers start another
    # one, and refusing the callee by name cannot work -- `auto_symbol` turns
    # every unknown name into a callable, so the allowlist would have to be of
    # *all* names. What is refused instead is the quote.
    marker = tmp_path / f"executed-{carrier[:4]}"
    payload = f"{carrier}(\"__import__('os').system('touch {marker}')\")"
    assert judge_answers([payload], ["42"], ["EX"]) == [False]
    assert not marker.exists(), f"{carrier}(...) reached the interpreter"


def test_a_quote_free_call_is_still_parsed_normally():
    # The guard refuses quotes, not calls -- a nested sympify with no string
    # literal to read gets a sympy object and does nothing (`chr(112)` stays
    # symbolic). Refusing calls outright would drop every `sin(pi*x/5)` gold.
    assert judge_answer("sin(pi/6)", "1/2", "EX") is True
    assert judge_answer("eval(chr(112))", "42", "EX") is False


def test_refusing_a_quoted_prediction_costs_only_the_sympy_reading():
    # A refused prediction is not a refused sample: `_parse_sympy_source` is one
    # of three readings, and the LaTeX and literal-equality paths still run. A
    # correct answer that happens to carry a quote must still grade correct.
    assert judge_answer("42'", "42'", "EX") is True


def test_a_power_tower_grades_wrong_instead_of_hanging():
    # `^` is rewritten to `**` before parsing and sympy exponentiates eagerly,
    # so `9^9^9^9` asks for a 370-million-digit integer. Grading has to reject
    # it, not compute it: `feedback()` offloads to a worker process, so one
    # such sample would hold a worker for the rest of the run.
    assert judge_answer("9^9^9^9", "x+1", "EX") is False
    assert judge_answer("2^{2^{100}}", "x+1", "EX") is False


def test_the_exponent_cap_leaves_real_answers_alone():
    # Only the tower shape and absurd exponents are refused. A large-but-sane
    # power, and a left-nested one, still compare normally.
    assert math_equal("2^100", "2**100")
    assert math_equal("(x^2)^3", "x**6")
    assert math_equal("4^20", "1.09951162778E+12")


def test_a_mangled_numeric_gold_still_reaches_the_substitution_pass():
    # The LaTeX reader turns `2**100` into `2`. Returning that comparison's
    # verdict made the substitution pass unreachable for every pair the mangling
    # happens to turn into a *number*, which is the exact defect the pass exists
    # to repair -- so a correct `2^100` was graded against 2 and lost.
    assert math_equal("2^100", "2**100")


def test_single_answer_branch_describes_only_the_first_declared_type():
    # Upstream's single-answer branch reads `answer_type[0]` alone, while its
    # multi-answer branch joins every declared type. Every pinned row agrees
    # (one answer means one type), so this pins the shape rather than a
    # coincidence of the current data cut.
    prompt = build_prompt("Algebra", "p", 1, ["NV", "EX"])
    assert "The answer type is a numerical value without units." in prompt
    assert "an expression" not in prompt
