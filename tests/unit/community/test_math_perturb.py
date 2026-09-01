"""Unit tests for the vendored MATH-Perturb grader.

The point of most of these is that a specific *upstream* behaviour survived the
port — MATH-Perturb's evaluation code is a fork of DeepSeek-Math's, and every
case below is one of the differences the fork introduced. A test that would also
pass against `sieval.community.deepseek_math` proves nothing about this file, so
several assert against both.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest
from lark.exceptions import LarkError
from sympy.parsing.latex import parse_latex as parse_latex_core

from sieval.community import deepseek_math
from sieval.community.math_perturb import (
    MAX_ABS_TOL,
    answer_check,
    eval_math,
    extract_boxed_answers,
    extract_ground_truth_answer,
    extract_math_answer,
    extract_math_perturb_ground_truth_answer,
    extract_predicted_answer,
    is_correct,
    math_equal,
    parse_latex,
    strip_string,
    symbolic_equal,
)

# --- upstream's own self-check, which is why the lark backend is a dependency ---


def test_upstream_test_parse_latex_passes():
    """`evaluate.py::test_parse_latex`, kept as a test rather than module code.

    Upstream asserts this at import of its `__main__` with the comment "there may
    be error in sympy package!", so it is the one behaviour it self-checks.
    """
    assert answer_check(
        "A fake problem statement.",
        "\nThe answer is \n\\[ \\boxed{1 - \\sqrt[4]{x + 1}} \\]\n",
        "1 - (x + 1)^{\\frac14}",
        "perturb",
    )


def test_parse_latex_uses_the_lark_backend():
    """Upstream passes ``backend="lark"``, which is NOT sympy's default.

    Pinned by its consequence rather than by reading the argument: on sympy 1.14
    the lark grammar cannot parse ``\\pi`` at all, while the default ANTLR
    backend can. The exception TYPE is the second half of the pin — an ANTLR
    failure would not be a ``LarkError``. If this ever starts parsing, or starts
    raising something else, the backend changed underneath and the scores moved.
    """
    with pytest.raises(LarkError):
        parse_latex("\\pi")
    assert str(parse_latex_core("\\pi")) == "pi"


# --- fork difference: tolerance tightened 1e-3 -> 1e-7 ---


def test_max_abs_tol_is_upstreams_tightened_tolerance():
    assert MAX_ABS_TOL == 1e-7


def test_tightened_tolerance_separates_two_close_fractions():
    """Upstream's stated reason: 1e-3 called 1/5120 equal to 1/28800."""
    assert not math_equal("1/5120", "1/28800")
    # ... and DeepSeek-Math, at abs_tol=1e-3, does not.
    assert deepseek_math.math_equal("1/5120", "1/28800")


# --- fork difference: `{,}` thousands separators ---


def test_braced_thousands_separator_is_a_number():
    assert math_equal("18{,}234", "18234")
    assert is_correct({"prediction": "18{,}234", "answer": "18234"}, prec=MAX_ABS_TOL)


# --- fork difference: strip_string keeps spaces and \cdot, drops spacing macros ---


def test_strip_string_keeps_spaces():
    """Upstream disabled DeepSeek-Math's ``string.replace(" ", "")``."""
    assert strip_string("1 + 2") == "1 + 2"
    assert deepseek_math.strip_string("1 + 2") == "1+2"


def test_strip_string_keeps_cdot():
    assert "\\cdot" in strip_string("2 \\cdot 3")
    assert "\\cdot" not in deepseek_math.strip_string("2 \\cdot 3")


@pytest.mark.parametrize("macro", ["\\,", "\\:", "\\;", "\\quad"])
def test_strip_string_drops_spacing_macros(macro):
    assert strip_string(f"12{macro}34") == "1234"


def test_strip_string_drops_single_backslash_space_but_not_double():
    assert strip_string("12\\ 34") == "1234"
    # `\\\\ ` (a line break followed by a space) is deliberately not touched.
    assert "\\\\" in strip_string("12\\\\ 34")


# --- fork difference: unicode normalization, added for o1 / o3-mini ---


@pytest.mark.parametrize(
    ("raw", "expected_fragment"),
    [
        ("√2", "\\sqrt{2}"),
        ("∛8", "\\sqrt[3]{8}"),
        ("π", "\\pi"),
        ("∞", "\\infty"),
        ("x²", "^{2}"),
        ("𝟯", "3"),
        ("x₂", "_{2}"),
        ("½", "\\frac{1}{2}"),
        ("2×3", "\\times"),
    ],
)
def test_unicode_normalization(raw, expected_fragment):
    assert expected_fragment in strip_string(raw)


def test_unicode_normalization_maps_both_dashes_to_hyphen():
    assert strip_string("5–3") == "5-3"
    assert strip_string("5−3") == "5-3"


#: Upstream's `_fix_unicode` replacement table at the pinned commit, keyed by
#: CODEPOINT rather than by character, and with every value ASCII.
#:
#: Written this way on purpose. The characters this table is keyed on are exactly
#: the ones an editor, a clipboard or a transcription step silently folds to an
#: ASCII lookalike — and a folded key is not a syntax error, it is a live entry
#: that maps a character to itself. That is how U+2003 (EM SPACE) once became
#: U+0020 here: `' ': ' '`, an identity no-op, visually identical to the original
#: in every diff, invisible to `ruff` (which skips `sieval/community`) and to
#: every behavioural test that did not happen to name that character.
#:
#: So this file must contain no non-ASCII literal of its own: an integer cannot
#: be folded, and `chr()` reconstructs the character at runtime.
UPSTREAM_UNICODE_REPLACEMENTS = {
    0x00B2: "^{2}",
    0x00B3: "^{3}",
    0x207F: "^{n}",
    0x03C0: "\\pi ",
    0x221E: "\\infty ",
    0x23A3: "\\lfloor ",
    0x23A6: "\\rfloor ",
    0x2013: "-",
    0x2212: "-",
    0x222A: "\\cup ",
    0x2229: "\\cap ",
    0x00B7: "\\cdot ",
    0x00D7: "\\times ",
    0x2003: " ",
    0x2044: "/",
    0x00A0: " ",
    0x00BD: "\\frac{1}{2}",
    0x220F: "\\prod ",
    0x2211: "\\sum ",
}


def _replacement_table():
    """`_fix_unicode`'s literal ``replacements`` dict, read out of the source.

    By AST rather than by calling the function: the point is to compare the
    table itself against upstream's, including keys whose effect is a no-op and
    which therefore cannot be detected from behaviour alone.
    """
    import ast
    import inspect

    from sieval.community import math_perturb

    tree = ast.parse(inspect.getsource(math_perturb))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == "replacements"
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("`replacements` table not found in _fix_unicode")


def test_unicode_replacement_table_matches_upstream_codepoint_for_codepoint():
    """Every key is the character upstream keyed on, not an ASCII lookalike."""
    assert {ord(k): v for k, v in _replacement_table().items()} == (
        UPSTREAM_UNICODE_REPLACEMENTS
    )


def test_no_replacement_key_is_an_identity_no_op():
    """The shape the EM SPACE regression took: a key that maps to itself.

    Discriminating on its own — it fails for a folded key even if the pinned
    table above were updated to match the damage.
    """
    assert [k for k, v in _replacement_table().items() if k == v] == []


@pytest.mark.parametrize(
    ("codepoint", "expected_fragment"),
    [
        (0x2003, " "),  # EM SPACE -> a plain space, so LaTeX still parses
        (0x00A0, " "),  # NBSP
        # FRACTION SLASH. The `/` it becomes is then folded by `_fix_a_slash_b`,
        # so the fraction IS the evidence it was normalized at all.
        (0x2044, "\\frac{1}{2}"),
        (0x222A, "\\cup"),
        (0x2229, "\\cap"),
        (0x00B7, "\\cdot"),
        (0x207F, "^{n}"),
        (0x00B3, "^{3}"),
        (0x220F, "\\prod"),
        (0x2211, "\\sum"),
        (0x23A3, "\\lfloor"),
        (0x23A6, "\\rfloor"),
    ],
)
def test_unicode_normalization_covers_every_replacement(codepoint, expected_fragment):
    """Behavioural cover for the table entries the cases above do not name.

    `chr()`, not a literal, for the reason given on
    :data:`UPSTREAM_UNICODE_REPLACEMENTS`.
    """
    assert expected_fragment in strip_string("1" + chr(codepoint) + "2")


def test_em_space_normalization_reaches_a_verdict():
    """The regression's real cost: a correct answer typeset with U+2003.

    Graded wrong while the entry was an identity no-op — 31 of the 54 pinned
    rows whose gold contains a space flipped. Asserted end-to-end through
    `answer_check`, so it fails if the table is right but nothing consumes it.
    """
    gold = "\\sqrt{2} - 1"
    response = "\\boxed{" + gold.replace(" ", chr(0x2003)) + "}"
    assert answer_check("A problem.", response, gold, "perturb")


def test_unicode_normalization_is_silent():
    """A library must not write to stdout; upstream prints on every conversion."""
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        strip_string("π + √2")
    assert buffer.getvalue() == ""


# --- fork difference: boxed( ), boxed[ ], and `boxed {` spacing ---


def test_extract_boxed_answers_tolerates_space_before_brace():
    assert extract_boxed_answers("so \\boxed {42}.") == ["42"]


def test_extract_boxed_answers_accepts_paren_and_bracket_forms():
    assert extract_boxed_answers("so boxed(42).") == ["42"]
    assert extract_boxed_answers("so boxed[42].") == ["42"]


def test_paren_form_is_a_fallback_only():
    """Upstream tries ``boxed(``/``boxed[`` only when ``boxed{`` found nothing."""
    assert extract_boxed_answers("\\boxed{7} and boxed(9)") == ["7"]


# --- fork difference: the two extractors split multi-valued answers differently ---


def test_prediction_extractor_splits_on_text_or_and_bare_and():
    assert extract_math_answer("Q?", "\\boxed{2 \\text{ or } 3}", "") == ["2", "3"]
    assert extract_math_answer("Q?", "\\boxed{2 and 3}", "") == ["2", "3"]


def test_gold_extractor_splits_on_a_BARE_or_only():
    """The gold-side hack: ``" or "`` splits, but a bare ``and`` does not."""
    assert extract_math_perturb_ground_truth_answer("Q?", "\\boxed{2 or 3}", "") == [
        "2",
        "3",
    ]
    assert extract_math_perturb_ground_truth_answer("Q?", "\\boxed{2 and 3}", "") == [
        "2 and 3"
    ]


def test_commas_split_when_the_question_says_separated_by_commas():
    question = "List them, separated by commas."
    assert extract_math_answer(question, "\\boxed{1,2,3}", "") == ["1", "2", "3"]
    # ... but not when the answer is a tuple or interval.
    assert extract_math_answer(question, "\\boxed{(1,2)}", "") == ["(1,2)"]


# --- the entry point: gold wrapping, prediction dedup, dataset_type ---


def test_gold_is_wrapped_in_boxed_and_re_extracted():
    assert extract_ground_truth_answer("Q?", "42", "perturb") == ["42"]


def test_gold_accepts_int_and_float_labels():
    """Upstream str()s a non-string label; the loader casts for the same reason."""
    assert extract_ground_truth_answer("Q?", 42, "perturb") == ["42"]
    assert extract_ground_truth_answer("Q?", 0.5, "perturb") == ["0.5"]


def test_gold_original_arm_reads_a_solution_not_a_bare_label():
    """``dataset_type="original"`` is carried for the unported third column."""
    assert extract_ground_truth_answer("Q?", "Work.\nSo \\boxed{7}.", "original") == [
        "7"
    ]


def test_dataset_type_is_checked():
    with pytest.raises(AssertionError):
        extract_ground_truth_answer("Q?", "1", "simple")


def test_prediction_extraction_dedups_preserving_order():
    assert extract_predicted_answer("Q?", "\\boxed{5} then \\boxed{5}") == ["5"]
    assert extract_predicted_answer("Q?", "\\boxed{5} then \\boxed{6}") == ["5", "6"]


def test_prediction_extraction_is_silent():
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        extract_predicted_answer("Q?", "\\boxed{5} then \\boxed{6}")
        extract_ground_truth_answer("Q?", "2 or 3", "perturb")
    assert buffer.getvalue() == ""


# --- answer_check == its three composed steps, which is what the task calls ---


@pytest.mark.parametrize(
    ("gold", "response"),
    [
        ("42", "so \\boxed{42}"),
        ("42", "so \\boxed{43}"),
        ("\\frac{1}{2}", "so \\boxed{0.5}"),
        ("2 or 3", "so \\boxed{2 \\text{ or } 3}"),
        ("(1,2)", "so \\boxed{(1, 2)}"),
        ("42", "no answer at all here"),
    ],
)
def test_answer_check_equals_the_composition_the_task_uses(gold, response):
    """The task grades from the stored extraction rather than the raw text.

    That is only the same measurement because ``answer_check`` is exactly these
    three steps and nothing else — asserted here so a change upstream that adds a
    fourth cannot pass silently.
    """
    composed = eval_math(
        {
            "answer": extract_ground_truth_answer("Q?", gold, "perturb"),
            "prediction": extract_predicted_answer("Q?", response),
        },
        prec=MAX_ABS_TOL,
    )
    assert bool(composed) == bool(answer_check("Q?", response, gold, "perturb"))


# --- grading behaviour ---


def test_symbolic_equivalence_still_reaches_sympy():
    assert math_equal("\\frac{1}{2}", "0.5")
    assert symbolic_equal("x + x", "2*x")


def test_multi_valued_answer_needs_every_atom_matched():
    assert eval_math({"answer": ["2", "3"], "prediction": ["2", "3"]}, prec=MAX_ABS_TOL)
    assert not eval_math(
        {"answer": ["2", "3"], "prediction": ["2", "4"]}, prec=MAX_ABS_TOL
    )


def test_eval_math_keeps_the_last_atoms_when_the_model_boxed_extra():
    """Upstream's ``pred = _pred[-len(ans):]`` — a boxed intermediate is dropped."""
    assert eval_math(
        {"answer": ["7"], "prediction": ["99", "7"]},
        prec=MAX_ABS_TOL,
    )


# --- execution safety: the one sieval divergence ---


def test_symbolic_equal_refuses_an_unparseable_answer_rather_than_sympifying_it():
    """Upstream returns the raw string here, which reaches ``N`` and executes."""
    payload = "__import__('os').getcwd()"
    assert symbolic_equal(payload, payload) is False


def test_a_payload_prediction_grades_wrong_without_running():
    payload = "__import__('os').system('true')"
    assert not answer_check("Q?", f"\\boxed{{{payload}}}", "42", "perturb")


# --- byte-level fidelity: what a diff and a behavioural corpus both miss ---

#: First 128 bits of each verbatim-ported function's sha256, over its source as
#: written -- computed from UPSTREAM's four files at the pinned commit, never from
#: this repo's copy, which would make the check tautological.
#:
#: Bytes rather than behaviour because behaviour missed it once: the EM SPACE
#: regression folded one key of `_fix_unicode`'s table to its ASCII lookalike, and
#: `ruff` (which skips this package), a unified diff, and an all-ASCII 9043-case
#: corpus were all blind to it. `UPSTREAM_UNICODE_REPLACEMENTS` pins that table;
#: this pins every function carrying no intentional divergence.
UPSTREAM_FUNCTION_SHA256 = {
    "_fix_a_slash_b": "932a4664bb1c52119b6cf44b4697843a",
    "_fix_fracs": "06796c7d6409b17c2054744512a4d90e",
    "_fix_sqrt": "423a16781943d814048e2ad60c787eb1",
    "_fix_tan": "81ca4ddc20bad4c254b25dd9128e0014",
    "answer_check": "4b2cc94dd4d082159ab88007a90185cb",
    "call_with_timeout": "7e41447ad6b385881b888aebb345e03a",
    "eval_math": "1c61c3dea602d0b079133ecaade15e5c",
    "extract_answer": "e9a3a04af17d2d2b8baedaa8b984a01b",
    "extract_boxed_answers": "674801311523f78e523a2e06aa9bb7c7",
    "extract_math_answer": "3d6264538ef51e173abd1b96675e6904",
    "extract_math_perturb_ground_truth_answer": "b114eeaa5367e521bd58503e3fafd792",
    "extract_program_output": "b256b99fadb6f681f4c3ce1be3a4d117",
    "is_digit": "5317e6ecb34cd9af8afd12f7a29c9851",
    "math_equal": "5087d32d97b3dde078cbaec5755f0e0e",
    "parse_digits": "ea0da0346a469bc837393fb75875c6f7",
    "parse_latex": "8c6832e62b2f49f5656d3ec0567878c1",
    "strip_string": "8a34124b26d0c9b0ab60fcf6b9821b08",
    "symbolic_equal_process": "70e12fc5e6daca5627b91acdb74ce35d",
}

#: The rest, with what each changes -- the module docstring gives the reasons.
#: Every ported function is in exactly one of the two maps, so both a new silent
#: divergence and a quietly dropped documented one fail the partition test.
DECLARED_DIVERGENT = {
    "_fix_unicode": "drops the `before` local and its DEBUG print",
    "extract_ground_truth_answer": "drops the 'Multi-valued ground truth:' print",
    "extract_predicted_answer": "drops the 'multi-valued prediction:' print",
    "is_correct": "drops the '2,3,4' debug print",
    "symbolic_equal": "execution safety: guarded parse, unparseable -> None",
}

#: Defined by this port, with no upstream counterpart.
PORT_ONLY = {"_guarded_parse_expr"}


def _ported_function_sources() -> dict[str, str]:
    """Top-level functions of the vendored module, as source text.

    Located via `inspect.getsourcefile`, so a stale copy elsewhere on the path
    cannot satisfy this.
    """
    import ast
    import inspect
    import pathlib

    from sieval.community import math_perturb

    path = inspect.getsourcefile(math_perturb)
    assert path is not None
    source = pathlib.Path(path).read_text(encoding="utf-8")
    sources: dict[str, str] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        # Only `None` for a node with no position info, impossible for a tree
        # parsed from this very text -- so it means a broken read.
        segment = ast.get_source_segment(source, node)
        assert segment is not None, f"no source segment for `{node.name}`"
        sources[node.name] = segment
    return sources


@pytest.mark.parametrize("name", sorted(UPSTREAM_FUNCTION_SHA256))
def test_a_verbatim_ported_function_is_byte_identical_to_upstream(name):
    """Whitespace and comments included: the regression that motivated this
    changed one character inside a string literal."""
    import hashlib

    source = _ported_function_sources()[name]
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]
    assert digest == UPSTREAM_FUNCTION_SHA256[name], (
        f"`{name}` is a verbatim port and no longer matches upstream's bytes. If "
        "the change is deliberate, move it to DECLARED_DIVERGENT and add it to "
        "the module docstring -- refreshing the hash records the drift as the new "
        "reference."
    )


def test_every_ported_function_is_pinned_or_declared_divergent():
    """A helper copied in, or a divergence added to a verbatim function, lands in
    neither map."""
    ported = set(_ported_function_sources()) - PORT_ONLY
    pinned = set(UPSTREAM_FUNCTION_SHA256)
    declared = set(DECLARED_DIVERGENT)
    assert pinned & declared == set(), "a function cannot be both"
    assert ported == pinned | declared
