# Adapted from GSM-Plus (Li et al., ACL 2024), pinned commit:
# https://github.com/qtli/GSM-Plus/blob/3474129ec12fcd3e8ac08cb037aca1928efca98c/scripts/utils/extract_ans.py
#
# SPDX-License-Identifier: CC-BY-SA-4.0
#
# LICENSE: this file is CC-BY-SA-4.0, not the repository's Apache-2.0. Upstream's
# CODE repo states no license at the pinned commit (no LICENSE/COPYING/NOTICE, no
# SPDX, nothing in the README, `"license": null` from the GitHub API), so the
# CC-BY-SA-4.0 the GSM-Plus DATASET carries (per the `qintongli/GSM-Plus` HF card)
# is applied to the scoring code accompanying it, attributed above. Two things
# that do not follow automatically:
#
#   - Share-alike is per-file. It attaches here and to modifications of this
#     file; the importing task and the rest of sieval stay Apache-2.0.
#   - The dataset's grant covers the dataset. Reading it onto the code is this
#     project's construction, not a grant the code's authors published — recorded
#     as the basis relied on, not as a quoted licence.
#
# Parts with independent permissive provenance: SUBSTITUTIONS /
# REMOVED_EXPRESSIONS / normalize_final_answer are Minerva (Lewkowycz et al., per
# upstream's own attribution comment below, and also shipped in
# lm-evaluation-harness under Apache-2.0); delete_extra_zero is MetaMath (MIT).
"""
GSM-Plus answer extraction and answer equivalence (zero-shot CoT protocol).

Faithful port of the GSM-Plus scoring path from the pinned commit
(``scripts/utils/extract_ans.py``), serving ``sieval.tasks.gsm_plus_0shot_gen``.

GSM-Plus scores two kinds of item with two different extractors, dispatched on
the sample's ``perturbation_type`` (upstream ``test_answer``):

* every perturbation except ``critical thinking`` — ``extract_pred_ans`` pulls a
  number out of the reasoning: the last ``#### ...`` segment containing a digit,
  else the last number anywhere in the text.
* ``critical thinking`` — the seed problem had a required quantity *deleted*, so
  the gold answer is the literal string ``"None"`` (meaning "unanswerable").
  A number would be meaningless, so ``extract_pred_ans_none`` looks for a
  refusal phrase instead.

Both then run through ``is_equivalent``: normalize (``normalize_final_answer``),
then compare symbolically (``check_sympy_equivalence``).

``extract_prediction`` / ``is_equivalent`` are upstream's ``test_answer`` (its
``mv == 1`` path) split at sieval's postprocess/feedback boundary. The order of
operations is unchanged, so verdicts are identical; only the seam moved.

Fidelity is measured against upstream's own stored predictions for both models it
published on this path; the numbers live on ``gsm_plus_0shot_gen``.

Deviations from upstream (documented, not silent):

- **Only the ``prompt_type == "cot"`` branch is ported.** Upstream's
  ``extract_pred_ans`` multiplexes ten prompting techniques (``pot``,
  ``complex``, ``ltm``, ``llama``, ``codellama``, ``sego``, ``mammoth``,
  ``metamath``, ``tora``, plus a generic ``match_pattern`` fallback), three of
  which ``exec()`` model-generated Python. Only the zero-shot CoT protocol is
  implemented, so those branches and the program-execution machinery they need
  (``safe_execute`` / ``synthesize_program_*`` / ``func_timeout``) are
  deliberately not vendored, and ``prompt_type`` disappears from the signatures
  rather than becoming an argument that accepts one value.
- **Self-consistency (``mv > 1``) is not ported.** Upstream's ``cot_sc``
  majority-votes 5 samples at temperature 0.7; this is the ``mv == 1`` path.
- ``extract_gold_ans`` raises ``ValueError`` where upstream calls
  ``pdb.set_trace()`` (gold with neither ``####`` nor ``boxed{}``, and a
  non-``"None"`` gold containing no number) — a debugger is not library
  behaviour. Both are unreachable at the pinned dataset revision, where every
  ``solution`` ends in ``#### <number|None>``.
- Regex/substitution literals are spelled as raw strings. The string *values* are
  byte-identical; this only avoids the invalid-escape ``SyntaxWarning`` upstream
  emits under Python 3.12.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import re

import sympy
from sympy.parsing.latex import parse_latex

# The gold answer of a `critical thinking` item, and the extracted prediction
# that matches it: upstream's spelling of "this question is unanswerable". It is
# a real answer, not the protocol's `None` ("could not extract").
NONE_ANSWER = "None"

# The only perturbation type named in code, being the only one extraction
# dispatches on; the other seven are read off the sample as report keys, never
# compared against a literal. (Spelling drift, should a dump ever be replayed:
# upstream's `results/*.json` say "distractor insertion" where the released
# dataset — what a run sees — says "distraction insertion".)
CRITICAL_THINKING = "critical thinking"

# Part of the code is modified from the code snippets provided in "Solving Quantitative Reasoning Problems with Language Models" by Lewkowycz et al.
SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), (r'\%', '%'),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
    ('\\text{and}', ','), ('\\text{m}', '\\text{}')
]
REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
    r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
]

# The number GSM-Plus reads out of a gold or a prediction: optionally signed,
# optionally decimal, optionally a fraction ("-3", "4.33", "3/5").
_NUMBER_PATTERN = r'-?\d+(?:\.\d+)?(?:/\d+)?'


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split('=')[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(
        r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(
        r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer


def delete_extra_zero(n):
    '''删除小数点后多余的0'''
    try:
        n=float(n)
    except:
        # print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  # 删除小数点后多余的0
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
        n=str(n)
        return n


def check_sympy_equivalence(formatted_target_str, formatted_prediction_str):
    formatted_target_str = delete_extra_zero(formatted_target_str)
    formatted_prediction_str = delete_extra_zero(formatted_prediction_str)

    flag = False
    try:
        target_expr = parse_latex(formatted_target_str)
    except:
        target_expr = formatted_target_str
        flag = True

    try:
        prediction_expr = parse_latex(formatted_prediction_str)
    except:
        prediction_expr = formatted_prediction_str
        flag = True

    if flag == True:
        return formatted_target_str == formatted_prediction_str

    try:
        return sympy.simplify(target_expr - prediction_expr) == 0
    except:
        return False


def extract_gold_ans(answer_str):
    """Read the gold answer out of a GSM-Plus ``solution``.

    ``"...#### 4.33"`` -> ``"4.33"``; a `critical thinking` solution ends in
    ``"#### None"`` -> ``"None"``.
    """
    answer_str = answer_str.strip("\n").strip(" ").rstrip(".").replace(",", "")
    pattern = "####(.*)"
    if len(re.findall(pattern, answer_str)) >= 1:
        target = re.findall(pattern, answer_str)[-1].strip(' ')
    else:
        pattern = "boxed{(.*)}"
        if len(re.findall(pattern, answer_str)) < 1:
            # Upstream: pdb.set_trace()
            raise ValueError(
                f"GSM-Plus gold answer has neither '####' nor 'boxed{{}}': {answer_str!r}"
            )
        target = re.findall(pattern, answer_str)[-1].strip(' ')
    if target != NONE_ANSWER:
        if len(re.findall(_NUMBER_PATTERN, target)) < 1:
            # Upstream: print(answer_str); pdb.set_trace()
            raise ValueError(
                f"GSM-Plus gold answer is neither a number nor {NONE_ANSWER!r}: {answer_str!r}"
            )
        temp_ans = re.findall(_NUMBER_PATTERN, target)[0]
        temp_ans = delete_extra_zero(temp_ans)
    else:
        temp_ans = NONE_ANSWER
    return temp_ans


def extract_pred_ans(pred_str):
    """Extract a numeric prediction (upstream ``prompt_type == "cot"`` branch)."""
    pred_str = pred_str.rstrip(".").replace(",", "")

    pattern = "####(.*)"
    if "Question" in pred_str:
        pred_str = pred_str.split("Question")[0]
    preds = re.findall(pattern, pred_str)
    pred = delete_extra_zero(preds[-1].strip(" ")) if len(preds) >= 1 and bool(re.search(r"\d", preds[-1])) else ""
    if pred == "":
        pred = re.findall(_NUMBER_PATTERN, pred_str)
        if len(pred) >= 1:
            pred = delete_extra_zero(pred[-1].replace(",", "").strip(".").strip(" "))
        else:
            pred = ""
    else:
        pred = delete_extra_zero(re.findall(_NUMBER_PATTERN, pred.replace(",", ""))[0].strip(".").strip(" "))
    if "</s>" in pred:
        pred = pred[:-4]

    pred = pred.rstrip(".").strip(" ")
    return pred


# Phrases that count as "the model recognized the question is unanswerable".
# Verbatim from upstream, order preserved (any hit wins, so order is cosmetic).
_NONE_PATTERNS = ["does not provide enough information", "does not specify", "does not provide", "can't provide", "can not provide", "don't know", "do not know", "doesn't specify", "not specify", "not mention", "doesn't mention", "don't have enough information", "do not have enough", "not provide", "doesn't provide", "cannot calculate", "can't calculate", "can't determine", "cannot determine", "missing necessary information", "none"]


def extract_pred_ans_none(pred_str):
    """Extract a `critical thinking` prediction (upstream ``"cot" in prompt_type``).

    Returns ``"None"`` (the model recognized the question as unanswerable) or
    ``""`` (it answered anyway).

    The second branch is upstream's verbatim, and is the format gate: a response
    with **no** ``####`` marker scores ``"None"`` — i.e. correct — even when it
    confidently computed a number. A leniency toward models that ignore the
    output format, load-bearing for reproducing upstream's published numbers, and
    the reason a format-compliant model is judged on refusal phrasing alone.
    """
    pred_str = pred_str.rstrip(".").replace(",", "").lower()
    pred = ""
    for p in _NONE_PATTERNS:
        if p in pred_str:
            pred = NONE_ANSWER
    match_pattern = "####"
    if pred != NONE_ANSWER and match_pattern not in pred_str:
        pred = NONE_ANSWER
    return pred


def extract_prediction(pred_str, perturbation_type):
    """Extract the answer from *pred_str*, dispatching on *perturbation_type*.

    Upstream ``test_answer``'s ``mv == 1`` extraction half.
    """
    if perturbation_type == CRITICAL_THINKING:
        return extract_pred_ans_none(pred_str)
    return extract_pred_ans(pred_str)


def is_equivalent(gold, pred):
    """Whether *pred* answers *gold*. Upstream ``test_answer``'s scoring half."""
    if gold != NONE_ANSWER:
        gold = normalize_final_answer(gold)
    if pred != NONE_ANSWER:
        pred = normalize_final_answer(str(pred))
    return check_sympy_equivalence(gold, pred)
