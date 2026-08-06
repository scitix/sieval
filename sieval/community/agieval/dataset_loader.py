# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# adapted from https://github.com/ruixiangcui/AGIEval/blob/84ab72d94318290aad2e4ec820d535a95a1f7552/src/dataset_loader.py
"""AGIEval subset taxonomy + zero-shot prompt construction.

Two things upstream keeps in ``src/dataset_loader.py`` and every other AGIEval
module keys off:

* the **subset families** — four disjoint tuples (english/chinese × qa/cloze)
  that decide prompt language and answer-parsing rules, plus the two scoring
  overrides (``MULTI_CHOICE_SUBSETS``, ``MATH_OUTPUT_SUBSETS``);
* the **zero-shot prompt** (``convert_zero_shot``) and the **second-stage
  answer-extraction prompt** (``generate_second_stage_input``). Upstream's
  zero-shot protocol is two calls: the model answers freely, then a second call
  re-reads its own answer under a "the answer is" cue so a short, parseable
  letter/value can be extracted. Both stages are needed to reproduce AGIEval's
  published zero-shot numbers.

Kept out on purpose: the few-shot paths (``combine_prompt`` / ``concat_prompt``
/ ``convert_few_shot``) and their ``tiktoken`` budget trimming — sieval ships
the zero-shot task only, and dead vendored code rots.

Deltas from upstream, all deliberate:

* ``convert_zero_shot`` / ``generate_second_stage_input`` are per-sample
  functions here (upstream's operate on whole files) and take the subset name
  first; the emitted strings are byte-identical.
* An unknown subset raises ``ValueError``. Upstream wraps the family dispatch in
  ``try/except NameError`` and returns ``None`` for a name in no family, which
  surfaces much later as a ``TypeError`` on the prompt.
* ``MATH_SUBSETS`` is a **sieval** grouping, not upstream's; see its comment.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from collections.abc import Mapping

# Subset families, verbatim from upstream (order included: it is the order the
# families are declared in, and sieval reuses it as the canonical subset order).
# `gaokao-english` sits in the ENGLISH family because its *prompt* is English —
# the exam is Chinese, and upstream's own leaderboard averages count it as
# Chinese. Do not conflate the two groupings; see evaluation.py.
ENGLISH_QA_SUBSETS: tuple[str, ...] = (
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "logiqa-en",
    "sat-math",
    "sat-en",
    "aqua-rat",
    "sat-en-without-passage",
    "gaokao-english",
)
CHINESE_QA_SUBSETS: tuple[str, ...] = (
    "logiqa-zh",
    "jec-qa-kd",
    "jec-qa-ca",
    "gaokao-chinese",
    "gaokao-geography",
    "gaokao-history",
    "gaokao-biology",
    "gaokao-chemistry",
    "gaokao-physics",
    "gaokao-mathqa",
)
ENGLISH_CLOZE_SUBSETS: tuple[str, ...] = ("math",)
CHINESE_CLOZE_SUBSETS: tuple[str, ...] = ("gaokao-mathcloze",)

#: Answers are compared as *sets* of letters, not strings (upstream's
#: `multi_choice_datasets`). `gaokao-physics` stayed on this list after v1.1
#: made its labels single-answer — kept, because set-vs-set and string-vs-string
#: agree on single letters.
MULTI_CHOICE_SUBSETS: tuple[str, ...] = ("jec-qa-kd", "jec-qa-ca", "gaokao-physics")

#: Answers are compared by math equivalence, not string equality (upstream's
#: `math_output_datasets`). Same membership as the two cloze families.
MATH_OUTPUT_SUBSETS: tuple[str, ...] = ("gaokao-mathcloze", "math")

#: All 21 data files under `data/v1_1`, in upstream's family-declaration order.
#: The AGIEval paper and README say "20 tasks" — they count `sat-en` and
#: `sat-en-without-passage` as one task in two prompt variants, while upstream's
#: own driver script evaluates and averages over all 21 files.
SUBSETS: tuple[str, ...] = (
    ENGLISH_QA_SUBSETS + CHINESE_QA_SUBSETS + ENGLISH_CLOZE_SUBSETS + CHINESE_CLOZE_SUBSETS
)

#: sieval-defined grouping — upstream has no "math" group. The five subsets
#: drawn from mathematics exams: SAT math, AQuA-RAT algebraic word problems,
#: Gaokao math (MCQ + cloze), and MATH competition problems. Everything else in
#: AGIEval is language/logic/law/science.
MATH_SUBSETS: tuple[str, ...] = (
    "sat-math",
    "aqua-rat",
    "gaokao-mathqa",
    "math",
    "gaokao-mathcloze",
)

_OPTION_LETTERS = "ABCDEFG"

# Second-stage cue per family, verbatim from `generate_second_stage_input`
# (with_format_prompt=False, the setting upstream's run_prediction.py uses).
# The hardcoded "A through E" / "A到D" do not track the actual option count —
# upstream's text, kept as-is.
_SECOND_STAGE_CUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (ENGLISH_QA_SUBSETS, "Therefore, among A through E, the answer is"),
    (CHINESE_QA_SUBSETS, "因此，从A到D, 我们应选择"),
    (ENGLISH_CLOZE_SUBSETS, "Therefore, the answer is"),
    (CHINESE_CLOZE_SUBSETS, "因此，答案是"),
)


def zero_shot_prompt(subset: str, row: Mapping) -> str:
    """Upstream ``convert_zero_shot(line, dataset_name)`` for one row.

    *row* needs ``passage`` / ``question`` / ``options`` (the AGIEval sample
    fields); ``options`` may be empty for the cloze subsets, which do not use it.
    """
    passage = row["passage"] if row["passage"] is not None else ""
    question = row["question"]
    options = row["options"] or []

    if subset in ENGLISH_QA_SUBSETS:
        count = len(options)
        if count == 1:
            count = 5
        return (
            passage
            + "Q: "
            + question
            + " "
            + "Answer Choices: "
            + " ".join(options)
            + "\n"
            + "A: Among A through {}, the answer is".format(_OPTION_LETTERS[count - 1])
        )
    if subset in CHINESE_QA_SUBSETS:
        count = len(options)
        if count == 1:
            count = 4
        return (
            passage
            + "问题："
            + question
            + " "
            + "选项："
            + " ".join(options)
            + "\n"
            + "答案：从A到{}, 我们应选择".format(_OPTION_LETTERS[count - 1])
        )
    if subset in ENGLISH_CLOZE_SUBSETS:
        return passage + "Q: " + question + "\nA: The answer is"
    if subset in CHINESE_CLOZE_SUBSETS:
        return passage + "问题：" + question + "\n答案："
    raise ValueError(f"Unknown AGIEval subset {subset!r}; expected one of {SUBSETS}")


def second_stage_prompt(subset: str, context: str, first_stage_output: str) -> str:
    """Upstream ``generate_second_stage_input`` for one row.

    *context* is the first-stage prompt and *first_stage_output* the model's
    reply to it; the cue that follows asks for the answer alone, which is what
    :func:`sieval.community.agieval.post_process.post_process` parses.
    """
    for subsets, cue in _SECOND_STAGE_CUES:
        if subset in subsets:
            return "{0}\n{1}\n{2}".format(context, first_stage_output, cue)
    raise ValueError(f"Unknown AGIEval subset {subset!r}; expected one of {SUBSETS}")
