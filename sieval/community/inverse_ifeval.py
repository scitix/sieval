# Sources (Inverse IFEval, M-A-P; Zhang et al., 2025, arXiv:2509.04292), pinned to
# Hub revision 35f1da157640526e62b7685b682d748fa55ccfd0:
#   - https://huggingface.co/datasets/m-a-p/Inverse_IFEval — `Inverse_IFEval_Dataset.json`
#     ships each sample's own `judge_system_prompt` and `judge_prompt_template`.
#   - The 0/1 rubric, the `【评分依据】/【评分】/【JSON】` reply layout, and the
#     `{"answer_score": N}` contract are stated inside those shipped judge system
#     prompts. Upstream publishes NO evaluation harness, so there is no reference
#     implementation to vendor: this module is the reply contract read off the
#     prompts, plus the metric recovered from the paper's tables (see below).
"""Inverse IFEval judge-reply parsing + metric aggregation.

An LLM judge grades each response 0/1 against the sample's own
``response_reference``; the dataset itself (1,012 prompts, 506 zh / 506 en, eight
challenge types) is described in :mod:`sieval.datasets.inverse_ifeval`.

Metric, recovered from Tables 2/3 (22 models x 2 languages), which state neither
the aggregation nor the scale:

* **Overall is a pooled mean of the 0/1 verdicts**, x100, not a macro-average:
  reconstructing every published Overall as the count-weighted mean of its eight
  cells fits to MAE 0.002 (max 0.01, the tables' own 2-dp rounding) over all 44
  rows,
  where unweighted misses by 1.4 mean / 4.4 max. So ``score`` is
  ``100 * correct / total``.
* **The paper's ``CC``/``CCF`` labels are swapped** against its own type list.
  Every cell is a multiple of ``1 / (6n)`` for the type's row count, fixing ``CC``
  to n=41 (Counter-Conventional Formatting, denominator 246) and ``CCF`` to n=99
  (Code without Comments, 594): exact for 44/44 ``CC`` and 43/44 ``CCF`` (the miss
  prints 29.25 where its grid implies 29.29 — DeepSeek-V3-0324 English, a typo),
  against 0/44 unswapped, where the pooled fit misses by up to 7.3.
  :data:`PAPER_COLUMNS` records it.

The same quantization implies **six rollouts per sample**; 6 is the smallest
multiplier fitting all eight columns (m=3 leaves ``CC`` at 33/44). The paper
states no repeat count, so the task defaults to ``n=1`` and records the protocol
in its ``reference_impl.notes``.

Parsing. The judge ends with a fenced ``{"answer_score": N}``, N in {0, 1}, after
``【JSON】``. **Every shipped judge system prompt ends with a worked example
scoring 1**, so parsing a prompt — or an echo of one — returns a silent PASS: the
one misread direction that inflates invisibly. Hence the anchor on the LAST
``【JSON】`` marker, and an absent verdict staying ``score=None``.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import re
from collections import defaultdict
from collections.abc import Sequence

#: The eight challenge types, in the paper's Section 2.1 order.
INSTRUCTION_TYPES: tuple[str, ...] = (
    "Question Correction",
    "Intentional Textual Flaws",
    "Code without Comments",
    "Counter-Conventional Formatting",
    "Deliberately Incorrect Answers",
    "Instructional Induction",
    "Mid-turn Instruction Modification",
    "Counterfactual Answering",
)

#: The two language subsets, each 506 samples at the pinned revision.
LANGUAGES: tuple[str, ...] = ("english", "chinese")

#: Paper Table 2/3 column label -> dataset ``instruction_types`` value.
#: ``CC``/``CCF`` are swapped relative to the intuitive expansion; see the module
#: docstring for how the row counts prove it.
PAPER_COLUMNS: dict[str, str] = {
    "QC": "Question Correction",
    "ITF": "Intentional Textual Flaws",
    "CC": "Counter-Conventional Formatting",
    "CCF": "Code without Comments",
    "DIA": "Deliberately Incorrect Answers",
    "II": "Instructional Induction",
    "MIM": "Mid-turn Instruction Modification",
    "CA": "Counterfactual Answering",
}

#: The rubric is two-tier; anything else is a rubric violation, not a score.
VALID_SCORES: frozenset[int] = frozenset({0, 1})

# The authoritative output is the fenced block after `【JSON】`, so the LAST marker
# anchors the search: anything before it is prior reasoning or an echoed example,
# and every shipped example scores 1.
_JSON_MARKER_RE = re.compile(r"【\s*JSON\s*】")
# Preferred WITHIN that region, so a chatty aside naming the key after the verdict
# cannot outrank the block.
_FENCE_RE = re.compile(r"```[A-Za-z]*\s*\n?(.*?)```", re.DOTALL)
# `"answer_score": 1`, tolerating unquoted keys, full-width colons, and floats
# (a judge writing `1.0` still means the `1` tier).
_ANSWER_SCORE_RE = re.compile(
    r"[\"']?answer_score[\"']?\s*[:：]\s*[\"']?(-?\d+(?:\.\d+)?)"
)
# Last-resort: the human-readable `【评分】：1分` line the same prompts mandate.
_SCORE_LINE_RE = re.compile(r"【\s*评分\s*】\s*[:：]?\s*(-?\d+(?:\.\d+)?)\s*分?")


def type_key(instruction_type: str) -> str:
    """Slugify an ``instruction_types`` value into a report key fragment.

    ``"Code without Comments"`` -> ``"code_without_comments"``. Non-alphanumerics
    collapse to ``_``, so an upstream label gaining punctuation cannot emit an
    awkward report key.
    """
    return re.sub(r"[^a-z0-9]+", "_", instruction_type.lower()).strip("_")


def _match_last(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def _authoritative_region(reply: str) -> str:
    """The slice of *reply* the judge was told to put its verdict in.

    Everything after the LAST ``【JSON】`` marker, or the whole reply when the judge
    wrote none — a format deviation the fallbacks still read.
    """
    markers = list(_JSON_MARKER_RE.finditer(reply))
    return reply[markers[-1].end() :] if markers else reply


def parse_answer_score(reply: str) -> tuple[int | None, str | None]:
    """Read the judge's ``answer_score`` out of *reply*.

    Returns ``(score, raw)``:

    * ``(0, "0")`` / ``(1, "1")`` — a verdict on the rubric.
    * ``(None, "100")`` — off-rubric, so no verdict; the raw token is returned so
      the violation stays inspectable rather than collapsing into "unparseable".
    * ``(None, None)`` — nothing matched: empty reply, refusal, or a reasoning
      judge that spent its whole budget thinking.

    Resolution is POSITIONAL first: confined to :func:`_authoritative_region`, and
    only inside it does the last fenced block win over a bare ``answer_score`` —
    preferring a fence globally would let an echoed example (all fenced, all
    scoring 1) outrank a verdict written without one. ``【评分】：N分`` is read from
    the whole reply, since the shipped layout puts it BEFORE the marker; a
    located-but-malformed block (``{"answer_score": null}``) stays ``(None, None)``
    rather than reaching behind the marker.

    Only ever call this on a judge REPLY (the caller passes
    ``ModelOutput.texts[0]``): a prompt reads as PASS. One case stays undecidable
    by position — a judge that echoes an example AFTER its own verdict.
    """
    region = _authoritative_region(reply)
    fenced = [block for block in _FENCE_RE.findall(region) if "answer_score" in block]
    if fenced:
        raw = _match_last(_ANSWER_SCORE_RE, fenced[-1])
    else:
        raw = _match_last(_ANSWER_SCORE_RE, region)
        if raw is None:
            raw = _match_last(_SCORE_LINE_RE, reply)
    if raw is None:
        return None, None

    value = float(raw)
    if value.is_integer() and int(value) in VALID_SCORES:
        return int(value), raw
    return None, raw


def breakdown_metrics(graded: Sequence[tuple[str, str, bool]]) -> dict[str, float]:
    """Per-language and per-type pass rates, x100.

    *graded* is one ``(language, instruction_type, correct)`` triple per judged
    rollout. Breakdowns only — the headline stays with
    :func:`sieval.core.tasks.metrics.sampling_report`, whose ``pass@1`` already
    *is* this benchmark's pooled mean, and a second overall here would be two
    names for one number. An unscored rollout has no language or type to attribute
    it to, so it is absent from these groups while still counting against that
    headline's denominator; the caller reports ``n_graded``/``fails``.

    A group absent from *graded* is OMITTED rather than reported 0.0 (the
    :mod:`sieval.core.tasks.metrics` convention), and ``n_<group>`` accompanies
    each score so a sliced run's denominators stay checkable against the paper's.
    Per language, in emission order — :data:`INSTRUCTION_TYPES`, which puts Code
    without Comments BEFORE Counter-Conventional Formatting — they are
    45 / 43 / 99 / 41 / 93 / 77 / 54 / 54, where the paper's labels read
    45 / 43 / 41 / 99 / …: comparing across the two orders transposes two
    denominators. Hence :data:`PAPER_COLUMNS`.
    """
    results: dict[str, float] = {}

    by_language: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for language, instruction_type, correct in graded:
        for bucket, key in ((by_language, language), (by_type, instruction_type)):
            bucket[key][0] += int(bool(correct))
            bucket[key][1] += 1

    # Declared vocabularies first, so key order does not depend on which sample was
    # graded first; anything upstream adds still surfaces, appended after them.
    for bucket, vocabulary in ((by_language, LANGUAGES), (by_type, INSTRUCTION_TYPES)):
        known = [name for name in vocabulary if name in bucket]
        extra = sorted(name for name in bucket if name not in vocabulary)
        for name in known + extra:
            correct_count, group_total = bucket[name]
            key = type_key(name)
            results[f"score_{key}"] = 100 * correct_count / group_total
            results[f"n_{key}"] = float(group_total)
    return results
