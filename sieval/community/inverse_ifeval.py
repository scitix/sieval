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

Inverse IFEval measures "counter-intuitive" instruction following: 1,012
bilingual prompts (506 zh / 506 en) across eight challenge types whose
instructions deliberately conflict with the conventions models absorb during
SFT (answer correctly, be verbose, comment your code, format tidily). An LLM
judge grades each response pass/fail against a per-sample ``response_reference``.

Metric, recovered from the paper's tables (2 and 3, 22 models x 2 languages),
which state neither the aggregation nor the scale:

* **Overall is a pooled mean of the per-sample 0/1 verdicts**, x100 — not a
  macro-average over the eight types. Reconstructing every published Overall as
  the count-weighted mean of that row's eight cells matches to a mean absolute
  error of 0.002 (max 0.01, the tables' own 2-dp rounding) over all 44 rows;
  unweighted, it misses by 1.4 mean / 4.4 max. So ``score`` is
  ``100 * correct / total``, with no per-type reweighting.
* **The paper's ``CC`` and ``CCF`` labels are swapped** relative to the
  expansion of its own type list. Every cell is a multiple of ``1 / (6n)`` for
  the type's row count ``n``, fixing ``CC`` to n=41 (Counter-Conventional
  Formatting, denominator 246) and ``CCF`` to n=99 (Code without Comments, 594).
  Exact for 44 of 44 ``CC`` cells and 43 of 44 ``CCF`` — the miss prints 29.25
  where its grid implies 29.29 (DeepSeek-V3-0324 English, a table typo) —
  against 0 of 44 unswapped, where the pooled fit above misses by up to 7.3.
  :data:`PAPER_COLUMNS` records the corrected mapping.

That ``1 / (6n)`` quantization also implies the cells average **six rollouts per
sample**; 6 is the smallest multiplier fitting all eight columns (m=3 fits ITF,
DIA and MIM but leaves ``CC`` at 33 of 44). The paper states no repeat count, so
the task defaults to ``n=1`` and records the protocol in ``reference_impl.notes``.

Parsing. The judge is asked to end with a fenced ``{"answer_score": N}`` block,
N in {0, 1}, introduced by ``【JSON】``. :func:`parse_answer_score` reads that
block and nothing else, and is never applied to a prompt: **every shipped judge
system prompt ends with a worked example scoring 1**, so parsing one returns a
silent PASS. That is the one misread direction that inflates invisibly, which is
why the search is anchored after the judge's LAST ``【JSON】`` marker — no
fallback can reach back into an echo — and why an absent verdict stays
``score=None`` rather than defaulting to a tier.

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

# The judge's authoritative output is the fenced block after `【JSON】`, so that
# marker is the anchor: everything BEFORE the last one is prior reasoning or an
# echoed worked example. Every shipped example ends on a score of 1, so a rule
# that can reach behind the marker can only ever inflate.
_JSON_MARKER_RE = re.compile(r"【\s*JSON\s*】")
# Fenced blocks are still preferred WITHIN the anchored region, so a chatty
# aside that mentions the key after the verdict cannot outrank the block.
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
    collapse to ``_`` so an upstream label gaining punctuation cannot emit a
    report key that is awkward to consume.
    """
    return re.sub(r"[^a-z0-9]+", "_", instruction_type.lower()).strip("_")


def _match_last(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def _authoritative_region(reply: str) -> str:
    """The slice of *reply* the judge was told to put its verdict in.

    Everything after the LAST ``【JSON】`` marker, or the whole reply when the
    judge wrote none (a format deviation the fallbacks still read). Anchoring
    here is what keeps an echoed worked example out of reach: those sit before
    the judge's own marker, and all of them score 1.
    """
    markers = list(_JSON_MARKER_RE.finditer(reply))
    return reply[markers[-1].end() :] if markers else reply


def parse_answer_score(reply: str) -> tuple[int | None, str | None]:
    """Read the judge's ``answer_score`` out of *reply*.

    Returns ``(score, raw)``:

    * ``(0, "0")`` / ``(1, "1")`` — a verdict on the rubric.
    * ``(None, "100")`` — a number was found but it is off-rubric, so there is no
      verdict to record. The raw token is returned so the violation is
      inspectable rather than collapsing into "unparseable".
    * ``(None, None)`` — nothing matched: an empty reply, a refusal, or a
      reasoning judge that spent its whole budget thinking.

    Resolution is POSITIONAL first, format second: the search is confined to
    :func:`_authoritative_region`, and only inside it does the last fenced block
    win over a bare ``answer_score``. Preferring a fence globally would let any
    echoed worked example (all fenced, all scoring 1) outrank a real verdict the
    judge wrote without one. ``【评分】：N分`` is read from the whole reply, since
    the shipped layout puts it BEFORE the marker; a located-but-malformed block
    (``{"answer_score": null}``) stays ``(None, None)``, because a further search
    could only reach behind the marker.

    Only ever call this on a judge REPLY (the caller passes
    ``ModelOutput.texts[0]``) — every shipped ``judge_system_prompt`` ends with an
    example scoring 1, so parsing a prompt reads as PASS. One residual case is
    undecidable by position and left as such: a judge that echoes an example
    AFTER stating its own verdict.
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
    :func:`sieval.core.tasks.metrics.sampling_report`, whose ``pass@1`` over the
    requested denominator already *is* this benchmark's pooled mean (module
    docstring); a second overall here would be two names for one number.

    An unscored rollout (pipeline failure) has no language or type to attribute
    it to, so it is absent from these groups while still counting against the
    headline's denominator — the two agree exactly on a clean run, and the caller
    reports ``n_graded``/``fails`` so a partial one is visible.

    A group absent from *graded* is OMITTED rather than reported 0.0 (the
    :mod:`sieval.core.tasks.metrics` convention: a 0.0 for lack of input reads
    identically to a real one), and ``n_<group>`` accompanies each score so a
    sliced run's denominators can be checked against the paper's. Per language,
    in emission order — :data:`INSTRUCTION_TYPES`, which puts Code without
    Comments BEFORE Counter-Conventional Formatting — they are
    45 / 43 / 99 / 41 / 93 / 77 / 54 / 54; under the paper's labels the same
    counts read 45 / 43 / 41 / 99 / …, so comparing across the two orders
    transposes two denominators. Hence :data:`PAPER_COLUMNS`.
    """
    results: dict[str, float] = {}

    by_language: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for language, instruction_type, correct in graded:
        for bucket, key in ((by_language, language), (by_type, instruction_type)):
            bucket[key][0] += int(bool(correct))
            bucket[key][1] += 1

    # Iterate the declared vocabularies first so key order is stable across runs
    # and independent of which sample happened to be graded first; anything
    # upstream adds still surfaces, appended after the known names.
    for bucket, vocabulary in ((by_language, LANGUAGES), (by_type, INSTRUCTION_TYPES)):
        known = [name for name in vocabulary if name in bucket]
        extra = sorted(name for name in bucket if name not in vocabulary)
        for name in known + extra:
            correct_count, group_total = bucket[name]
            key = type_key(name)
            results[f"score_{key}"] = 100 * correct_count / group_total
            results[f"n_{key}"] = float(group_total)
    return results
