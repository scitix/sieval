"""ComplexConstraints rubric-grading assets: judge prompt, verdict parsing, metrics.

ComplexConstraints (Mehta et al., 2026, arXiv:2606.09118) is a 75-prompt
instruction-following benchmark. Each prompt ships 10-40 *atomic* rubric criteria
(1,559 in total) describing what a correct response must satisfy; criteria are
graded by rubric -- human or LLM-as-a-judge -- never by exact match.

Upstream publishes **no evaluation code and no judge prompt**. The paper names
GPT-5-mini as the per-criterion judge and defines the metrics, but the template,
its decoding settings, and the call structure are all unstated, and the dataset
card adds nothing. So ``GRADER_TEMPLATE`` and :func:`parse_verdicts` below are
**authored by this port**, not reproduced from upstream -- which is why
``sieval.tasks.complex_constraints_0shot_gen`` ships ``status="experimental"``.
Contrast ``sieval.community.aa_lcr``, whose templates at least come from the
upstream dataset card verbatim.

Two published metrics, both computed by :func:`aggregate_metrics`:

* **task pass rate** -- the fraction of prompts whose response satisfies *every*
  criterion. This is what the paper's public 75-prompt leaderboard reports
  (its Table 1, snapshot 2026-06-03), so it is the port's headline.
* **mean per-criterion pass rate** -- "the fraction of rubric criteria satisfied,
  averaged across tasks" (its Table 3 caption), i.e. a **macro** average over
  prompts. Criteria counts vary 10-40, so the pooled (**micro**) rate is a
  genuinely different number; both are reported and named, and the macro one is
  the published one.

Grading is **one judge call per rollout**, covering all of that prompt's criteria,
with the verdicts emitted as an indexed list. Upstream never says whether it
grades one criterion per call; batching keeps a rollout's whole verdict set in a
single persisted ``ModelOutput`` -- which is also what the runner's grader-spend
accounting expects, since it reads exactly one output per rollout -- and the
indexing makes misalignment detectable: an index the judge never emits is
recorded as unparsed rather than silently shifting its neighbours' verdicts.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import re
from collections.abc import Sequence

#: One rubric line as the judge sees it. 1-based, matching the verdict indices
#: the judge is asked to emit.
CRITERION_TEMPLATE = "{number}. {criterion}"

#: Rubric-grading prompt. Authored by this port (upstream publishes none).
#: The verdict block is requested *last* so a reasoning judge puts it after its
#: deliberation, which is what makes "last verdict per index wins" correct.
GRADER_TEMPLATE = """You are grading one model RESPONSE against a rubric of atomic criteria.

Judge each criterion independently, against the RESPONSE alone. A criterion is satisfied
only if the RESPONSE clearly meets it. If the response only partially meets a criterion, or
gives you nothing to check it against, that criterion is NOT satisfied. Grade exactly what
each criterion asks for -- do not reward or penalise anything else about the response.

BEGIN PROMPT
{prompt}
END PROMPT

BEGIN RESPONSE
{response}
END RESPONSE

BEGIN CRITERIA
{criteria}
END CRITERIA

Grade all {n_criteria} criteria. End your reply with one verdict per criterion, in order,
one per line, in exactly this format and nothing after it:

1: <PASS|FAIL>
2: <PASS|FAIL>
...
{n_criteria}: <PASS|FAIL>
"""


def format_criteria(criteria: Sequence[str]) -> str:
    """Render *criteria* as the 1-based numbered block the judge grades."""
    return "\n".join(
        CRITERION_TEMPLATE.format(number=i + 1, criterion=criterion)
        for i, criterion in enumerate(criteria)
    )


def build_grader_prompt(prompt: str, response: str, criteria: Sequence[str]) -> str:
    """Assemble the rubric-grading prompt for one response.

    The original *prompt* is included because criteria are written against it
    ("the response should schedule ... 15th-21st December 2025"): many are
    uncheckable from the response alone.
    """
    return GRADER_TEMPLATE.format(
        prompt=prompt,
        response=response,
        criteria=format_criteria(criteria),
        n_criteria=len(criteria),
    )


# A verdict line: leading list/emphasis punctuation ("- ", "* ", "**"), an
# optional "criterion" word, the 1-based index, a separator, more optional
# emphasis, then the verdict. Anchored to line starts so prose that merely
# mentions a number cannot register as a verdict.
_VERDICT_RE = re.compile(
    r"^[^\w\n]*(?:criterion\s*)?(\d{1,3})\s*[:.)\-]\s*[^\w\n]*(PASS|FAIL)\b",
    re.IGNORECASE | re.MULTILINE,
)


def parse_verdicts(reply: str, n_criteria: int) -> list[bool | None]:
    """Map a judge reply to one verdict per criterion, in criterion order.

    Returns a list of length *n_criteria*: ``True`` (satisfied), ``False`` (not
    satisfied), or ``None`` for a criterion the judge never returned a readable
    verdict for. ``None`` is deliberately distinct from ``False`` -- the caller
    scores it as not-satisfied (an unreadable verdict must not inflate a score)
    but records the count separately, so judge format drift stays visible
    instead of masquerading as a model that failed the rubric.

    The **last** verdict for an index wins: the judge is asked to put the verdict
    block at the end, so a reasoning judge's earlier tentative pass over the
    criteria must not override its final answer. Indices outside ``1..n_criteria``
    are ignored rather than clamped -- a hallucinated "41: PASS" is not evidence
    about criterion 41 of a 40-criterion rubric.
    """
    verdicts: list[bool | None] = [None] * n_criteria
    for index_text, verdict in _VERDICT_RE.findall(reply):
        index = int(index_text)
        if 1 <= index <= n_criteria:
            verdicts[index - 1] = verdict.upper() == "PASS"
    return verdicts


def aggregate_metrics(units: Sequence[tuple[int, int]]) -> dict[str, float]:
    """Aggregate ``(n_satisfied, n_criteria)`` pairs into the published metrics.

    One *unit* is one graded rollout, plus one stand-in per attempt that never
    produced a gradeable response (contributing ``(0, n_criteria)``) so the rates
    span the full requested set rather than only the successfully-graded subset.

    Returns rates in ``[0, 1]``:

    * ``task_pass_rate`` -- units satisfying every criterion. The leaderboard's
      metric, and the port's headline.
    * ``criterion_pass_rate_macro`` -- per-unit satisfied fraction, averaged over
      units. The paper's "mean per-criterion pass rate".
    * ``criterion_pass_rate_micro`` -- criteria satisfied pooled over all units.
      Differs from the macro rate because criteria counts vary 10-40 per prompt.

    A unit with ``n_criteria == 0`` (a failure whose rubric size could not be
    recovered) counts as a task failure at rate 0 and adds nothing to the pooled
    denominator -- so it can only ever drag the score down, never flatter it.
    """
    total = len(units)
    if total == 0:
        return {
            "task_pass_rate": 0.0,
            "criterion_pass_rate_macro": 0.0,
            "criterion_pass_rate_micro": 0.0,
        }

    pooled_criteria = sum(count for _, count in units)
    return {
        "task_pass_rate": sum(
            1 for satisfied, count in units if count > 0 and satisfied == count
        )
        / total,
        "criterion_pass_rate_macro": sum(
            satisfied / count if count else 0.0 for satisfied, count in units
        )
        / total,
        "criterion_pass_rate_micro": (
            sum(satisfied for satisfied, _ in units) / pooled_criteria
            if pooled_criteria
            else 0.0
        ),
    }
