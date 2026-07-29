# prompt template + equality-checker prompt: AA-LCR dataset card (Apache-2.0),
# reproduced verbatim from the "Prompt Template" and "Scoring Approach" sections
# at the pinned revision:
# https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/blob/bdae010bbce259820c0e34c1d7cce210d966fb75/README.md
"""Artificial Analysis Long Context Reasoning (AA-LCR) prompt + scoring assets.

AA-LCR (Artificial Analysis) is a 100-question long-context reasoning benchmark:
each question is answered against a set of real-world documents (~100k tokens per
set) loaded into the same prompt, and answers must be *reasoned* across multiple
sources rather than directly retrieved.

The upstream repo ships no evaluation code — only the dataset card. Both the
input ``PROMPT_TEMPLATE`` (with per-document ``BEGIN DOCUMENT``/``END DOCUMENT``
wrapping via ``DOCUMENT_TEMPLATE``) and the ``GRADER_TEMPLATE`` below are the
card's own snippets reproduced verbatim. Implementation notes:

* ``build_prompt`` mirrors the card's two-step assembly (join documents, then
  fill the outer template); documents are joined in the given order — the card
  requires ``data_source_filenames`` order, which the loader preserves.
* Grading is **binary** ``CORRECT``/``INCORRECT`` (no ``NOT_ATTEMPTED`` tier)
  and the card requires the candidate be *consistent with* the official answer.
  So ``parse_grade`` maps both an unrecognized reply and a qualified verdict
  (``not correct``, ``partially correct``) to ``INCORRECT``, and
  ``aggregate_metrics`` reports plain accuracy — unlike the SimpleQA-style
  A/B/C + F1 aggregation in ``sieval.community.simpleqa_verified``.
* The port's one score-affecting deviation from the card lives in the task
  module (``sieval.tasks.aa_lcr_0shot_gen``): empty/whitespace candidates are
  graded INCORRECT without invoking the checker.

The official equality checker is Qwen3 235B A22B 2507 Non-reasoning; the checker
model is a runtime choice supplied by the task, not pinned here.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import re

# Per-document wrapper; ``index`` is 0-based, printed 1-based per the card.
DOCUMENT_TEMPLATE = "BEGIN DOCUMENT {number}:\n{document}\nEND DOCUMENT {number}"

# Outer prompt framing the joined documents and the question.
PROMPT_TEMPLATE = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

# LLM equality-checker prompt (binary CORRECT/INCORRECT).
GRADER_TEMPLATE = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {official_answer}
CANDIDATE ANSWER TO ASSESS: {candidate_answer}

Reply only with CORRECT or INCORRECT.
"""


def build_prompt(documents: list[str], question: str) -> str:
    """Assemble the AA-LCR user prompt from ordered *documents* + *question*.

    Documents are wrapped and joined in the order given — the caller is
    responsible for supplying them in ``data_source_filenames`` order, which the
    dataset loader guarantees.
    """
    documents_text = "\n\n".join(
        DOCUMENT_TEMPLATE.format(number=i + 1, document=doc)
        for i, doc in enumerate(documents)
    )
    return PROMPT_TEMPLATE.format(documents_text=documents_text, question=question)


# Qualifiers that, placed directly on the verdict, mean the candidate is not
# consistent with the official answer: negation plus the hedges a checker reaches
# for when the answer is only a partial match. Enumerated rather than "any
# preceding word" so ordinary lead-ins ("VERDICT: CORRECT", "IS CORRECT") still
# read as a bare verdict.
_QUALIFIER = r"NOT|PARTIALLY|PARTLY|SEMI|MOSTLY|LARGELY|SOMEWHAT|NEARLY|ALMOST"
# Verdict phrase, qualifier-aware: matches CORRECT / INCORRECT as well as
# qualified forms (NOT CORRECT, PARTIALLY CORRECT, SEMI-CORRECT). Input is
# uppercased before matching; a qualified match is not the bare "CORRECT"
# string, so it falls through to INCORRECT.
_VERDICT_RE = re.compile(rf"\b(?:(?:{_QUALIFIER})[\s-]+)?(?:IN)?CORRECT\b")


def parse_grade(grading_response: str) -> str:
    """Map an equality-checker reply to ``CORRECT`` / ``INCORRECT``.

    The grader is instructed to reply with only ``CORRECT`` or ``INCORRECT``.
    The **last** verdict phrase wins — a model that reasons before answering
    puts the verdict at the end. Matching is word-bounded and qualifier-aware:

    * ``CORRECTNESS`` never matches the ``CORRECT`` token;
    * ``not correct`` counts as INCORRECT rather than falling through to a bare
      ``CORRECT``;
    * so does a hedged verdict (``partially correct``, ``semi-correct``) —
      AA-LCR's checker is binary and requires the candidate be *consistent
      with* the official answer, so a partial match is not CORRECT.

    Replies with no recognizable verdict — empty or malformed — are INCORRECT;
    AA-LCR has no not-attempted tier.
    """
    matches = _VERDICT_RE.findall(grading_response.upper())
    if matches and matches[-1] == "CORRECT":
        return "CORRECT"
    return "INCORRECT"


def aggregate_metrics(grades: list[str]) -> dict[str, float]:
    """Aggregate per-sample grades into AA-LCR accuracy (in ``[0, 1]``).

    Returns the correct/incorrect rates and the headline ``accuracy`` (the
    correct rate). Empty input yields all-zero metrics.
    """
    total = len(grades)
    if total == 0:
        return {"accuracy": 0.0, "is_correct": 0.0, "is_incorrect": 0.0}

    is_correct = sum(g == "CORRECT" for g in grades) / total
    return {
        "accuracy": is_correct,
        "is_correct": is_correct,
        "is_incorrect": 1.0 - is_correct,
    }
