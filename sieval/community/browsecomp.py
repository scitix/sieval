# query + grader prompts: verbatim from OpenAI simple-evals
# https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/browsecomp_eval.py
# (BrowseComp, Wei et al., arXiv:2504.12516). The grader prompt is adapted by
# OpenAI from the Humanity's-Last-Exam autorater.
"""BrowseComp autorater assets.

BrowseComp answers are short and verifiable, so grading is automatable via an
LLM autorater that decides a single ``correct: yes|no`` against the gold. This
module holds:

* ``QUERY_TEMPLATE`` — wraps the question so the model emits an
  ``Explanation / Exact Answer / Confidence`` block (applied in the task's
  ``preprocess``);
* ``GRADER_TEMPLATE`` — the yes/no autorater prompt (applied in ``feedback``);
* ``parse_grade`` — maps the grader reply to ``CORRECT``/``INCORRECT``
  (default INCORRECT, matching upstream's default-to-"no");
* ``parse_confidence`` — extracts the grader's echoed confidence (for post-hoc
  calibration; not part of the headline metric);
* ``aggregate_metrics`` — accuracy over all grades (BrowseComp's headline
  metric; there is no NOT_ATTEMPTED bucket — a non-answer is simply incorrect).

Prompts are byte-for-byte upstream. Local additions: the CORRECT/INCORRECT +
accuracy rollup, ``parse_confidence`` (upstream parses none), and the extraction
fix in ``parse_grade`` (see its docstring).

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import re

# `{Question}` is the only real format field; the example braces are doubled so
# `.format()` leaves them as literal single braces (upstream does the same).
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def parse_grade(grading_response: str) -> str:
    """Map a grader reply to ``CORRECT``/``INCORRECT``.

    Regex and default-``no`` are upstream's. Extraction deviates on purpose:
    upstream compares its whole match (``"correct: yes"``) to bare ``"yes"``, so
    its ``is_correct`` is never true (a latent bug at 652c89d0 / main); we read
    group 1 so ``correct: yes`` scores CORRECT, matching published numbers.
    """
    match = re.search(r"correct: (yes|no)", grading_response)
    verdict = match.group(1) if match else "no"
    return "CORRECT" if verdict == "yes" else "INCORRECT"


def parse_confidence(grading_response: str) -> int:
    """Extract the grader's echoed confidence (0-100); default 100.

    Not used in the headline accuracy — persisted for post-hoc calibration-error
    analysis (BrowseComp reports calibration alongside accuracy).
    """
    match = re.search(r"confidence: (\d+)", grading_response)
    try:
        return int(match.group(1)) if match else 100
    except (TypeError, ValueError):
        return 100


def aggregate_metrics(grades: list[str]) -> dict[str, float]:
    """Aggregate per-sample grades into BrowseComp's accuracy (all in ``[0, 1]``).

    Accuracy = fraction CORRECT over ALL grades (a non-answer / pipeline failure
    counts as INCORRECT, not a separate bucket). Empty input yields zeros.
    """
    total = len(grades)
    if total == 0:
        return {"accuracy": 0.0, "is_correct": 0.0, "is_incorrect": 0.0}
    is_correct = sum(g == "CORRECT" for g in grades) / total
    is_incorrect = sum(g == "INCORRECT" for g in grades) / total
    return {
        "accuracy": is_correct,
        "is_correct": is_correct,
        "is_incorrect": is_incorrect,
    }
