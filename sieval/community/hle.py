# Sources (Humanity's Last Exam, Center for AI Safety), pinned to commit
# 26dca2e253b405105b4c3d8c2f5af06f86f90c66:
#   - SYSTEM_PROMPT (verbatim) and the user-message assembly this task mirrors:
#     hle_eval/run_model_predictions.py (SYSTEM_PROMPT, format_message)
#   - JUDGE_PROMPT (verbatim), the ExtractedAnswer field set, and the accuracy /
#     confidence-interval / calibration-error formulas: hle_eval/run_judge_results.py
#     (JUDGE_PROMPT, ExtractedAnswer, dump_metrics)
#   - calib_err (verbatim): hle_eval/run_judge_results.py, itself vendored from
#     https://github.com/hendrycks/outlier-exposure/blob/404c6268865e84dbab5ab4ccf855e8ae39de853f/utils/calibration_tools.py
"""Humanity's Last Exam (HLE) prompt + grading assets.

HLE (Center for AI Safety) is a ~2,500-question, multi-domain, closed-ended
academic benchmark graded by an LLM judge. This module vendors the upstream
model system prompt, the judge prompt, and the metric kernel (accuracy, 95%
Wald confidence interval, and calibration error).

Deviations from upstream (``hle_eval/run_judge_results.py`` @ 26dca2e):

* Judge invocation. Upstream uses ``beta.chat.completions.parse`` with the
  ``ExtractedAnswer`` schema; sieval goes through the generic ``ChatModel``, so
  ``parse_judge`` reads ``correct``/``confidence`` textually — the judge runs on
  an arbitrary ``api_base`` and not every endpoint enforces a JSON schema.
  Correctness follows upstream semantics (``"yes" in ...["correct"]``). An
  unparseable reply returns ``parsed=False`` and the caller drops it from the
  grading arrays, mirroring upstream's ``None`` on judge failure: still in
  ``n`` (counted incorrect), out of the calibration arrays.
* ``aggregate_metrics`` returns a dict instead of printing, and guards
  ``len(confidence) < beta`` (the vendored ``calib_err`` indexes ``bins[-1]``
  and raises below that), reporting ``calibration_error`` as ``None`` —
  distinct from a real 0.0. Numerator and calibration arrays come from judged
  attempts only; the caller supplies ``n`` (finals + failures) as the
  denominator, matching upstream where failed predictions stay in ``n`` but out
  of ``correct``.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import math
import re

import numpy as np

SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

JUDGE_PROMPT = r"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""

# Target bin size for calibration error (upstream default).
BETA = 100

# `\b` anchors the field name so it never matches inside "incorrect:"; the
# bracketed runs tolerate markdown bold (`*`), quotes (JSON-shaped replies) and
# whitespace around the field name, the colon and the value.
_CORRECT_RE = re.compile(r"\bcorrect[\s\"'*]*:[\s\"'*]*(yes|no)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"\bconfidence[\s\"'*]*:[\s\"'*]*(\d+)", re.IGNORECASE)


def parse_judge(reply: str) -> tuple[bool, int, bool]:
    """Extract ``(correct, confidence, parsed)`` from a judge reply.

    ``JUDGE_PROMPT`` puts ``correct: yes|no`` and ``confidence: <int>`` after
    the free-form ``reasoning``, so the last match of each wins. ``parsed`` is
    ``True`` only when ``correct`` was found; otherwise the caller drops the
    record rather than materializing a spurious verdict. ``confidence`` falls
    back to 100 only when ``correct`` parsed without one — the prompt's own
    "Put 100 if there is no confidence score" case, which is about the
    candidate's score, not judge failure.
    """
    correct_matches = _CORRECT_RE.findall(reply)
    parsed = bool(correct_matches)
    correct = parsed and correct_matches[-1].lower() == "yes"

    confidence_matches = _CONFIDENCE_RE.findall(reply)
    confidence = int(confidence_matches[-1]) if confidence_matches else 100
    return correct, confidence, parsed


# source: https://github.com/hendrycks/outlier-exposure/blob/404c6268865e84dbab5ab4ccf855e8ae39de853f/utils/calibration_tools.py
def calib_err(confidence, correct, p="2", beta=100):
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0] : bins[i][1]]
        bin_correct = correct[bins[i][0] : bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p == "infty" or p == "infinity" or p == "max":
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == "2":
        cerr = np.sqrt(cerr)

    return cerr


def aggregate_metrics(
    correct: list[bool], confidence: list[int], n: int
) -> dict[str, float | None]:
    """Aggregate judged ``correct``/``confidence`` into HLE metrics.

    Mirrors upstream ``dump_metrics``: ``correct``/``confidence`` are the judged
    attempts (numerator + calibration arrays), while ``n`` is the full requested
    count (denominator) so failed/ungraded attempts count as incorrect.

    Returns accuracy (%), the 95% Wald half-width (``confidence_interval``, in
    percentage points), and ``calibration_error`` (0..100), which is ``None``
    when it cannot be computed — ``n == 0`` or fewer than ``BETA`` graded
    attempts — so "not computed" is not conflated with a real 0.0.

    One band escapes that sentinel: at ``BETA``..``2 * BETA`` graded attempts
    ``calib_err``'s ``range(len(bins) - 1)`` loop never runs and it returns a
    structural ``0.0``. That is upstream's dropped-last-bin behaviour, kept
    as-is and unreachable on the full ~2.5k set; read it as uninformative.
    """
    if n == 0:
        return {
            "accuracy": 0.0,
            "confidence_interval": 0.0,
            "calibration_error": None,
        }

    accuracy = round(100 * sum(correct) / n, 2)
    # Wald estimator, 95% confidence interval
    confidence_interval = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / n), 2)

    calibration_error: float | None
    if len(confidence) >= BETA:
        confidence_arr = np.array(confidence) / 100
        correct_arr = np.array(correct)
        calibration_error = 100 * round(
            float(calib_err(confidence_arr, correct_arr, p="2", beta=BETA)), 2
        )
    else:
        calibration_error = None

    return {
        "accuracy": accuracy,
        "confidence_interval": confidence_interval,
        "calibration_error": calibration_error,
    }
