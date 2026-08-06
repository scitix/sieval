# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# adapted from https://github.com/ruixiangcui/AGIEval/blob/84ab72d94318290aad2e4ec820d535a95a1f7552/src/post_process.py
"""AGIEval answer extraction from the second-stage model reply.

Three parsers, picked by subset family (upstream ``post_process``):

* MCQ, single answer -> ``find_first_capital_letter``: the first A-F character in
  the reply. Only sane on the *second-stage* reply (a bare "A" / " D."), which is
  exactly why upstream's zero-shot protocol has a second stage — run against a
  free-form chain of thought it would happily return the "A" of an option label.
* MCQ, multi answer (``MULTI_CHOICE_SUBSETS``) -> ``parse_qa_multiple_answer``:
  every A-F character, compared as a set.
* cloze (``MATH_OUTPUT_SUBSETS``) -> ``parse_math_answer``: last ``\\boxed{}``,
  else last ``$...$``, else a trailing ``=``-expression or bare number.

Deltas from upstream, all deliberate:

* Zero-shot only: the ``setting_name`` parameter is gone, along with the
  ``few-shot-CoT`` ``extract_last_line`` branches it gated and the few-shot
  format-compliance helpers. ``remove_few_shot_prefix`` stays — upstream calls it
  unconditionally inside ``parse_math_answer``, in every setting.
* A failed extraction returns ``None`` rather than ``""`` / ``[]``, which is
  sieval's "could not extract" contract (``PredictionRecord``). Verdicts are
  unchanged: neither ``""`` nor ``[]`` can equal a gold answer.
* Regexes are raw strings. The patterns are byte-identical — upstream's
  ``"\\$(.*)\\$"`` and ``"(?:\\\\$)?\\d+..."`` are plain strings whose invalid
  escapes Python leaves untouched.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import re

from .dataset_loader import (
    CHINESE_CLOZE_SUBSETS,
    CHINESE_QA_SUBSETS,
    ENGLISH_CLOZE_SUBSETS,
    ENGLISH_QA_SUBSETS,
    MULTI_CHOICE_SUBSETS,
    SUBSETS,
)

_FEW_SHOT_PREFIXES = ("The answer is therefore", "答案是", "The answer is")
_CAPITAL_LETTERS = {"A", "B", "C", "D", "E", "F"}


def remove_few_shot_prefix(string: str) -> str:
    for prefix in _FEW_SHOT_PREFIXES:
        if string.startswith(prefix):
            string = string[len(prefix) :].strip()
        elif prefix in string:
            index = string.rfind(prefix)
            if index >= 0:
                string = string[index + len(prefix) :].strip()
    return string


def find_first_capital_letter(answer: str) -> str:
    for c in answer:
        if c in _CAPITAL_LETTERS:
            return c
    return ""


def parse_qa_multiple_answer(string: str) -> list[str]:
    return re.findall(r"\(*([A-F])\)*", string)


def parse_math_answer(raw_string: str) -> str | None:
    def remove_boxed(s):
        left = "\\boxed{"
        try:
            assert s[: len(left)] == left
            assert s[-1] == "}"
            answer = s[len(left) : -1]
            if "=" in answer:
                answer = answer.split("=")[-1].lstrip(" ")
            return answer
        except Exception:
            return None

    def last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]

        return retval

    def get_answer_with_dollar_sign(s):
        first_pattern = r"\$(.*)\$"
        last_match = None
        matches = re.findall(first_pattern, s)
        if matches:
            last_match = matches[-1]
            if "=" in last_match:
                last_match = last_match.split("=")[-1].lstrip(" ")
        return last_match

    def get_answer_without_dollar_sign(s):
        last_match = None
        if "=" in s:
            last_match = s.split("=")[-1].lstrip(" ").rstrip(".")
            if "\\" in last_match:
                last_match = last_match.split("\\")[0]
        else:
            pattern = r"(?:\$)?\d+(?:\.\d+)?(?![\w\d])"
            matches = re.findall(pattern, s)
            if matches:
                last_match = matches[-1]
        return last_match

    raw_string = remove_few_shot_prefix(raw_string)
    if "\\boxed" in raw_string:
        answer = remove_boxed(last_boxed_only_string(raw_string))
    else:
        answer = get_answer_with_dollar_sign(raw_string)
        if not answer:
            answer = get_answer_without_dollar_sign(raw_string)
    return answer


def post_process(subset: str, prediction: str) -> str | list | None:
    """Extract the answer from a second-stage reply. ``None`` = nothing found."""
    if subset in ENGLISH_CLOZE_SUBSETS or subset in CHINESE_CLOZE_SUBSETS:
        return parse_math_answer(prediction) or None

    if subset in MULTI_CHOICE_SUBSETS:
        return parse_qa_multiple_answer(prediction) or None

    if subset in ENGLISH_QA_SUBSETS or subset in CHINESE_QA_SUBSETS:
        return find_first_capital_letter(prediction) or None

    raise ValueError(f"Unknown AGIEval subset {subset!r}; expected one of {SUBSETS}")
