# adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/1dd931087362abba74e0375c8c631295559f48b2/lm_eval/tasks/hellaswag/utils.py

import re
from collections.abc import Mapping
from typing import Any, TypedDict


class ProcessedDoc(TypedDict):
    query: str
    choices: list[str]
    gold: int


def preprocess(text: str) -> str:
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_doc(doc: Mapping[str, Any]) -> ProcessedDoc:
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return {
        "query": preprocess(doc["activity_label"] + ": " + ctx),
        "choices": [preprocess(ending) for ending in doc["endings"]],
        "gold": int(doc["label"]),
    }
