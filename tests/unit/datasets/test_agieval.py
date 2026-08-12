"""Unit tests for the AGIEval dataset loader.

Covers the two things this loader adds over "read some jsonl": subset selection
(the reason the loader exists in this shape) and the schema normalizations that
let 21 files with disagreeing per-file schemas concatenate at all.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
from pathlib import Path

import pytest

from sieval.datasets.agieval import SUBSET_GROUPS, AGIEvalDataset

# Minimal rows in each source shape, keyed by subset. Field presence and dtypes
# mirror the pinned v1.1 files: sat-* have no `answer` key at all, cloze rows
# null out `options`/`label`, jec-qa wraps the label in a list, and only
# math.jsonl carries `level` (as an int).
_ROWS: dict[str, dict] = {
    "sat-math": {
        "passage": "",
        "question": "sm?",
        "options": ["(A)1", "(B)2", "(C)3", "(D)4"],
        "label": "D",
        "other": {"solution": "because"},
    },
    "aqua-rat": {
        "passage": None,
        "question": "ar?",
        "options": ["(A)1", "(B)2", "(C)3", "(D)4", "(E)5"],
        "label": "B",
        "answer": None,
        "other": {"solution": "because"},
    },
    "gaokao-mathqa": {
        "passage": None,
        "question": "gm?",
        "options": ["(A)1", "(B)2", "(C)3", "(D)4"],
        "label": "A",
        "answer": None,
        "other": {"source": "2021年浙江卷—数学"},
    },
    "math": {
        "passage": None,
        "question": "m?",
        "options": None,
        "label": None,
        "answer": "(3,4]",
        "other": {"solution": "s", "level": 5, "type": "Intermediate Algebra"},
    },
    "gaokao-mathcloze": {
        "passage": None,
        "question": "gc?",
        "options": None,
        "label": None,
        "answer": "2",
        "other": {"source": "2021年浙江卷—数学"},
    },
    "jec-qa-kd": {
        "passage": None,
        "question": "j?",
        "options": ["(A)1", "(B)2", "(C)3", "(D)4"],
        "label": ["B"],
        "answer": None,
        "other": None,
    },
}


def _stage(
    tmp_path: Path,
    *subsets: str,
    label_override=None,
    row_overrides: dict | None = None,
) -> str:
    """Write one .jsonl per subset, the way `sieval dataset download` stages them."""
    for subset in subsets:
        row = dict(_ROWS[subset])
        if label_override is not None:
            row["label"] = label_override
        if row_overrides:
            row.update(row_overrides)
        path = tmp_path / f"{subset}.jsonl"
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(tmp_path)


def test_group_math_loads_the_five_math_subsets(tmp_path):
    path = _stage(tmp_path, *SUBSET_GROUPS["math"])
    test = AGIEvalDataset(path, group="math").dataset_dict["test"]

    assert len(test) == 5
    assert set(test["subset"]) == set(SUBSET_GROUPS["math"])


def test_explicit_subsets_load_in_canonical_order(tmp_path):
    path = _stage(tmp_path, "math", "sat-math")
    # Argument order must not leak into row order, or the same selection spelled
    # two ways would produce different sample ids.
    forward = AGIEvalDataset(path, subsets=["sat-math", "math"]).dataset_dict["test"]
    reverse = AGIEvalDataset(path, subsets=["math", "sat-math"]).dataset_dict["test"]

    assert forward["subset"] == reverse["subset"] == ["sat-math", "math"]


def test_row_keeps_upstream_fields_and_gains_subset(tmp_path):
    path = _stage(tmp_path, "sat-math")
    row = AGIEvalDataset(path, subsets=["sat-math"]).dataset_dict["test"][0]

    assert row["subset"] == "sat-math"
    assert row["question"] == "sm?"
    assert row["label"] == "D"
    # `answer` is absent from the sat-* source rows entirely -> None, not "".
    assert row["answer"] is None
    assert row["other"]["solution"] == "because"
    assert row["other"]["level"] is None


def test_cloze_rows_null_out_label_and_empty_options(tmp_path):
    path = _stage(tmp_path, "math")
    row = AGIEvalDataset(path, subsets=["math"]).dataset_dict["test"][0]

    assert row["label"] is None
    assert row["answer"] == "(3,4]"
    assert row["options"] == []
    # int64 upstream, stringified so the struct has one dtype across subsets.
    assert row["other"]["level"] == "5"


def test_jec_qa_list_label_is_unwrapped(tmp_path):
    path = _stage(tmp_path, "jec-qa-kd")
    row = AGIEvalDataset(path, subsets=["jec-qa-kd"]).dataset_dict["test"][0]

    assert row["label"] == "B"


def test_multi_label_row_raises_instead_of_silently_rescoring(tmp_path):
    # v1.0 had genuine multi-label jec-qa rows; if the pinned data ever grows one
    # back, set-comparison semantics change and must be revisited deliberately.
    path = _stage(tmp_path, "jec-qa-kd", label_override=["A", "B"])
    with pytest.raises(ValueError, match="single-answer label"):
        AGIEvalDataset(path, subsets=["jec-qa-kd"])


def test_mcq_row_without_options_raises_instead_of_dropping_the_choices(tmp_path):
    # Only the two cloze subsets may omit `options`. Coercing a null to [] on an
    # MCQ subset would prompt the question with no answer choices and score
    # whatever came back, so the shape change has to surface as an error rather
    # than as a low number.
    path = _stage(tmp_path, "sat-math", row_overrides={"options": None})
    with pytest.raises(ValueError, match="only the cloze subsets"):
        AGIEvalDataset(path, subsets=["sat-math"])


def test_subsets_and_group_together_is_rejected(tmp_path):
    path = _stage(tmp_path, "math")
    with pytest.raises(ValueError, match="not both"):
        AGIEvalDataset(path, subsets=["math"], group="math")


def test_unknown_selection_is_rejected(tmp_path):
    path = _stage(tmp_path, "math")
    with pytest.raises(ValueError, match="unknown subset"):
        AGIEvalDataset(path, subsets=["mmlu"])
    with pytest.raises(ValueError, match="unknown group"):
        AGIEvalDataset(path, group="en")
    with pytest.raises(ValueError, match="`subsets` is empty"):
        AGIEvalDataset(path, subsets=[])


def test_missing_subset_file_names_the_path_and_the_fix(tmp_path):
    path = _stage(tmp_path, "math")
    with pytest.raises(FileNotFoundError, match="sieval dataset download agieval"):
        AGIEvalDataset(path, group="math")
