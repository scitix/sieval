"""Unit tests for the Inverse IFEval dataset loader.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.datasets.inverse_ifeval import (
    INVERSE_IFEVAL_DATA_FILE,
    InverseIFEvalDataset,
)

_TEMPLATE = "<标准答案>：\n{response_reference}\n\n<学生答案>：\n{response}"


def _row(language: str = "english", instruction_type: str = "Question Correction"):
    return {
        "instruction_types": instruction_type,
        "prompt": "Answer without any reasoning.",
        "response_reference": "The answer must omit all reasoning.",
        "language": language,
        "judge_prompt_template": _TEMPLATE,
        "judge_system_prompt": "你是判卷老师。",
    }


def _patch_load(monkeypatch: pytest.MonkeyPatch, rows: list[dict], captured=None):
    def fake_load_dataset(name_or_path: str, **kwargs):
        if captured is not None:
            captured["name"] = name_or_path
            captured.update(kwargs)
        return HFDatasetDict({"train": HFDataset.from_list(rows)})

    monkeypatch.setattr(
        "sieval.datasets.inverse_ifeval.load_dataset", fake_load_dataset
    )


def test_load_mirrors_the_single_split_to_test(monkeypatch: pytest.MonkeyPatch):
    _patch_load(monkeypatch, [_row()])

    dataset = InverseIFEvalDataset("m-a-p/Inverse_IFEval")

    assert set(dataset.dataset_dict) == {"train", "test"}
    assert dataset.test_set is not None
    assert len(dataset.test_set) == 1
    assert dataset.language is None


def test_load_pins_the_json_data_file(monkeypatch: pytest.MonkeyPatch):
    # The repo ships a second, differently-dated CSV snapshot. Auto-detection
    # happens to resolve to the JSON today, so this pins the choice rather than
    # leaving which snapshot gets evaluated to extension precedence.
    captured: dict[str, object] = {}
    _patch_load(monkeypatch, [_row()], captured)

    InverseIFEvalDataset("m-a-p/Inverse_IFEval")

    assert captured["data_files"] == INVERSE_IFEVAL_DATA_FILE
    assert "split" not in captured


def test_load_forwards_kwargs_without_leaking_language(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}
    _patch_load(monkeypatch, [_row()], captured)

    InverseIFEvalDataset(
        "m-a-p/Inverse_IFEval", language="english", trust_remote_code=False
    )

    assert "language" not in captured
    assert captured["trust_remote_code"] is False


# --- language subset ---


def test_language_filter_keeps_one_subset_and_records_it(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_load(monkeypatch, [_row("english"), _row("chinese"), _row("chinese")])

    dataset = InverseIFEvalDataset("m-a-p/Inverse_IFEval", language="chinese")

    assert dataset.language == "chinese"
    assert dataset.test_set is not None
    assert len(dataset.test_set) == 2
    assert {row["language"] for row in dataset.test_set} == {"chinese"}


def test_language_none_keeps_both(monkeypatch: pytest.MonkeyPatch):
    _patch_load(monkeypatch, [_row("english"), _row("chinese")])

    dataset = InverseIFEvalDataset("m-a-p/Inverse_IFEval")

    assert dataset.test_set is not None
    assert len(dataset.test_set) == 2


def test_unknown_language_raises_instead_of_filtering_to_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    # A typo would otherwise produce a clean 0.0 over no samples.
    _patch_load(monkeypatch, [_row("english")])

    with pytest.raises(ValueError, match="Unknown Inverse IFEval language"):
        InverseIFEvalDataset("m-a-p/Inverse_IFEval", language="Chinese")


def test_language_selecting_nothing_raises(monkeypatch: pytest.MonkeyPatch):
    # A declared language that the source no longer uses: schema drift, not an
    # empty run.
    _patch_load(monkeypatch, [_row("english")])

    with pytest.raises(ValueError, match="selected no samples"):
        InverseIFEvalDataset("m-a-p/Inverse_IFEval", language="chinese")


# --- required columns ---


@pytest.mark.parametrize(
    "missing",
    [
        "instruction_types",
        "prompt",
        "response_reference",
        "language",
        "judge_prompt_template",
        "judge_system_prompt",
    ],
)
def test_missing_column_raises_at_load_time(
    monkeypatch: pytest.MonkeyPatch, missing: str
):
    # The judge prompt is per-sample DATA here, so a dropped column must fail
    # before the candidate is paid for, not as a KeyError mid-run.
    row = _row()
    del row[missing]
    _patch_load(monkeypatch, [row])

    with pytest.raises(ValueError, match=f"missing required column\\(s\\): {missing}"):
        InverseIFEvalDataset("m-a-p/Inverse_IFEval")
