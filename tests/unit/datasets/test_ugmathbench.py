"""Unit tests for the UGMathBench loader: version unpacking and subject selection.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset

from sieval.datasets import ugmathbench as module
from sieval.datasets.ugmathbench import UGMathBenchDataset, _unpack_versions


def _packed_row(problem_id: str = "Algebra_0001") -> dict:
    return {
        "id": problem_id,
        "subject": "Algebra",
        "topic": "Linear equations",
        "subtopic": "Solving",
        "level": "2",
        "keywords": ["algebra"],
        "problem_v1": "v1 text",
        "answer_v1": ["1"],
        "answer_type_v1": ["NV"],
        "options_v1": [[]],
        "problem_v2": "v2 text",
        "answer_v2": ["2"],
        "answer_type_v2": ["NV"],
        "options_v2": [[]],
        "problem_v3": "v3 text",
        "answer_v3": ["A"],
        "answer_type_v3": ["MCS"],
        "options_v3": [["A", "B"]],
    }


def test_unpacks_one_sample_per_randomized_version():
    samples = _unpack_versions(_packed_row())

    assert [s["version"] for s in samples] == [1, 2, 3]
    assert [s["problem"] for s in samples] == ["v1 text", "v2 text", "v3 text"]
    assert [s["answer"] for s in samples] == [["1"], ["2"], ["A"]]
    assert samples[2]["answer_type"] == ["MCS"]
    assert samples[2]["options"] == [["A", "B"]]
    # Shared metadata rides along on every version.
    assert {s["id"] for s in samples} == {"Algebra_0001"}
    assert {s["subject"] for s in samples} == {"Algebra"}


def test_load_is_problem_major_so_slicing_keeps_whole_problems(monkeypatch):
    rows = [_packed_row("Algebra_0001"), _packed_row("Algebra_0002")]
    monkeypatch.setattr(
        module, "load_dataset", lambda *a, **kw: HFDataset.from_list(rows)
    )

    dataset = UGMathBenchDataset("ignored", subjects=["Algebra"])
    test_set = dataset.test_set

    assert test_set is not None
    assert len(test_set) == 6
    # A problem's three versions are adjacent, so slice(3) keeps one whole
    # problem rather than one version of three different problems.
    assert [(r["id"], r["version"]) for r in test_set][:4] == [
        ("Algebra_0001", 1),
        ("Algebra_0001", 2),
        ("Algebra_0001", 3),
        ("Algebra_0002", 1),
    ]


def test_unknown_subject_is_rejected_with_the_valid_list(monkeypatch):
    monkeypatch.setattr(
        module, "load_dataset", lambda *a, **kw: HFDataset.from_list([_packed_row()])
    )
    with pytest.raises(ValueError, match="Unknown UGMathBench subject"):
        UGMathBenchDataset("ignored", subjects=["Astrology"])


def test_explicitly_empty_subjects_is_rejected_not_read_as_all(monkeypatch):
    # `[]` is falsy, so a truthiness check would quietly load all 16 instead.
    loaded: list[str] = []
    monkeypatch.setattr(
        module,
        "load_dataset",
        lambda _p, config, **_kw: (
            loaded.append(config) or HFDataset.from_list([_packed_row()])
        ),
    )
    with pytest.raises(ValueError, match="`subjects` is empty"):
        UGMathBenchDataset("ignored", subjects=[])
    assert loaded == []


def test_all_sixteen_subjects_load_by_default(monkeypatch):
    requested: list[str] = []

    def _fake_load(_path, config, **_kwargs):
        requested.append(config)
        return HFDataset.from_list([_packed_row()])

    monkeypatch.setattr(module, "load_dataset", _fake_load)

    UGMathBenchDataset("ignored")
    assert requested == list(UGMathBenchDataset.SUBJECTS)
    assert len(requested) == 16
