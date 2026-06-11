import json
from unittest.mock import patch

import pytest
from datasets import Dataset as HFDataset

from sieval.datasets.ruler.ruler_qa import (
    RulerQaDataset,
    _read_hotpotqa,
    _read_squad,
)


@pytest.fixture
def squad_dir(tmp_path):
    """A tiny SQuAD v2.0-shaped file, including one impossible question."""
    data = {
        "data": [
            {
                "paragraphs": [
                    {
                        "context": f"Context number {i} about topic {i}.",
                        "qas": [
                            {
                                "question": f"What is topic {i}?",
                                "answers": [{"text": f"topic {i}"}],
                                "is_impossible": False,
                            }
                        ],
                    }
                    for i in range(30)
                ]
                + [
                    {
                        "context": "An unanswerable paragraph.",
                        "qas": [
                            {
                                "question": "Unanswerable?",
                                "answers": [],
                                "is_impossible": True,
                            }
                        ],
                    }
                ]
            }
        ]
    }
    (tmp_path / "dev-v2.0.json").write_text(json.dumps(data), encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def hotpot_hf_dataset():
    """HF-schema hotpotqa fixture: context={'title':[...], 'sentences':[[...]]}."""
    rows = [
        {
            "question": f"Who did thing {i}?",
            "answer": f"person {i}",
            "context": {
                "title": [f"Title {i}", f"Other {i}"],
                "sentences": [
                    [f"person {i} did thing {i}.", " More text."],
                    [f"distractor {i}."],
                ],
            },
        }
        for i in range(30)
    ]
    return HFDataset.from_list(rows)


def test_read_squad_filters_impossible(squad_dir):
    qas, docs = _read_squad(f"{squad_dir}/dev-v2.0.json")
    # The is_impossible question must be dropped.
    assert len(qas) == 30
    assert all("topic" in qa["query"] for qa in qas)
    assert qas[0]["outputs"] == ["topic 0"]
    # Docs deduped + sorted; the unanswerable context still lands in the pool.
    assert "An unanswerable paragraph." in docs


def test_read_hotpotqa_shape(hotpot_hf_dataset):
    with patch(
        "sieval.datasets.ruler.ruler_qa.load_dataset", return_value=hotpot_hf_dataset
    ):
        qas, docs = _read_hotpotqa("hotpotqa/hotpot_qa")
    assert len(qas) == 30
    assert qas[0]["outputs"] == ["person 0"]
    # Two context docs per question.
    assert len(qas[0]["context"]) == 2


def test_squad_synthesis_row_schema(squad_dir):
    ds = RulerQaDataset(squad_dir, dataset="squad", max_seq_length=512, num_samples=2)
    test = ds.test_set
    assert test is not None and len(test) == 2
    row = test[0]
    # Schema produced by the RULER QA loader.
    assert set(row) == {"index", "input", "outputs", "length", "answer_prefix"}
    # `answer_prefix` is split off the prompt tail; `input` no longer ends in it.
    assert row["answer_prefix"] == " Answer:"
    assert not row["input"].endswith("Answer:")
    # Distractor documents are assembled into the prompt.
    assert "Document 1:" in row["input"]
    assert "Question:" in row["input"]
    # The gold answer's source document is in the assembled context.
    assert any(a in row["input"] for a in row["outputs"])


def test_remove_newline_tab_single_line(squad_dir):
    ds = RulerQaDataset(squad_dir, dataset="squad", max_seq_length=512, num_samples=1)
    # Default remove_newline_tab=True collapses the prompt to one line.
    assert "\n" not in ds.test_set[0]["input"]


def test_hotpotqa_synthesis(hotpot_hf_dataset):
    with patch(
        "sieval.datasets.ruler.ruler_qa.load_dataset", return_value=hotpot_hf_dataset
    ):
        ds = RulerQaDataset(
            "hotpotqa/hotpot_qa", dataset="hotpotqa", max_seq_length=512, num_samples=2
        )
    row = ds.test_set[0]
    assert "Document 1:" in row["input"]
    assert any(a in row["input"] for a in row["outputs"])


def test_deterministic_under_seed(squad_dir):
    kw = {"dataset": "squad", "max_seq_length": 512, "num_samples": 2, "random_seed": 7}
    a = RulerQaDataset(squad_dir, **kw).test_set[0]
    b = RulerQaDataset(squad_dir, **kw).test_set[0]
    assert a["input"] == b["input"]
    assert a["outputs"] == b["outputs"]


def test_unknown_dataset_rejected(squad_dir):
    with pytest.raises(NotImplementedError):
        RulerQaDataset(squad_dir, dataset="triviaqa")
