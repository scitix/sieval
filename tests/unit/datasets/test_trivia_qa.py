"""
Unit tests for the TriviaQA dataset loader.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from unittest.mock import patch

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.datasets.trivia_qa as trivia_qa_module
from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.downloaders.hf import parse_hf_source
from sieval.datasets.trivia_qa import TRIVIA_QA_REVISION, TriviaQADataset


def _row(qid: str, value: str) -> dict:
    return {
        "question": f"q {qid}",
        "question_id": qid,
        "question_source": "http://example.com/",
        "entity_pages": {"title": [], "wiki_context": []},
        "search_results": {"title": [], "search_context": []},
        "answer": {
            "aliases": [] if value == "<unk>" else [value],
            "normalized_aliases": [],
            "matched_wiki_entity_name": "",
            "normalized_matched_wiki_entity_name": "",
            "normalized_value": value.lower(),
            "type": "WikipediaEntity",
            "value": value,
        },
    }


def _hf_dict() -> HFDatasetDict:
    return HFDatasetDict(
        {
            "train": HFDataset.from_list([_row("tr_1", "Paris")]),
            "validation": HFDataset.from_list([_row("va_1", "David Seville")]),
            "test": HFDataset.from_list([_row("te_1", "<unk>")]),
        }
    )


def test_source_is_pinned_and_narrowed_to_the_nocontext_config():
    assert TRIVIA_QA_REVISION == "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"
    (source,) = get_dataset_meta(TriviaQADataset).source
    parsed = parse_hf_source(source)
    assert parsed.repo_id == "mandarjoshi/trivia_qa"
    assert parsed.revision == TRIVIA_QA_REVISION
    assert parsed.allow_pattern == "rc.nocontext/*"


def test_load_reads_the_config_by_data_dir():
    with patch.object(
        trivia_qa_module, "load_dataset", return_value=_hf_dict()
    ) as mock:
        TriviaQADataset("/staged/mandarjoshi/trivia_qa")
    # By `data_dir`, not config name: the card's other configs are not staged.
    assert mock.call_args.args == ("/staged/mandarjoshi/trivia_qa",)
    assert mock.call_args.kwargs == {"data_dir": "rc.nocontext"}


def test_load_defaults_eval_split_to_labelled_validation():
    with patch.object(trivia_qa_module, "load_dataset", return_value=_hf_dict()):
        loaded = TriviaQADataset("/staged/mandarjoshi/trivia_qa").dataset_dict

    assert set(loaded) == {"train", "validation", "test"}
    assert loaded["test"]["question_id"] == ["va_1"]
    assert loaded["train"]["question_id"] == ["tr_1"]
    assert loaded["test"][0]["answer"]["aliases"] == ["David Seville"]
    # The evidence columns, always empty in this config, are dropped.
    for rows in loaded.values():
        assert rows.column_names == [
            "question",
            "question_id",
            "question_source",
            "answer",
        ]


def test_load_explicit_eval_split_test_keeps_unlabelled_test():
    with patch.object(trivia_qa_module, "load_dataset", return_value=_hf_dict()):
        loaded = TriviaQADataset(
            "/staged/mandarjoshi/trivia_qa", eval_split="test"
        ).dataset_dict

    assert loaded["test"]["question_id"] == ["te_1"]
