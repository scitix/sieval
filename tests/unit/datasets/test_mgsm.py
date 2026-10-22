"""
Unit tests for the MGSM dataset loader.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from unittest.mock import patch

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.datasets.mgsm as mgsm_module
from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.mgsm import MGSM_LANGUAGES, MGSM_REVISION, MGSMDataset


def _hf_dict() -> HFDatasetDict:
    return HFDatasetDict(
        {
            "train": HFDataset.from_list(
                [
                    {
                        "question": "Question: q",
                        "answer": "Step-by-Step Answer: a. The answer is 11.",
                        "answer_number": 11,
                        "equation_solution": "5 + 6 = 11.",
                    }
                ]
            ),
            "test": HFDataset.from_list(
                [
                    {
                        "question": "q",
                        "answer": None,
                        "answer_number": 18,
                        "equation_solution": None,
                    }
                ]
            ),
        }
    )


def test_source_revision_is_pinned():
    assert MGSM_REVISION == "b2f13d426afe3be8d69a7e739b36724db8b66bbc"
    assert get_dataset_meta(MGSMDataset).source == (
        f"hf:juletxara/mgsm@{MGSM_REVISION}",
    )


def test_load_reads_one_language_config_as_is():
    hf_dict = _hf_dict()
    with patch.object(mgsm_module, "load_dataset", return_value=hf_dict) as mock:
        loaded = MGSMDataset("/staged/juletxara/mgsm", lang="zh").dataset_dict

    assert mock.call_args.args == ("/staged/juletxara/mgsm", "zh")
    # Both splits kept: `train` is the same language's exemplar source.
    assert set(loaded) == {"train", "test"}
    assert loaded["test"][0] == {
        "question": "q",
        "answer": None,
        "answer_number": 18,
        "equation_solution": None,
    }


@pytest.mark.parametrize("lang", ["xx", "ca", "ZH"])
def test_unknown_language_is_refused_before_loading(lang):
    with (
        patch.object(mgsm_module, "load_dataset") as mock,
        pytest.raises(ValueError, match="Unknown MGSM language"),
    ):
        MGSMDataset("/staged/juletxara/mgsm", lang=lang)
    mock.assert_not_called()


def test_language_is_required():
    with (
        patch.object(mgsm_module, "load_dataset") as mock,
        pytest.raises(ValueError, match="MGSM needs `lang`"),
    ):
        MGSMDataset("/staged/juletxara/mgsm")
    mock.assert_not_called()


def test_languages_are_the_11_upstream_configs():
    assert len(MGSM_LANGUAGES) == 11
