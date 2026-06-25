"""
Unit tests for the CMMLU dataset loader.

AI-Generated Code - GPT-5.5 (OpenAI)
"""

import zipfile

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.datasets.cmmlu import CMMLU_REVISION, CMMLU_SOURCE_URL, CMMLUDataset


def test_processes_structured_cmmlu_row():
    dataset = CMMLUDataset(_hf_dict=HFDatasetDict({"test": HFDataset.from_list([])}))

    sample = dataset._process_sample(
        {
            "Question": "问题",
            "A": "甲",
            "B": "乙",
            "C": "丙",
            "D": "丁",
            "Answer": "c",
            "Subject": "logical",
        }
    )

    assert sample["question"] == "问题"
    assert sample["answer"] == "C"
    assert sample["subject"] == "logical"


def test_malformed_row_raises():
    dataset = CMMLUDataset(_hf_dict=HFDatasetDict({"test": HFDataset.from_list([])}))

    with pytest.raises(ValueError, match="Malformed CMMLU row"):
        dataset._process_sample({"unexpected": "no question column"}, "anatomy")


def test_empty_load_raises(tmp_path):
    zip_path = tmp_path / "cmmlu.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "CMMLU-sha/data/test/anatomy.csv", ",Question,A,B,C,D,Answer\n"
        )

    with pytest.raises(ValueError, match="No CMMLU samples were loaded"):
        CMMLUDataset(str(zip_path), subjects=["nonexistent_subject"])


def test_loads_from_staged_directory(tmp_path):
    # Production path: `sieval dataset download` saves `<sha>.zip` into the
    # dataset dir, and load() receives that directory.
    data_dir = tmp_path / "cmmlu"
    data_dir.mkdir()
    with zipfile.ZipFile(data_dir / f"{CMMLU_REVISION}.zip", "w") as archive:
        archive.writestr(
            f"CMMLU-{CMMLU_REVISION}/data/test/anatomy.csv",
            ",Question,A,B,C,D,Answer\n0,女性生殖腺是,卵巢,前庭大腺,前庭球,乳腺,A\n",
        )

    dataset = CMMLUDataset(str(data_dir), subjects=["anatomy"])

    assert dataset.dataset_dict["test"][0]["question"] == "女性生殖腺是"
    assert dataset.dataset_dict["test"][0]["answer"] == "A"


def test_missing_archive_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="sieval dataset download cmmlu"):
        CMMLUDataset(str(tmp_path), subjects=["anatomy"])


def test_dataset_source_uses_official_github_archive():
    expected_url = f"https://github.com/haonan-li/CMMLU/archive/{CMMLU_REVISION}.zip"

    assert expected_url == CMMLU_SOURCE_URL


def test_loads_official_github_archive_layout(tmp_path):
    zip_path = tmp_path / "cmmlu.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "CMMLU-sha/data/dev/anatomy.csv",
            ",Question,A,B,C,D,Answer\n"
            "0,壁胸膜的分部不包括,肋胸膜,肺胸膜,膈胸膜,胸膜顶,B\n",
        )
        archive.writestr(
            "CMMLU-sha/data/test/anatomy.csv",
            ",Question,A,B,C,D,Answer\n0,女性生殖腺是,卵巢,前庭大腺,前庭球,乳腺,A\n",
        )

    dataset = CMMLUDataset(str(zip_path), subjects=["anatomy"])

    assert dataset.dataset_dict["dev"][0]["question"] == "壁胸膜的分部不包括"
    assert dataset.dataset_dict["dev"][0]["answer"] == "B"
    assert dataset.dataset_dict["test"][0]["question"] == "女性生殖腺是"
    assert dataset.dataset_dict["test"][0]["answer"] == "A"
