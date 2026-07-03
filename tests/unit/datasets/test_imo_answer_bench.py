"""Unit tests for the IMO-AnswerBench dataset wrapper (official v2 CSV).

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from unittest.mock import patch

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.datasets.imo_answer_bench as imo_module
from sieval.datasets.imo_answer_bench import (
    IMO_ANSWER_BENCH_URL,
    IMOAnswerBenchDataset,
)


def _v2_dict() -> HFDatasetDict:
    # v2 columns: Problem ID / Problem / Short Answer / Category / Subcategory /
    # Source. combinatorics-005 is a v2-discriminating fix (v1 was 1431655765).
    row = {
        "Problem ID": "imo-bench-combinatorics-005",
        "Problem": "How many ...?",
        "Short Answer": "1431655764",
        "Category": "Combinatorics",
        "Subcategory": "Counting",
        "Source": "IMO Shortlist",
    }
    ds = HFDataset.from_list([row])
    return HFDatasetDict({"train": ds, "test": ds})


def test_source_pins_official_v2_csv():
    # The dataset must point at the official answerbench_v2.csv (deprecated v1
    # hf mirror is NOT used). check_datasets separately enforces the checksum.
    assert IMO_ANSWER_BENCH_URL.startswith("url:")
    assert IMO_ANSWER_BENCH_URL.endswith("/imobench/answerbench_v2.csv")
    assert "google-deepmind/superhuman" in IMO_ANSWER_BENCH_URL


def test_load_reads_v2_csv_and_renames_columns():
    hf_dict = _v2_dict()
    dataset = IMOAnswerBenchDataset(_hf_dict=hf_dict)
    with (
        patch.object(imo_module, "load_dataset", return_value=hf_dict) as mock_load,
        patch("os.path.isdir", return_value=True),
    ):
        loaded = dataset.load("/staged/imo_answer_bench")

    # reads a CSV, resolving answerbench_v2.csv under the staged directory
    assert mock_load.call_args.args[0] == "csv"
    data_files = mock_load.call_args.kwargs["data_files"]
    assert data_files["train"].endswith("/imo_answer_bench/answerbench_v2.csv")
    assert data_files["test"].endswith("/imo_answer_bench/answerbench_v2.csv")

    # renamed to the shared math schema; both splits present, no leftover columns
    for split in ("train", "test"):
        cols = loaded[split].column_names
        assert "question" in cols and "answer" in cols
        assert "Problem" not in cols and "Short Answer" not in cols
    assert loaded["test"][0]["answer"] == "1431655764"


def test_load_accepts_direct_csv_file_path():
    hf_dict = _v2_dict()
    dataset = IMOAnswerBenchDataset(_hf_dict=hf_dict)
    with (
        patch.object(imo_module, "load_dataset", return_value=hf_dict) as mock_load,
        patch("os.path.isdir", return_value=False),
    ):
        dataset.load("/staged/imo_answer_bench/answerbench_v2.csv")

    data_files = mock_load.call_args.kwargs["data_files"]
    assert data_files["test"] == "/staged/imo_answer_bench/answerbench_v2.csv"
