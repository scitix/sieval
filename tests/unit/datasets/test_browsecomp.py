"""Unit tests for the BrowseComp dataset wrapper (download + in-memory decrypt).

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import base64
import csv

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.browsecomp import (
    CSV_BASENAME,
    CSV_SHA256,
    BrowseCompDataset,
    derive_key,
)


def _encrypt(plaintext: str, password: str) -> str:
    """Inverse of the module's ``decrypt`` (XOR is symmetric)."""
    data = plaintext.encode()
    key = derive_key(password, len(data))
    return base64.b64encode(
        bytes(a ^ b for a, b in zip(data, key, strict=True))
    ).decode()


def _write_csv(path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["problem", "answer", "problem_topic", "canary"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _stub() -> HFDatasetDict:
    # A pre-built dict so the Dataset can be constructed without hitting load().
    return HFDatasetDict({"test": HFDataset.from_list([{"original_index": 0}])})


def test_source_pins_url_and_checksum():
    meta = get_dataset_meta(BrowseCompDataset)
    assert meta.source == (
        f"url:https://openaipublic.blob.core.windows.net/simple-evals/{CSV_BASENAME}",
    )
    assert dict(meta.checksums)[CSV_BASENAME] == f"sha256:{CSV_SHA256}"


def test_load_decrypts_rows(tmp_path):
    canary = "browsecomp-canary-token"
    csv_path = tmp_path / CSV_BASENAME
    _write_csv(
        csv_path,
        [
            {
                "problem": _encrypt("Who wrote Hamlet?", canary),
                "answer": _encrypt("William Shakespeare", canary),
                "problem_topic": "Art",
                "canary": canary,
            }
        ],
    )
    dataset = BrowseCompDataset(_hf_dict=_stub())
    # dir path -> joins CSV_BASENAME
    loaded = dataset.load(str(tmp_path))
    row = loaded["test"][0]
    assert row["problem"] == "Who wrote Hamlet?"
    assert row["answer"] == "William Shakespeare"
    assert row["problem_topic"] == "Art"
    assert row["original_index"] == 0
    # decrypted plaintext is NOT written back to the on-disk csv (stays encrypted)
    with open(csv_path, encoding="utf-8") as f:
        assert "William Shakespeare" not in f.read()


def test_empty_test_split_raises(tmp_path):
    csv_path = tmp_path / CSV_BASENAME
    _write_csv(csv_path, [])  # header only, no rows
    dataset = BrowseCompDataset(_hf_dict=_stub())
    with pytest.raises(ValueError, match="empty 'test' split"):
        dataset.load(str(csv_path))
