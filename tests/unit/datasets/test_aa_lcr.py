"""Unit tests for the AA-LCR dataset loader.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import csv
import unicodedata
import zipfile
from pathlib import Path

import pytest

from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.aa_lcr import (
    AA_LCR_REVISION,
    AALCRDataset,
    _member_utf8_name,
)


def _unflagged_zipinfo(utf8_bytes: bytes) -> zipfile.ZipInfo:
    """A ZipInfo mimicking this archive's members: UTF-8 name bytes stored
    WITHOUT the 0x800 flag, so ``zipfile`` exposes them cp437-decoded."""
    info = zipfile.ZipInfo()
    info.filename = utf8_bytes.decode("cp437")
    info.flag_bits = 0x08  # data descriptor present; crucially no 0x800
    return info


_CSV_COLUMNS = [
    "",
    "document_category",
    "document_set_id",
    "question_id",
    "question",
    "answer",
    "data_source_filenames",
    "data_source_urls",
    "input_tokens",
]


def _stage_dataset(
    root: Path, rows: list[dict[str, str]], docs: dict[str, str]
) -> None:
    """Write a minimal AA-LCR CSV + extracted-text zip under *root*.

    *docs* maps ``"{category}/{set_id}/{filename}"`` to file contents.
    """
    with (root / "AA-LCR_Dataset.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"": str(i), "data_source_urls": "", **row})

    zip_dir = root / "extracted_text"
    zip_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_dir / "AA-LCR_extracted-text.zip", "w") as archive:
        for relpath, content in docs.items():
            archive.writestr(f"lcr/{relpath}", content)


def _one_question_rows() -> list[dict[str, str]]:
    return [
        {
            "document_category": "Academia",
            "document_set_id": "ac_markets",
            "question_id": "1",
            "question": "What is the trend?",
            "answer": "Rising",
            # Note: b before a — loader must preserve this order, not sort.
            "data_source_filenames": "b.txt;a.txt",
            "input_tokens": "1234",
        }
    ]


def test_source_pins_hf_revision():
    meta_source = get_dataset_meta(AALCRDataset).source
    assert meta_source == (f"hf:ArtificialAnalysis/AA-LCR@{AA_LCR_REVISION}",)


def test_load_assembles_documents_in_filename_order(tmp_path: Path):
    _stage_dataset(
        tmp_path,
        _one_question_rows(),
        {
            "Academia/ac_markets/a.txt": "ALPHA",
            "Academia/ac_markets/b.txt": "BRAVO",
        },
    )
    dataset = AALCRDataset(str(tmp_path))
    test = dataset.test_set
    assert test is not None
    assert len(test) == 1
    sample = test[0]
    assert sample["question_id"] == 1
    assert sample["document_category"] == "Academia"
    assert sample["answer"] == "Rising"
    assert sample["input_tokens"] == 1234
    # Order follows data_source_filenames ("b.txt;a.txt"), not lexical sort.
    assert sample["documents"] == ["BRAVO", "ALPHA"]
    assert sample["data_source_filenames"] == "b.txt;a.txt"


def test_missing_document_raises(tmp_path: Path):
    _stage_dataset(
        tmp_path,
        _one_question_rows(),
        {"Academia/ac_markets/a.txt": "ALPHA"},  # b.txt absent
    )
    with pytest.raises(FileNotFoundError, match="missing from the extracted-text"):
        AALCRDataset(str(tmp_path))


def test_empty_csv_raises_empty_split(tmp_path: Path):
    _stage_dataset(tmp_path, [], {})
    with pytest.raises(ValueError, match="empty 'test' split"):
        AALCRDataset(str(tmp_path))


def test_missing_csv_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="AA-LCR CSV not found"):
        AALCRDataset(str(tmp_path))


# --- member-name recovery: the real archive stores UTF-8 names without the
# 0x800 flag (so zipfile cp437-decodes them) and some are NFD; the loader must
# recover the CSV's NFC UTF-8 names to match, else documents look "missing". ---


def test_member_name_recovers_cp437_mojibake():
    # "EU’s" (U+2019) — the legal_eu_ai failure. UTF-8 bytes, flag off.
    name = "lcr/Legal/legal_eu_ai/EU’s Official Journal.txt"
    info = _unflagged_zipinfo(name.encode("utf-8"))
    # Sanity: zipfile would expose it as cp437 mojibake, not the real name.
    assert info.filename != name and "ΓÇÖ" in info.filename
    assert _member_utf8_name(info) == name


def test_member_name_recovers_nfd_to_nfc():
    # "Başev" stored NFD ("s" + combining cedilla U+0327) — the mkt_gaming case.
    nfd = "lcr/Marketing/mkt_gaming/Sinem Eyice Başev.txt"
    assert unicodedata.is_normalized("NFC", nfd) is False
    info = _unflagged_zipinfo(nfd.encode("utf-8"))
    recovered = _member_utf8_name(info)
    assert recovered == unicodedata.normalize("NFC", nfd)
    assert "ş" in recovered  # ş (NFC single codepoint)


def test_member_name_utf8_flagged_passthrough():
    # A well-formed archive (0x800 set) is decoded correctly by zipfile already;
    # recovery must not double-decode it, only NFC-normalize.
    name = "lcr/Academia/x/Café.txt"
    info = zipfile.ZipInfo(filename=name)
    info.flag_bits = 0x800
    assert _member_utf8_name(info) == unicodedata.normalize("NFC", name)


def test_member_name_genuine_cp437_falls_through():
    # A name whose bytes aren't valid UTF-8 is left as zipfile decoded it.
    info = zipfile.ZipInfo(filename="ÿ.txt")  # cp437 char, lone 0x98 byte
    info.flag_bits = 0x08
    assert _member_utf8_name(info) == "ÿ.txt"


def test_load_reads_nfd_named_document(tmp_path: Path):
    # End-to-end: CSV references the NFC name; the archive member is NFC-stored
    # (writestr auto-sets the UTF-8 flag), so the lookup must resolve it.
    rows = [
        {
            "document_category": "Marketing",
            "document_set_id": "mkt_gaming",
            "question_id": "3",
            "question": "q",
            "answer": "a",
            "data_source_filenames": "Başev.txt",  # NFC in the CSV
            "input_tokens": "10",
        }
    ]
    _stage_dataset(tmp_path, rows, {"Marketing/mkt_gaming/Başev.txt": "CEDILLA"})
    dataset = AALCRDataset(str(tmp_path))
    test = dataset.test_set
    assert test is not None
    assert test[0]["documents"] == ["CEDILLA"]
