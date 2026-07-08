"""Unit tests for the full MMMLU dataset loader.

AI-Generated Code - GPT-5-Codex (OpenAI)
"""

import csv

import pytest

from sieval.datasets.mmmlu import (
    MMMLUDataset,
    _normalize_answer,
    _normalize_locale,
    _normalize_subject,
)


def _write_csv(path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Question", "A", "B", "C", "D", "Answer", "Subject"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_normalizers_handle_mmmlu_shapes():
    assert _normalize_answer(" b ") == "B"
    assert _normalize_answer(2) == "C"
    assert _normalize_answer("9") == ""
    assert _normalize_subject("abstract_algebra_test.csv_zh-CN.csv") == (
        "abstract_algebra"
    )
    assert _normalize_locale("ZH-CN") == "zh_cn"
    assert _normalize_locale("ZH_CN") == "zh_cn"


def test_load_local_multilingual_csvs_adds_locale_and_category(tmp_path):
    _write_csv(
        tmp_path / "test" / "mmlu_ZH-CN.csv",
        [
            {
                "Question": "一加一等于几？",
                "A": "1",
                "B": "2",
                "C": "3",
                "D": "4",
                "Answer": "b",
                "Subject": "abstract_algebra_test.csv_zh-CN.csv",
            }
        ],
    )
    _write_csv(
        tmp_path / "test" / "mmlu_DE-DE.csv",
        [
            {
                "Question": "Was ist zwei plus zwei?",
                "A": "1",
                "B": "2",
                "C": "3",
                "D": "4",
                "Answer": "D",
                "Subject": "abstract_algebra_test.csv_de-DE.csv",
            }
        ],
    )

    dataset = MMMLUDataset(
        str(tmp_path),
        locales=["zh_cn", "de_de"],
        subjects=["abstract_algebra"],
    )

    test_set = dataset.test_set
    assert test_set is not None
    assert len(test_set) == 2
    assert test_set[0]["Locale"] == "zh_cn"
    assert test_set[0]["LocaleDisplayName"] == "Simplified Chinese"
    assert test_set[0]["Category"] == "stem"
    assert test_set[0]["Answer"] == "B"
    assert test_set[1]["Locale"] == "de_de"
    assert test_set[1]["LocaleDisplayName"] == "German"


def test_load_filters_categories(tmp_path):
    _write_csv(
        tmp_path / "mmlu_ZH-CN.csv",
        [
            {
                "Question": "Math",
                "A": "A",
                "B": "B",
                "C": "C",
                "D": "D",
                "Answer": "A",
                "Subject": "abstract_algebra",
            },
            {
                "Question": "Ethics",
                "A": "A",
                "B": "B",
                "C": "C",
                "D": "D",
                "Answer": "A",
                "Subject": "business_ethics",
            },
        ],
    )

    dataset = MMMLUDataset(str(tmp_path), locales=["zh_cn"], categories=["other"])

    test_set = dataset.test_set
    assert test_set is not None
    assert len(test_set) == 1
    assert test_set[0]["Subject"] == "business_ethics"


def test_load_rejects_unknown_locale(tmp_path):
    with pytest.raises(ValueError, match="Unknown MMMLU locale"):
        MMMLUDataset(str(tmp_path), locales=["xx_yy"])
