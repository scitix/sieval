"""Tests for the RULER headline aggregation.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import pytest

from sieval.cli.leaderboard._ruler_avg import (
    length_tag,
    parse_length,
    ruler_average,
)


@pytest.mark.parametrize(
    ("task_name", "expected"),
    [
        ("ruler_cwe_64k", 65536),
        ("ruler_vt_128k", 131072),
        ("ruler_qa_4096", 4096),
        ("ruler_cwe", None),  # no suffix
        ("ruler_niah_single_1", None),
    ],
)
def test_parse_length(task_name, expected):
    assert parse_length(task_name) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [(65536, "64k"), (4096, "4k"), (131072, "128k"), (5000, "5000")],
)
def test_length_tag(length, expected):
    assert length_tag(length) == expected


def test_average_groups_by_length():
    runs = [
        ("m", "ruler_cwe_64k", {"score": 80.0}),
        ("m", "ruler_vt_64k", {"score": 90.0}),
        ("m", "ruler_cwe_128k", {"score": 40.0}),
        ("m", "ruler_vt_128k", {"score": 60.0}),
    ]
    out = ruler_average(runs)
    assert out["m"]["per_length"]["64k"] == {"avg": 85.0, "n": 2}
    assert out["m"]["per_length"]["128k"] == {"avg": 50.0, "n": 2}
    # overall = mean of all four subtask scores
    assert out["m"]["overall"] == {"avg": 67.5, "n": 4}


def test_ignores_non_ruler_and_non_numeric():
    runs = [
        ("m", "ruler_cwe_64k", {"score": 80.0}),
        ("m", "gsm8k_kshot_base_gen", {"score": 95.0}),  # not ruler_
        ("m", "ruler_vt_64k", {"score": None}),  # non-numeric
        ("m", "ruler_fwe_64k", {}),  # no score
        ("m", "ruler_qa_64k", {"score": True}),  # bool is not a real score
    ]
    out = ruler_average(runs)
    # only the one valid ruler_ run counts
    assert out["m"]["per_length"]["64k"] == {"avg": 80.0, "n": 1}
    assert out["m"]["overall"]["n"] == 1


def test_single_length_buckets_under_all():
    runs = [
        ("m", "ruler_cwe", {"score": 70.0}),
        ("m", "ruler_vt", {"score": 80.0}),
    ]
    out = ruler_average(runs)
    assert out["m"]["per_length"]["all"] == {"avg": 75.0, "n": 2}
    assert out["m"]["overall"] == {"avg": 75.0, "n": 2}


def test_multiple_models_kept_separate():
    runs = [
        ("a", "ruler_cwe_64k", {"score": 100.0}),
        ("b", "ruler_cwe_64k", {"score": 50.0}),
    ]
    out = ruler_average(runs)
    assert out["a"]["overall"]["avg"] == 100.0
    assert out["b"]["overall"]["avg"] == 50.0


def test_empty_runs():
    assert ruler_average([]) == {}
