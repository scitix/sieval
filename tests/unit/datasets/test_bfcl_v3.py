import json

import pytest

from sieval.community.bfcl_v3 import GOLDLESS_CATEGORIES
from sieval.datasets._bfcl_v3 import load_categories


def _write(root, category, rows, gold=None):
    (root / f"BFCL_v3_{category}.json").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    if gold is not None:
        answers = root / "possible_answer"
        answers.mkdir(exist_ok=True)
        (answers / f"BFCL_v3_{category}.json").write_text(
            "\n".join(json.dumps(g) for g in gold), encoding="utf-8"
        )


def test_gold_is_joined_onto_its_prompt_row(tmp_path):
    _write(
        tmp_path,
        "simple",
        [
            {
                "id": "simple_0",
                "question": [[{"role": "user", "content": "hi"}]],
                "function": [{"name": "f"}],
            }
        ],
        gold=[{"id": "simple_0", "ground_truth": [{"f": {"a": [1]}}]}],
    )
    row = load_categories(str(tmp_path), {"simple": 1})["test"][0]
    assert row["id"] == "simple_0"
    assert row["language"] == "Python"
    assert json.loads(row["ground_truth"]) == [{"f": {"a": [1]}}]
    assert row["question"] == [{"role": "user", "content": "hi"}]


def test_a_goldless_category_needs_no_answer_file(tmp_path):
    assert "irrelevance" in GOLDLESS_CATEGORIES
    _write(
        tmp_path,
        "irrelevance",
        [
            {
                "id": "irrelevance_0",
                "question": [[{"role": "user", "content": "hi"}]],
                "function": [{"name": "f"}],
            }
        ],
    )
    row = load_categories(str(tmp_path), {"irrelevance": 1})["test"][0]
    assert row["ground_truth"] is None


def test_java_and_javascript_carry_their_language(tmp_path):
    for category, language in (("java", "Java"), ("javascript", "JavaScript")):
        _write(
            tmp_path,
            category,
            [
                {
                    "id": f"{category}_0",
                    "question": [[{"role": "user", "content": "x"}]],
                    "function": [{"name": "f"}],
                }
            ],
            gold=[{"id": f"{category}_0", "ground_truth": [{"f": {}}]}],
        )
        row = load_categories(str(tmp_path), {category: 1})["test"][0]
        assert row["language"] == language


def test_a_moved_row_count_raises(tmp_path):
    _write(
        tmp_path,
        "simple",
        [
            {
                "id": "simple_0",
                "question": [[{"role": "user", "content": "hi"}]],
                "function": [],
            }
        ],
        gold=[{"id": "simple_0", "ground_truth": []}],
    )
    with pytest.raises(ValueError, match="expected 400"):
        load_categories(str(tmp_path), {"simple": 400})


def test_a_multi_turn_row_is_refused(tmp_path):
    _write(
        tmp_path,
        "simple",
        [
            {
                "id": "simple_0",
                "question": [
                    [{"role": "user", "content": "a"}],
                    [{"role": "user", "content": "b"}],
                ],
                "function": [],
            }
        ],
        gold=[{"id": "simple_0", "ground_truth": []}],
    )
    with pytest.raises(ValueError, match="turns"):
        load_categories(str(tmp_path), {"simple": 1})


def test_gold_joins_on_the_universal_index_not_the_whole_id(tmp_path):
    """Upstream sorts both files on the universal index and pairs them
    positionally, so a row whose sub-indices disagree between the two files is
    still graded. `live_multiple_1052-79-0` / `live_multiple_1052-279-0` is that
    row at the pinned revision; joining on the whole id would drop it.
    """
    _write(
        tmp_path,
        "live_multiple",
        [
            {
                "id": "live_multiple_1052-79-0",
                "question": [[{"role": "user", "content": "volume 50"}]],
                "function": [{"name": "set_volume"}],
            }
        ],
        gold=[
            {
                "id": "live_multiple_1052-279-0",
                "ground_truth": [{"set_volume": {"volume": [50]}}],
            }
        ],
    )
    row = load_categories(str(tmp_path), {"live_multiple": 1})["test"][0]
    assert row["id"] == "live_multiple_1052-79-0"
    assert json.loads(row["ground_truth"]) == [{"set_volume": {"volume": [50]}}]


def test_a_repeated_universal_index_raises(tmp_path):
    """Two rows sharing the join key make the pairing ambiguous. Upstream cannot
    see this -- it only checks that the two files are the same length.
    """
    _write(
        tmp_path,
        "live_multiple",
        [
            {
                "id": "live_multiple_0-1-0",
                "question": [[{"role": "user", "content": "a"}]],
                "function": [],
            },
            {
                "id": "live_multiple_0-2-0",
                "question": [[{"role": "user", "content": "b"}]],
                "function": [],
            },
        ],
        gold=[
            {"id": "live_multiple_0-1-0", "ground_truth": []},
            {"id": "live_multiple_0-2-0", "ground_truth": []},
        ],
    )
    with pytest.raises(ValueError, match="sharing a universal index"):
        load_categories(str(tmp_path), {"live_multiple": 2})


def test_a_gold_miss_raises(tmp_path):
    _write(
        tmp_path,
        "simple",
        [
            {
                "id": "simple_0",
                "question": [[{"role": "user", "content": "hi"}]],
                "function": [],
            }
        ],
        gold=[{"id": "simple_99", "ground_truth": []}],
    )
    with pytest.raises(ValueError, match="no possible_answer"):
        load_categories(str(tmp_path), {"simple": 1})
