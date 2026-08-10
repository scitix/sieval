"""Unit tests for the IHEval dataset loader.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
from pathlib import Path

import pytest

from sieval.datasets.iheval import IHEvalDataset


def _write(root: Path, cell: str, rows: list[dict]) -> None:
    path = root / "iheval" / cell / "input_data.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _tree(root: Path) -> Path:
    # One cell per answer shape, so the JSON-encoded columns are exercised by
    # types that could not share an Arrow schema.
    _write(
        root,
        "rule-following/single-turn/conflict/default",
        [
            {
                "id": 1000,
                "system": "No commas.",
                "instruction": "Summarize the page.",
                "answer": {
                    "instruction_id_list": ["punctuation:no_comma"],
                    "kwargs": [{}],
                },
            }
        ],
    )
    _write(
        root,
        "rule-following/multi-turn/reference/default",
        [
            {
                "id": 1000,
                "system": None,
                "conversation_history": ["first ask", "first reply"],
                "instruction": "Follow up.",
                "answer": {"instruction_id_list": [], "kwargs": []},
            }
        ],
    )
    _write(
        root,
        "task-execution/lang-detect/aligned/system_lang_detect_weak",
        [
            {
                "id": 1,
                "system": "Detect the language.",
                "instruction": "希伯伦",
                "summary": "a summary nothing scores",
                "answer": ["Chinese", "中文"],
            }
        ],
    )
    _write(
        root,
        "tool-use/slack-user/conflict/tool_prompt_strong",
        [
            {
                "id": 1,
                "system": "Use the tool.",
                "instruction": "Shortest name?",
                "tool": {
                    "definition": {
                        "name": "get_users",
                        "description": "",
                        "parameters": {},
                    },
                    "call": {"id": "c1", "name": "get_users", "arguments": {}},
                    "return": {"id": "c1", "name": "get_users", "content": "- Jack"},
                },
                "answer": "Jack",
            }
        ],
    )
    # Not a cell: a source corpus sitting in the same tree.
    (root / "iheval/task-execution/lang-detect/xlsum.json").write_text(
        json.dumps([{"text": "unused"}]), encoding="utf-8"
    )
    return root


def _rows(root: Path) -> list[dict]:
    test_set = IHEvalDataset(str(root)).test_set
    assert test_set is not None
    return list(test_set)


def _by_subtask(root: Path) -> dict[str, dict]:
    return {row["subtask"]: row for row in _rows(root)}


def test_loads_every_cell_into_one_test_split(tmp_path: Path):
    dataset = IHEvalDataset(str(_tree(tmp_path)))

    assert set(dataset.dataset_dict) == {"test"}
    rows = _rows(_tree(tmp_path))
    assert len(rows) == 4
    assert {row["subtask"] for row in rows} == {
        "single-turn",
        "multi-turn",
        "lang-detect",
        "slack-user",
    }


def test_rows_are_ordered_by_category_subtask_then_paper_setting_order(tmp_path: Path):
    rows = _rows(_tree(tmp_path))
    # reference sorts before aligned/conflict even though it is alphabetically
    # last, so a run's baseline cell is not buried at the end.
    assert [(row["subtask"], row["setting"]) for row in rows] == [
        ("multi-turn", "reference"),
        ("single-turn", "conflict"),
        ("lang-detect", "aligned"),
        ("slack-user", "conflict"),
    ]


def test_uid_is_unique_and_carries_the_cell(tmp_path: Path):
    rows = _rows(_tree(tmp_path))
    uids = [row["uid"] for row in rows]
    assert len(set(uids)) == len(uids)
    # Upstream ids repeat across cells (id 1000 and id 1 each appear twice), so
    # the cell path is what makes the key stable.
    assert "rule-following/single-turn/conflict/default#1000" in uids
    assert "tool-use/slack-user/conflict/tool_prompt_strong#1" in uids


def test_absent_system_and_history_become_empty_not_null(tmp_path: Path):
    rows = _by_subtask(_tree(tmp_path))
    multi = rows["multi-turn"]
    assert multi["system"] == ""
    assert multi["conversation_history"] == ["first ask", "first reply"]
    single = rows["single-turn"]
    assert single["system"] == "No commas."
    assert single["conversation_history"] == []


def test_heterogeneous_answers_round_trip_through_json(tmp_path: Path):
    rows = _by_subtask(_tree(tmp_path))
    assert json.loads(rows["lang-detect"]["answer_json"]) == ["Chinese", "中文"]
    assert json.loads(rows["slack-user"]["answer_json"]) == "Jack"
    assert json.loads(rows["single-turn"]["answer_json"])["instruction_id_list"] == [
        "punctuation:no_comma"
    ]


def test_tool_is_carried_only_where_it_exists(tmp_path: Path):
    rows = _by_subtask(_tree(tmp_path))
    assert rows["single-turn"]["tool_json"] == ""
    tool = json.loads(rows["slack-user"]["tool_json"])
    assert tool["call"]["name"] == "get_users"
    assert tool["return"]["content"] == "- Jack"


def test_source_corpora_in_the_same_tree_are_not_mistaken_for_cells(tmp_path: Path):
    # xlsum.json sits beside the lang-detect cells and has no `answer` field;
    # reading it as a cell would raise rather than silently add rows.
    rows = _rows(_tree(tmp_path))
    assert len(rows) == 4


def test_missing_data_points_at_the_download_command(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="sieval dataset download iheval"):
        IHEvalDataset(str(tmp_path))
