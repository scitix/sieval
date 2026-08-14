"""Unit tests for the SysBench dataset wrapper.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import json

from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.sysbench import (
    SYSBENCH_COMMIT,
    SYSBENCH_URL,
    SysBenchDataset,
)


def _criteria(*specs) -> dict:
    return {
        str(cid): {
            "criteria_id": cid,
            "criteria_content": content,
            "criteria_type": ctype,
        }
        for cid, content, ctype in specs
    }


def _session(system_id: int = 7, scenario: object = "答疑") -> dict:
    """One upstream session, in the released file's shape."""
    return {
        "system_id": system_id,
        "领域": "教育",
        "场景": scenario,
        "system_prompt": "你是一位中文助教。",
        "messages": [
            {"role": "system", "content": "你是一位中文助教。"},
            {"role": "user", "content": "第一轮问题"},
            {"role": "assistant", "content": "标准答案一"},
            {"role": "user", "content": "第二轮问题"},
            {"role": "assistant", "content": "标准答案二"},
        ],
        # Upstream keys the per-turn info by the user message's TEXT.
        "prompt_infos": {
            "第一轮问题": {
                "alignment": "align",
                "criteria": _criteria((1, "必须用中文回答", "格式约束")),
            },
            "第二轮问题": {
                "alignment": "misalign",
                "criteria": _criteria(
                    (1, "必须用中文回答", "格式约束"),
                    (2, "不得提及价格", "内容约束"),
                ),
            },
        },
    }


def _write(tmp_path, sessions):
    path = tmp_path / "system_benchmark_eval_datas.json"
    path.write_text(json.dumps(sessions, ensure_ascii=False), encoding="utf-8")
    return path


def _load(tmp_path, sessions) -> list[dict]:
    test_set = SysBenchDataset(str(_write(tmp_path, sessions))).test_set
    assert test_set is not None
    return list(test_set)


def test_source_pins_the_upstream_commit():
    # Addressed at a commit, not a branch: a branch would let the rows move
    # under a checksum that then fails, but only after a download.
    (source,) = get_dataset_meta(SysBenchDataset).source
    assert source == f"url:{SYSBENCH_URL}"
    assert f"/{SYSBENCH_COMMIT}/" in source


def test_one_row_is_one_whole_session(tmp_path):
    # The headline protocol feeds the model its own prior replies, so the five
    # turns are one sequential walk -- a row per turn could not represent it.
    rows = _load(tmp_path, [_session(1), _session(2)])
    assert len(rows) == 2
    assert rows[0]["n_turns"] == 2


def test_turns_keep_their_order_users_answers_and_checklists(tmp_path):
    (row,) = _load(tmp_path, [_session()])
    turns = json.loads(row["turns_json"])
    assert [t["user"] for t in turns] == ["第一轮问题", "第二轮问题"]
    # The reference answer rides along for the with-GT ablation, never scoring.
    assert [t["assistant"] for t in turns] == ["标准答案一", "标准答案二"]
    assert [t["alignment"] for t in turns] == ["align", "misalign"]
    assert list(turns[1]["criteria"]) == ["1", "2"]
    assert turns[1]["criteria"]["2"]["criteria_type"] == "内容约束"


def test_rows_are_ordered_by_session_id(tmp_path):
    # The file's own order is not sorted; a stable row order is what makes a
    # `--limit` run and a full run cover the same first N sessions.
    rows = _load(tmp_path, [_session(9), _session(2), _session(5)])
    assert [r["session_id"] for r in rows] == [2, 5, 9]


def test_a_null_scenario_does_not_break_the_arrow_schema(tmp_path):
    # Upstream leaves `场景` null on some sessions; a None in a string column
    # makes Arrow reject the batch.
    (row,) = _load(tmp_path, [_session(scenario=None)])
    assert row["scenario"] == ""
    assert row["domain"] == "教育"


def test_an_unlabelled_turn_keeps_its_place_in_the_walk(tmp_path):
    """A turn with no checklist still shapes what the later turns were asked from.

    Dropping it would change the history the model builds on, so it stays with
    an empty `criteria` for the scorer to skip.
    """
    session = _session()
    del session["prompt_infos"]["第一轮问题"]
    (row,) = _load(tmp_path, [session])
    turns = json.loads(row["turns_json"])
    assert len(turns) == 2
    assert turns[0]["criteria"] == {}
    assert turns[0]["user"] == "第一轮问题"
    assert turns[1]["criteria"] != {}


def test_load_accepts_the_downloaded_directory(tmp_path):
    # `sieval dataset download` hands over a directory; a hand-placed copy is
    # addressed by file. Both must work.
    _write(tmp_path, [_session()])
    ds = SysBenchDataset(str(tmp_path))
    test_set, train_set = ds.test_set, ds.train_set
    assert test_set is not None
    assert len(test_set) == 1
    # SysBench ships no train/test split, so a task asking for `train` must get
    # the data rather than an empty split.
    assert train_set is not None
    assert len(train_set) == 1


def test_upstream_declares_no_license_and_the_meta_says_so():
    # Null is the measured answer here, not a missing field: there is no LICENSE
    # file at the pinned commit. A guessed permissive label would be worse than
    # none, because it invites redistribution.
    assert get_dataset_meta(SysBenchDataset).license is None
