"""Unit tests for the Multi-IF task.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import asyncio
import subprocess
import sys

import pytest

from sieval.tasks.multi_if_0shot_gen import MultiIFZeroShotGenTask


def test_import_does_not_pull_evaluation_lib():
    # evaluation_lib pulls langdetect/nltk/emoji and the 3.5k-line checker
    # fork; registration must not import it.
    code = (
        "import sys\n"
        "import sieval.tasks.multi_if_0shot_gen\n"
        "assert 'sieval.community.multi_if.evaluation_lib' not in sys.modules, "
        "'evaluation_lib must be lazy-imported'\n"
    )
    # Run in a fresh interpreter so pytest's already-loaded modules
    # don't mask the check.
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def _task() -> MultiIFZeroShotGenTask:
    return MultiIFZeroShotGenTask.__new__(MultiIFZeroShotGenTask)


class _Ctx:
    def __init__(self, raw):
        self.raw_sample = raw


def _sample(n_turns: int = 3) -> dict:
    # Turn t's constraints extend turn t-1's, as they do in the real data.
    turns = [
        {
            "prompt": "write something in lowercase",
            "instruction_id_list": ["change_case:english_lowercase"],
            "kwargs": ["{}"],
        },
        {
            "prompt": "now end with the exact phrase",
            "instruction_id_list": [
                "change_case:english_lowercase",
                "startend:end_checker",
            ],
            "kwargs": ["{}", '{"end_phrase": "the end."}'],
        },
        {
            "prompt": "and use at least three words",
            "instruction_id_list": [
                "change_case:english_lowercase",
                "startend:end_checker",
                "length_constraints:number_words",
            ],
            "kwargs": [
                "{}",
                '{"end_phrase": "the end."}',
                '{"relation": "at least", "num_words": 3}',
            ],
        },
    ]
    return {"key": "k:1:en", "language": "English", "turns": turns[:n_turns]}


def _judge(responses: list[str], n_turns: int = 3) -> dict:
    task = _task()
    post = {
        "rollouts": [
            {
                "index": 0,
                "extracted": True,
                "prediction": [
                    {"turn": i, "response": r} for i, r in enumerate(responses, start=1)
                ],
            }
        ]
    }
    _final, judgement = asyncio.run(task.feedback(post, _Ctx(_sample(n_turns))))
    return judgement


class _StubOutput:
    def __init__(self, text):
        self.texts = [text] if text is not None else []


class _StubModel:
    """Records the conversation it is handed on each call."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.seen: list[list[dict]] = []

    async def agenerate(self, messages, **kwargs):
        # Copy: the task keeps mutating the same list across turns.
        self.seen.append([dict(m) for m in messages])
        return _StubOutput(self._replies[len(self.seen) - 1])


def _run_infer(replies, n_turns=3):
    task = _task()
    model = _StubModel(replies)
    task._model = model
    raw = _sample(n_turns)
    ctx = _Ctx(raw)
    pre = asyncio.run(task.preprocess(raw, ctx))
    outputs = asyncio.run(task.infer(pre, ctx))
    return model, outputs


def test_infer_feeds_each_turn_the_preceding_conversation():
    model, outputs = _run_infer(["reply one", "reply two", "reply three"])
    assert len(outputs) == 3
    assert len(model.seen) == 3

    # Turn 1: just the opening user message.
    assert model.seen[0] == [
        {"role": "user", "content": "write something in lowercase"}
    ]
    # Turn 2: turn 1 plus the model's own reply plus the new user turn.
    assert model.seen[1] == [
        {"role": "user", "content": "write something in lowercase"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "now end with the exact phrase"},
    ]
    # Turn 3: the whole history, in order, ending on the third user turn.
    assert [m["role"] for m in model.seen[2]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert model.seen[2][3] == {"role": "assistant", "content": "reply two"}
    assert model.seen[2][4]["content"] == "and use at least three words"


def test_infer_stops_at_the_last_turn_of_a_two_turn_conversation():
    model, outputs = _run_infer(["reply one", "reply two"], n_turns=2)
    assert len(outputs) == 2
    assert len(model.seen) == 2


def test_infer_continues_after_a_choiceless_response():
    # An aborted turn has no choices; the conversation must carry on with an
    # empty assistant turn rather than raising.
    model, outputs = _run_infer([None, "reply two", "reply three"])
    assert len(outputs) == 3
    assert model.seen[1][1] == {"role": "assistant", "content": ""}


def test_preprocess_records_only_the_first_turn():
    task = _task()
    raw = _sample()
    record = asyncio.run(task.preprocess(raw, _Ctx(raw)))
    assert record["prompt"] == [
        {"role": "user", "content": "write something in lowercase"}
    ]
    assert record["extra"]["n_turns"] == 3
    # The later user turns are recorded even though they are not sent yet.
    assert len(record["extra"]["later_turn_prompts"]) == 2
    # Reference is the per-turn cumulative constraint list.
    assert [len(ids) for ids in record["reference"]] == [1, 2, 3]


def test_feedback_grades_each_turn_against_its_cumulative_constraints():
    # Turn 1 satisfies its only constraint; turn 2 misses the end phrase;
    # turn 3 satisfies all three.
    judgement = _judge(
        [
            "all lowercase here.",
            "still lowercase but wrong ending",
            "still lowercase and long enough. the end.",
        ]
    )
    m = judgement["metrics"]
    assert m["turn_1_strict_follow_all"] is True
    assert m["turn_2_strict_follow_all"] is False
    assert m["turn_3_strict_follow_all"] is True
    # Turn 2 kept 1 of its 2 constraints.
    assert m["turn_2_strict_instruction_level"] == pytest.approx(0.5)
    # A single failed turn makes the whole conversation incorrect.
    assert judgement["rollouts"][0]["correct"] is False


def test_feedback_handles_a_two_turn_conversation():
    judgement = _judge(["all lowercase.", "lowercase. the end."], n_turns=2)
    assert judgement["extra"]["n_turns"] == 2
    assert "turn_3" not in judgement["extra"]
    assert judgement["rollouts"][0]["correct"] is True


def test_missing_turn_response_scores_zero_rather_than_raising():
    # An aborted turn arrives as "": every constraint fails, nothing raises.
    judgement = _judge(["", "", ""])
    m = judgement["metrics"]
    assert m["turn_1_strict_follow_all"] is False
    assert m["turn_1_strict_instruction_level"] == 0.0


def test_postprocess_reports_no_extraction_only_when_every_turn_is_blank():
    task = _task()

    class _Out:
        def __init__(self, text):
            self.texts = [text]

    blank = asyncio.run(task.postprocess([_Out(""), _Out("  ")], _Ctx(_sample())))
    assert blank["rollouts"][0]["extracted"] is False

    # A partly-blank conversation is still a real answer that scores badly.
    partial = asyncio.run(task.postprocess([_Out(""), _Out("hi")], _Ctx(_sample())))
    assert partial["rollouts"][0]["extracted"] is True
    assert partial["rollouts"][0]["prediction"][1] == {"turn": 2, "response": "hi"}


def test_report_pools_instruction_counts_and_skips_absent_turns():
    class _Final:
        def __init__(self, judgement):
            self.feedback_result = judgement

    # One three-turn conversation and one two-turn conversation.
    three = _judge(
        ["all lowercase.", "lowercase. the end.", "lowercase and long. the end."]
    )
    two = _judge(["all lowercase.", "lowercase. the end."], n_turns=2)
    report = asyncio.run(_task().report([_Final(three), _Final(two)], []))

    # Turn 3 exists for only one of the two conversations.
    assert report["turn_1_prompts_number"] == 2
    assert report["turn_3_prompts_number"] == 1
    # Both conversations pass everything, so every cell is 100.
    assert report["turn_1_all_languages_overall"] == pytest.approx(100.0)
    assert report["turn_1_English_overall"] == pytest.approx(100.0)
    assert report["score"] == pytest.approx(100.0)
    assert report["fails"] == 0
