"""Unit tests for the SysBench task.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import asyncio
import json

import pytest

from sieval.community.sysbench import build_judge_prompt, parse_verdict
from sieval.core.tasks import (
    GRADER_OUTPUT_KEY,
    TaskContext,
    build_prediction_record,
    iter_grader_outputs,
)
from sieval.tasks.sysbench_0shot_gen import SysBenchZeroShotGenTask


def _task(history: str = "model") -> SysBenchZeroShotGenTask:
    task = SysBenchZeroShotGenTask.__new__(SysBenchZeroShotGenTask)
    task._history = history
    return task


class _Ctx:
    def __init__(self, raw):
        self.raw_sample = raw


def _final(judgement, prediction: str = "answered") -> TaskContext:
    # A real TaskContext, because `report` ends in `health_metrics`, which reads
    # the POSTPROCESS record -- a stub carrying only the judgement would pass
    # while the metric it feeds went missing.
    return TaskContext(
        sample_id=judgement["extra"]["session_id"],
        postprocess_result=build_prediction_record([prediction]),
        feedback_result=judgement,
    )


def _criteria(*specs) -> dict:
    """``(id, content, type)`` triples in upstream's keyed-by-id-string shape."""
    return {
        str(cid): {
            "criteria_id": cid,
            "criteria_content": content,
            "criteria_type": ctype,
        }
        for cid, content, ctype in specs
    }


def _sample(session_id: int = 1, n_turns: int = 3) -> dict:
    turns = [
        {
            "user": "第一轮问题",
            "assistant": "标准答案一",
            "alignment": "align",
            "criteria": _criteria((1, "必须用中文回答", "格式约束")),
        },
        {
            "user": "第二轮问题",
            "assistant": "标准答案二",
            "alignment": "misalign",
            "criteria": _criteria(
                (1, "必须用中文回答", "格式约束"),
                (2, "不得提及价格", "内容约束"),
            ),
        },
        {
            "user": "第三轮问题",
            "assistant": "标准答案三",
            "alignment": "align",
            "criteria": _criteria((1, "保持角色设定", "角色约束")),
        },
    ]
    return {
        "session_id": session_id,
        "domain": "教育",
        "scenario": "答疑",
        "system_prompt": "你是一位中文助教。",
        "n_turns": n_turns,
        "turns_json": json.dumps(turns[:n_turns], ensure_ascii=False),
    }


class _StubOutput:
    def __init__(self, text, finish_reasons=("stop",)):
        self.texts = [text] if text is not None else []
        self.finish_reasons = list(finish_reasons) if finish_reasons else None


class _StubModel:
    """Records the conversation it is handed on each call."""

    def __init__(self, replies, fail_at: int | None = None):
        self._replies = list(replies)
        self._fail_at = fail_at
        self.seen: list[list[dict]] = []
        self.kwargs: list[dict] = []

    async def agenerate(self, messages, **kwargs):
        # Copy: the task keeps mutating the same list across turns.
        self.seen.append([dict(m) for m in messages])
        self.kwargs.append(kwargs)
        if self._fail_at is not None and len(self.seen) == self._fail_at:
            raise RuntimeError("upstream returned 503")
        return _StubOutput(self._replies[len(self.seen) - 1])


def _reply(verdicts: dict[int, str], reason: str = "理由") -> str:
    body = ", ".join(f'{cid}: "{v}"' for cid, v in verdicts.items())
    return (
        "'''json\n{\n"
        f'  "评判理由": "{reason}",\n'
        f'  "评判结果": {{{body}}}\n'
        "}\n'''"
    )


def _run_infer(replies, n_turns=3, fail_at=None, history="model"):
    task = _task(history)
    model = _StubModel(replies, fail_at=fail_at)
    task._model = model
    raw = _sample(n_turns=n_turns)
    ctx = _Ctx(raw)
    pre = asyncio.run(task.preprocess(raw, ctx))
    outputs = asyncio.run(task.infer(pre, ctx))
    return model, outputs


def _run_to_feedback(replies, grader_replies, n_turns=3, history="model"):
    task = _task(history)
    task._model = _StubModel(replies)
    grader = _StubModel(grader_replies)
    task._grader = grader
    raw = _sample(n_turns=n_turns)
    ctx = _Ctx(raw)
    pre = asyncio.run(task.preprocess(raw, ctx))
    inf = asyncio.run(task.infer(pre, ctx))
    post = asyncio.run(task.postprocess(inf, ctx))
    _ok, judgement = asyncio.run(task.feedback(post, ctx))
    return grader, post, judgement


# --------------------------------------------------------------------------
# preprocess
# --------------------------------------------------------------------------


def test_preprocess_records_the_system_prompt_and_only_the_first_turn():
    task = _task()
    raw = _sample()
    record = asyncio.run(task.preprocess(raw, _Ctx(raw)))
    assert record["prompt"] == [
        {"role": "system", "content": "你是一位中文助教。"},
        {"role": "user", "content": "第一轮问题"},
    ]
    assert record["extra"]["n_turns"] == 3
    # The later user turns are part of the sample's input and are recorded even
    # though they have not been sent yet.
    assert record["extra"]["later_turn_prompts"] == ["第二轮问题", "第三轮问题"]
    assert record["extra"]["alignments"] == ["align", "misalign", "align"]
    # Reference is the per-turn constraint checklist.
    assert record["reference"] == [["1"], ["1", "2"], ["1"]]


# --------------------------------------------------------------------------
# infer -- the protocol
# --------------------------------------------------------------------------


def test_infer_carries_the_models_own_reply_into_the_next_turn():
    """The headline protocol: turn t is answered from the model's own history."""
    model, outputs = _run_infer(["回答一", "回答二", "回答三"])
    assert len(outputs) == 3
    assert model.seen[0] == [
        {"role": "system", "content": "你是一位中文助教。"},
        {"role": "user", "content": "第一轮问题"},
    ]
    assert model.seen[1][2] == {"role": "assistant", "content": "回答一"}
    assert model.seen[1][3] == {"role": "user", "content": "第二轮问题"}
    assert model.seen[2][4] == {"role": "assistant", "content": "回答二"}
    assert [m["role"] for m in model.seen[2]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]


def test_ground_truth_history_substitutes_the_dataset_answer():
    """The SS4.5 ablation: the model never sees what it actually said.

    This is the whole difference between the two published protocols, so it is
    asserted on the messages rather than on a score -- a score could match by
    coincidence on a stub.
    """
    model, _ = _run_infer(["回答一", "回答二", "回答三"], history="ground_truth")
    assert model.seen[1][2] == {"role": "assistant", "content": "标准答案一"}
    assert model.seen[2][4] == {"role": "assistant", "content": "标准答案二"}
    # The user turns are unchanged -- only the history differs.
    assert model.seen[2][5] == {"role": "user", "content": "第三轮问题"}


def test_infer_asks_for_exactly_one_sample_per_turn():
    # n>1 would fork the conversation: each later turn would have to be answered
    # once per branch.
    model, _ = _run_infer(["回答一", "回答二", "回答三"])
    assert [k.get("n") for k in model.kwargs] == [1, 1, 1]


def test_infer_continues_after_a_choiceless_response():
    model, outputs = _run_infer([None, "回答二", "回答三"])
    assert len(outputs) == 3
    assert model.seen[1][2] == {"role": "assistant", "content": ""}


def test_infer_keeps_the_answered_turns_when_a_later_turn_fails():
    # Failing the whole session would delete turns 1 and 2 from their own
    # denominators because turn 3 timed out.
    model, outputs = _run_infer(["回答一", "回答二", "回答三"], fail_at=3)
    assert len(model.seen) == 3, "the third turn must still have been attempted"
    assert [o.texts[0] for o in outputs] == ["回答一", "回答二"]


def test_infer_propagates_a_failure_on_the_very_first_turn():
    with pytest.raises(RuntimeError, match="503"):
        _run_infer(["回答一", "回答二", "回答三"], fail_at=1)


def test_history_mode_is_validated_at_construction():
    with pytest.raises(ValueError, match="history must be one of"):
        SysBenchZeroShotGenTask(
            dataset=None, model=None, grader={"model": "judge"}, history="gt"
        )


# --------------------------------------------------------------------------
# postprocess
# --------------------------------------------------------------------------


def test_postprocess_builds_one_rollout_holding_every_turn():
    task = _task()
    raw = _sample()
    inf = [_StubOutput("回答一"), _StubOutput("回答二")]
    post = asyncio.run(task.postprocess(inf, _Ctx(raw)))
    assert len(post["rollouts"]) == 1
    assert post["rollouts"][0]["prediction"] == [
        {"turn": 1, "response": "回答一"},
        {"turn": 2, "response": "回答二"},
    ]
    # A walk cut short and a session that answered every turn blankly are
    # different facts; only this tells them apart on disk.
    assert post["extra"]["n_answered"] == 2
    assert post["extra"]["finish_reasons"] == [["stop"], ["stop"]]


def test_postprocess_marks_an_entirely_blank_session_unextracted():
    task = _task()
    raw = _sample()
    post = asyncio.run(task.postprocess([_StubOutput(""), _StubOutput("")], _Ctx(raw)))
    assert post["rollouts"][0]["prediction"] is None
    assert post["rollouts"][0]["extracted"] is False
    # ...but the turns were still answered, so the count stays truthful.
    assert post["extra"]["n_answered"] == 2


# --------------------------------------------------------------------------
# feedback -- grading
# --------------------------------------------------------------------------


def test_feedback_calls_the_grader_once_per_turn_on_the_history_the_model_saw():
    grader, _post, judgement = _run_to_feedback(
        ["回答一", "回答二", "回答三"],
        [_reply({1: "是"}), _reply({1: "是", 2: "否"}), _reply({1: "是"})],
    )
    assert len(grader.seen) == 3
    # Turn 3's judge prompt carries the model's own turn-1 and turn-2 answers.
    third = grader.seen[2][0]["content"]
    assert "回答一" in third and "回答二" in third and "回答三" in third
    # Turn 1's does not mention later turns.
    assert "回答二" not in grader.seen[0][0]["content"]
    # Per-turn scores: 1/1, 1/2, 1/1.
    assert judgement["rollouts"][0]["metrics"]["turn_2_csr"] == 0.5
    assert judgement["rollouts"][0]["metrics"]["turn_2_all_satisfied"] is False
    assert judgement["score"] == pytest.approx((1.0 + 0.5 + 1.0) / 3)


def test_the_judge_sees_ground_truth_history_but_the_models_current_answer():
    """Upstream's with-GT script grades the model's answer against GT history.

    The two halves are easy to conflate, and getting the current turn wrong
    would silently grade the dataset's own reference answer -- a near-perfect
    score that looks like a very good model.
    """
    grader, _post, _judgement = _run_to_feedback(
        ["回答一", "回答二", "回答三"],
        [_reply({1: "是"}), _reply({1: "是", 2: "是"}), _reply({1: "是"})],
        history="ground_truth",
    )
    third = grader.seen[2][0]["content"]
    assert "标准答案一" in third, "history must be the dataset's answers"
    assert "回答一" not in third
    assert "回答三" in third, "the CURRENT turn is always the model's own answer"
    assert "标准答案三" not in third


def test_feedback_records_every_grader_call_and_the_profiler_sees_them_all():
    """A five-turn session bills one judge call per turn.

    Recording only the last (or a mapping the profiler reads as one call) would
    make four fifths of the grader spend vanish from profile.json.
    """
    _grader, _post, judgement = _run_to_feedback(
        ["回答一", "回答二", "回答三"],
        [_reply({1: "是"}), _reply({1: "是", 2: "是"}), _reply({1: "是"})],
    )
    stored = judgement["rollouts"][0]["extra"][GRADER_OUTPUT_KEY]
    assert isinstance(stored, list) and len(stored) == 3
    assert len(iter_grader_outputs(judgement)) == 3


def test_an_unreadable_verdict_scores_zero_and_is_counted_separately():
    # A grader that rambled resolves nothing. The constraint scores
    # not-satisfied, but the turn is flagged so grader drift stays telling
    # itself apart from a model that satisfied nothing.
    _grader, _post, judgement = _run_to_feedback(
        ["回答一", "回答二", "回答三"],
        [_reply({1: "是"}), "评委没有给出结论", _reply({1: "是"})],
        n_turns=3,
    )
    extra = judgement["extra"]
    assert extra["turn_2"]["n_grader_unparsed"] == 2
    assert extra["turn_2"]["n_graded"] == 0
    assert judgement["rollouts"][0]["metrics"]["turn_2_csr"] == 0.0


def test_a_session_that_ended_early_is_not_marked_correct():
    """SSR is a claim about a whole session, so an unreached turn forfeits it."""
    task = _task()
    task._grader = _StubModel([_reply({1: "是"}), _reply({1: "是", 2: "是"})])
    raw = _sample()
    post = {
        "rollouts": [
            {
                "index": 0,
                "extracted": True,
                "prediction": [
                    {"turn": 1, "response": "回答一"},
                    {"turn": 2, "response": "回答二"},
                ],
            }
        ],
        "extra": {"n_answered": 2},
    }
    _ok, judgement = asyncio.run(task.feedback(post, _Ctx(raw)))
    assert judgement["extra"]["n_answered"] == 2
    # Every graded turn passed, yet the session did not finish.
    assert judgement["rollouts"][0]["correct"] is False


def test_feedback_keeps_the_constraint_type_of_every_verdict():
    # The paper's per-category breakdown is pooled over constraints in `report`,
    # so the type has to survive on the record.
    _grader, _post, judgement = _run_to_feedback(
        ["回答一", "回答二", "回答三"],
        [_reply({1: "是"}), _reply({1: "是", 2: "否"}), _reply({1: "否"})],
    )
    assert judgement["extra"]["turn_2"]["criterion_types"] == {
        "1": "格式约束",
        "2": "内容约束",
    }
    assert judgement["extra"]["turn_3"]["criterion_verdicts"] == {"1": False}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _report(judgements, fails=()):
    task = _task()
    return asyncio.run(task.report([_final(j) for j in judgements], list(fails)))


def _judged_session(session_id, per_turn):
    """A minimal judgement record: ``per_turn`` is a list of verdict dicts."""
    extra = {
        "session_id": session_id,
        "domain": "教育",
        "n_turns": len(per_turn),
        "n_answered": len(per_turn),
        "history_mode": "model",
    }
    for index, (verdicts, types, alignment) in enumerate(per_turn, start=1):
        n_satisfied = sum(1 for v in verdicts.values() if v)
        extra[f"turn_{index}"] = {
            "criterion_verdicts": verdicts,
            "criterion_types": types,
            "alignment": alignment,
            "n_criteria": len(verdicts),
            "n_satisfied": n_satisfied,
            "n_grader_unparsed": sum(1 for v in verdicts.values() if v is None),
            "n_graded": sum(1 for v in verdicts.values() if v is not None),
        }
    return {"reference": [], "rollouts": [], "extra": extra}


def test_report_nests_the_three_published_rates():
    # Session 1: turn 1 all satisfied, turn 2 half. Session 2: both turns full.
    judgements = [
        _judged_session(
            1,
            [
                ({"1": True}, {"1": "格式约束"}, "align"),
                (
                    {"1": True, "2": False},
                    {"1": "格式约束", "2": "内容约束"},
                    "misalign",
                ),
            ],
        ),
        _judged_session(
            2,
            [
                ({"1": True}, {"1": "格式约束"}, "align"),
                (
                    {"1": True, "2": True},
                    {"1": "格式约束", "2": "内容约束"},
                    "misalign",
                ),
            ],
        ),
    ]
    m = _report(judgements)
    # CSR averages turns: (1 + .5 + 1 + 1) / 4.
    assert m["csr"] == pytest.approx(87.5)
    # ISR counts turns satisfying everything: 3 of 4.
    assert m["isr"] == pytest.approx(75.0)
    # SSR counts sessions whose turns ALL do: 1 of 2 -- the sharpest of the three.
    assert m["ssr"] == pytest.approx(50.0)
    assert m["n_turns"] == 4
    assert m["n_sessions"] == 2
    assert m["score_key"] == "csr"


def test_report_splits_aligned_from_misaligned_turns():
    # The gap between these is the part of the score that is really about
    # system-message priority.
    judgements = [
        _judged_session(
            1,
            [
                ({"1": True}, {"1": "格式约束"}, "align"),
                ({"1": False}, {"1": "内容约束"}, "misalign"),
            ],
        )
    ]
    m = _report(judgements)
    assert m["csr_align"] == pytest.approx(100.0)
    assert m["csr_misalign"] == pytest.approx(0.0)


def test_report_breaks_down_by_turn_position_and_constraint_type():
    judgements = [
        _judged_session(
            1,
            [
                ({"1": True}, {"1": "格式约束"}, "align"),
                ({"1": False, "2": True}, {"1": "格式约束", "2": "内容约束"}, "align"),
            ],
        )
    ]
    m = _report(judgements)
    assert m["turn_1_csr"] == pytest.approx(100.0)
    assert m["turn_2_csr"] == pytest.approx(50.0)
    assert m["turn_1_n_turns"] == 1
    # Pooled over constraints: 格式约束 is 1 of 2, 内容约束 1 of 1.
    assert m["csr_type_format"] == pytest.approx(50.0)
    assert m["csr_type_content"] == pytest.approx(100.0)


def test_a_position_only_counts_the_sessions_that_reached_it():
    # A session that stopped after turn 1 must not enter turn 2's denominator as
    # a zero -- that would read as a model that stopped following its system
    # prompt when it was really a failed request.
    judgements = [
        _judged_session(1, [({"1": True}, {"1": "格式约束"}, "align")]),
        _judged_session(
            2,
            [
                ({"1": False}, {"1": "格式约束"}, "align"),
                ({"1": False}, {"1": "格式约束"}, "align"),
            ],
        ),
    ]
    m = _report(judgements)
    assert m["turn_1_n_turns"] == 2
    assert m["turn_2_n_turns"] == 1
    assert m["turn_2_csr"] == pytest.approx(0.0)


def test_report_sizes_the_gap_an_ungradeable_turn_leaves_in_the_headline():
    # csr counts the unreadable turn as 0.0 (upstream's denominator) so it is a
    # floor; csr_graded and ungradeable_rate say how far.
    judgements = [
        _judged_session(
            1,
            [
                ({"1": True}, {"1": "格式约束"}, "align"),
                ({"1": None}, {"1": "格式约束"}, "align"),
            ],
        )
    ]
    m = _report(judgements)
    assert m["csr"] == pytest.approx(50.0)
    assert m["csr_graded"] == pytest.approx(100.0)
    assert m["ungradeable_rate"] == pytest.approx(50.0)
    assert m["n_grader_unparsed"] == 1
    assert m["n_grader_unparsed_turns"] == 1


def test_report_names_the_protocol_that_produced_the_numbers():
    judgements = [_judged_session(1, [({"1": True}, {"1": "格式约束"}, "align")])]
    assert _report(judgements)["history_mode"] == "model"

    mixed = judgements + [
        _judged_session(2, [({"1": True}, {"1": "格式约束"}, "align")])
    ]
    mixed[1]["extra"]["history_mode"] = "ground_truth"
    # Only a resumed run whose config changed under it can do this; surfacing it
    # beats averaging two different measurements together.
    assert _report(mixed)["history_mode"] == "mixed"


def test_report_over_no_finals_is_zero_rather_than_a_crash():
    m = _report([], fails=["boom"])
    assert m["csr"] == 0.0 and m["ssr"] == 0.0
    assert m["fails"] == 1


# --------------------------------------------------------------------------
# the scoring kernel
# --------------------------------------------------------------------------


def test_unresolved_constraints_stay_apart_from_refused_ones():
    # A silent 0.0 for an unreadable reply is indistinguishable from a reply
    # that satisfied nothing, which is what makes grader drift invisible.
    assert parse_verdict(_reply({1: "是", 2: "否"}), ["1", "2", "3"]) == {
        "1": True,
        "2": False,
        "3": None,
    }


def test_digits_in_the_judges_prose_are_not_read_as_verdicts():
    reply = (
        '\'\'\'json\n{\n  "评判理由": "约束 2: 用户问了价格",\n'
        '  "评判结果": {1: "是"}\n}\n\'\'\''
    )
    assert parse_verdict(reply, ["1", "2"]) == {"1": True, "2": None}


def test_the_judge_prompt_carries_the_system_prompt_criteria_and_both_turns():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
    ]
    prompt = build_judge_prompt(messages, _criteria((1, "必须用中文", "格式约束")))
    assert "SYS" in prompt
    assert "<round-1>" in prompt and "U1" in prompt and "A1" in prompt
    assert "U2" in prompt and "A2" in prompt
    assert "1. 必须用中文 | 格式约束" in prompt
    # One history round only -- the turn under judgement is not also a round.
    assert "<round-2>" not in prompt
