"""Unit tests for the SysBench task.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import asyncio
import json

import pytest

from sieval.community.sysbench import (
    aggregate_metrics,
    build_judge_prompt,
    parse_verdict,
)
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
    # CSR pools over CONSTRAINTS: 5 satisfied of 6, not the mean of the per-turn
    # rates. The fixture's turns carry 1 and 2 constraints precisely so the two
    # disagree -- averaging turns gives (1 + .5 + 1 + 1) / 4 = 87.5, which is
    # `csr_macro` and is NOT what the paper's tables report.
    assert m["csr"] == pytest.approx(5 / 6 * 100)
    assert m["csr_macro"] == pytest.approx(87.5)
    assert m["n_criteria_graded"] == 6
    # ISR counts turns satisfying everything: 3 of 4. Its unit really is the turn.
    assert m["isr"] == pytest.approx(75.0)
    # SSR averages over PREFIX lengths, not over whole sessions (paper SS3.3):
    # session 1 keeps turn 1 and loses turn 2 -> 1/2; session 2 keeps both -> 2/2.
    # Mean = 75.0. Counting "sessions whose turns all do" would report 50.0, which
    # is only the alpha=2 term.
    assert m["ssr"] == pytest.approx(75.0)
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


def test_report_splits_alignment_by_isr_because_that_is_the_published_column():
    """Table 3 splits ISR, not CSR -- and the two disagree on a partial turn.

    Upstream appends the all-satisfied indicator (``是否可用``) to its align
    buckets, so a turn that met half its constraints contributes 0 there and 0.5
    to CSR. Reporting only ``csr_*`` gives a number that cannot be read against
    the paper's aligned/misaligned table.
    """
    half = ({"1": True, "2": False}, {"1": "格式约束", "2": "内容约束"}, "align")
    m = _report([_judged_session(1, [half])])
    assert m["csr_align"] == pytest.approx(50.0)
    assert m["isr_align"] == pytest.approx(0.0)


def test_a_walk_that_died_mid_session_cannot_report_a_perfect_session():
    """The turns it never reached shorten the prefix, they do not vanish.

    Scoring SSR over the turns merely *present* would let an infrastructure
    failure at turn 3 of 5 read as a model that followed its system prompt all
    the way through -- the exact inversion of what ending the walk is for.
    """
    perfect = ({"1": True}, {"1": "格式约束"}, "align")
    judgement = _judged_session(1, [perfect, perfect])
    judgement["extra"]["n_turns"] = 5  # declared 5, only 2 were reached
    m = _report([judgement])
    assert m["ssr"] == pytest.approx(40.0)  # 2 leading turns of 5, not 100.0
    # CSR and ISR keep the judged denominator: those two are means, so an absent
    # turn is genuinely absent rather than a zero.
    assert m["csr"] == pytest.approx(100.0)
    assert m["isr"] == pytest.approx(100.0)


def test_ssr_stops_at_the_first_failure_not_at_the_last():
    """It is a *leading* run: a later recovery cannot lengthen the prefix."""
    ok = ({"1": True}, {"1": "格式约束"}, "align")
    bad = ({"1": False}, {"1": "格式约束"}, "align")
    # fail, pass, pass, pass -> prefix 0, even though 3 of 4 turns are perfect.
    m = _report([_judged_session(1, [bad, ok, ok, ok])])
    assert m["ssr"] == pytest.approx(0.0)
    assert m["isr"] == pytest.approx(75.0)


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


def test_a_grader_that_restates_the_asked_format_first_is_still_read():
    """The prompt quotes the output format, so an echo of it is a decoy block.

    Its placeholder verdicts (``1: "……"``) parse as nothing, so anchoring on the
    FIRST `评判结果` would score the whole turn unresolved -- a grader-shaped
    zero that looks like a model failure. The real answer is the last block.
    """
    criteria = _criteria((1, "必须用中文回答", "格式约束"))
    decoy = build_judge_prompt(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ],
        criteria,
    )
    assert parse_verdict(decoy, ["1"]) == {"1": None}, "the prompt must not self-parse"
    assert parse_verdict(decoy + "\n" + _reply({1: "是"}), ["1"]) == {"1": True}


def test_a_grader_that_restates_the_asked_format_last_is_still_read():
    """The mirror of the decoy-first case: answer, then echo the format.

    Anchoring on the LAST block unconditionally reads the trailing echo, whose
    placeholder verdicts parse as nothing, and the whole turn goes unresolved --
    the same grader-shaped zero the first-block anchor produces, just for the
    opposite reply shape. Only skipping blocks that resolve nothing handles both.
    """
    criteria = _criteria((1, "必须用中文回答", "格式约束"))
    echo = build_judge_prompt(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ],
        criteria,
    )
    assert parse_verdict(_reply({1: "是"}) + "\n" + echo, ["1"]) == {"1": True}


def test_csr_pools_over_constraints_while_isr_averages_over_turns():
    """The two denominators, on turns of unequal constraint count.

    Upstream's published CSR comes from ``plot/tab6_csr_full.py``, which divides
    遵循数量 by 约束总量 -- one constraint, one vote. Averaging the per-turn rates
    instead over-weights a 1-constraint turn against an 11-constraint one; on
    published runs the two are up to ~1.9 points apart, which is enough to swap
    adjacent rows of Table 2. ISR keeps the turn as its unit, so it is unaffected.
    """
    # Two turns: 1 of 1, then 1 of 5. Pooled 2/6 = 33.3; averaged (1 + .2)/2 = 60.0.
    turns = [(1, 1, 1, 1), (1, 2, 1, 5)]
    m = aggregate_metrics(turns, {1: 2})
    assert m["csr"] == pytest.approx(2 / 6)
    assert m["csr_macro"] == pytest.approx(0.6)
    assert m["n_criteria"] == 6
    # ISR is a turn-level fact and the same either way: turn 1 met all 1 of its
    # constraints, turn 2 met 1 of 5. One turn of two, whatever the widths are.
    assert m["isr"] == pytest.approx(0.5)


def test_csr_and_the_constraint_type_breakdown_come_from_one_total():
    """They are the same table in upstream -- ``tab6_csr_full`` reads both columns.

    So the type cells must pool to the headline. This is the invariant that a
    turn-averaged headline breaks: it would report 87.5 here while its own type
    breakdown pooled to 83.3, with nothing in the report saying which is CSR.
    """
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
        )
    ]
    m = _report(judgements)
    satisfied = m["csr_type_format"] / 100 * 2 + m["csr_type_content"] / 100 * 1
    assert m["csr"] == pytest.approx(satisfied / 3 * 100)


def test_the_cumulative_turn_series_is_table_4s_r_columns_and_averages_to_ssr():
    """``turn_{t}_isr`` is position *t* alone; ``_isr_cumulative`` is Table 4's R_t.

    Published GPT-4o R1..R5 is 82.8/66.2/51.2/40.4/31.6 -- a decay the
    per-position series does not show, so reading one against the other's
    published column is a 40-point error rather than a rounding one.
    """
    ok = ({"1": True}, {"1": "格式约束"}, "align")
    bad = ({"1": False}, {"1": "格式约束"}, "align")
    # Session 1 fails at turn 2 then recovers; session 2 is perfect throughout.
    m = _report(
        [
            _judged_session(1, [ok, bad, ok]),
            _judged_session(2, [ok, ok, ok]),
        ]
    )
    # Position 3 alone: both sessions satisfied it.
    assert m["turn_3_isr"] == pytest.approx(100.0)
    # Cumulative: only session 2 has its first three turns all satisfied.
    assert m["turn_1_isr_cumulative"] == pytest.approx(100.0)
    assert m["turn_2_isr_cumulative"] == pytest.approx(50.0)
    assert m["turn_3_isr_cumulative"] == pytest.approx(50.0)
    # Upstream reaches SSR by averaging exactly this series (`plot/tab4_turn.py`
    # sums the same indicator it prints as R_t), so the two cannot drift apart.
    assert m["ssr"] == pytest.approx(
        sum(m[f"turn_{t}_isr_cumulative"] for t in (1, 2, 3)) / 3
    )


def test_every_breakdown_rate_reports_the_count_it_was_divided_by():
    """A breakdown cell without its denominator is a number, not a measurement.

    ``csr_misalign`` over 4 constraints and over 400 are the same value and a
    different claim, and this is the check that would have caught the report
    emitting one of the four families' denominators and none of the rest.
    """
    m = _report(
        [
            _judged_session(
                1,
                [
                    ({"1": True}, {"1": "格式约束"}, "align"),
                    (
                        {"1": False, "2": True},
                        {"1": "格式约束", "2": "内容约束"},
                        "misalign",
                    ),
                ],
            )
        ]
    )
    # Every rate key maps to the key naming its own denominator. Written as a
    # table rather than a loop over `m`, because the point is that a NEW rate
    # added without one has to appear here to be believed.
    for rate, denominator in (
        ("csr_align", "csr_align_n_criteria"),
        ("isr_align", "isr_align_n_turns"),
        ("csr_misalign", "csr_misalign_n_criteria"),
        ("isr_misalign", "isr_misalign_n_turns"),
        ("turn_1_csr", "turn_1_n_criteria"),
        ("turn_1_isr", "turn_1_n_turns"),
        ("turn_2_csr", "turn_2_n_criteria"),
        ("turn_1_isr_cumulative", "turn_1_isr_cumulative_n_sessions"),
        ("csr_type_format", "csr_type_format_n_criteria"),
        ("csr_type_content", "csr_type_content_n_criteria"),
    ):
        assert rate in m, rate
        assert denominator in m, f"{rate} has no denominator key {denominator}"
        assert m[denominator] > 0, denominator
    # And the three denominators are genuinely different counts, which is why one
    # of them cannot stand in for the others: turn 2 holds 2 constraints in 1 turn.
    assert m["turn_2_n_criteria"] == 2
    assert m["turn_2_n_turns"] == 1
    assert m["turn_2_isr_cumulative_n_sessions"] == 1


def test_the_type_cells_pool_to_the_headline_using_only_reported_numbers():
    """The same invariant as ``..._come_from_one_total``, weighted from the report.

    That test has to hardcode the per-category constraint counts, so it passes on
    a report a consumer cannot check. With the denominators emitted, the pooling
    is a reduction over the report alone -- and it fails if a cell goes missing.
    """
    m = _report(
        [
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
            )
        ]
    )
    cells = [
        k for k in m if k.startswith("csr_type_") and not k.endswith("_n_criteria")
    ]
    satisfied = sum(m[k] / 100 * m[f"{k}_n_criteria"] for k in cells)
    total = sum(m[f"{k}_n_criteria"] for k in cells) + m["n_criteria_untyped"]
    assert total == m["n_criteria_graded"]
    assert m["csr"] == pytest.approx(satisfied / total * 100)


def test_a_constraint_upstream_left_untyped_is_counted_not_dropped():
    """It is in ``csr``, so it must be visible in the breakdown's arithmetic.

    Silently skipping it made the type cells pool to less than the headline with
    nothing saying so: 2 satisfied of 4 reads as ``csr`` 50.0 beside a lone
    ``csr_type_format`` of 100.0, which is the shape of a model that aced the only
    category measured rather than one that failed half the constraints.
    """
    m = _report(
        [
            _judged_session(
                1,
                [
                    (
                        {"1": True, "2": True, "3": False, "4": False},
                        {"1": "格式约束", "2": "格式约束", "3": "", "4": ""},
                        "align",
                    )
                ],
            )
        ]
    )
    assert m["csr"] == pytest.approx(50.0)
    assert m["csr_type_format"] == pytest.approx(100.0)
    assert m["csr_type_format_n_criteria"] == 2
    # No invented category for them -- upstream's six stay upstream's six.
    assert not [k for k in m if k.startswith("csr_type_") and "untyped" in k]
    # The residual is what makes the 50.0 and the 100.0 reconcilable.
    assert m["n_criteria_untyped"] == 2
    assert (
        m["csr_type_format_n_criteria"] + m["n_criteria_untyped"]
        == m["n_criteria_graded"]
    )


def test_the_untyped_count_is_reported_as_zero_rather_than_omitted():
    # A key present only in the bad case gives a reader no baseline, and absence
    # then has to be read as either "clean" or "old build".
    m = _report([_judged_session(1, [({"1": True}, {"1": "格式约束"}, "align")])])
    assert m["n_criteria_untyped"] == 0
    assert m["n_turns_unaligned"] == 0
    assert m["n_criteria_unaligned"] == 0


def test_a_constraint_typed_null_is_counted_untyped_rather_than_killing_the_report():
    """An explicit null type must not reach the bucket key.

    `criteria` is stored raw by the loader, so ``criteria_type: null`` keys a
    bucket under None and the sort over type buckets raises `TypeError` -- losing
    the whole report rather than one constraint.
    """
    m = _report(
        [
            _judged_session(
                1,
                [
                    (
                        {"1": True, "2": False},
                        {"1": "格式约束", "2": None},
                        "align",
                    )
                ],
            )
        ]
    )
    assert m["csr"] == pytest.approx(50.0)
    assert m["csr_type_format_n_criteria"] == 1
    assert m["n_criteria_untyped"] == 1
    # No bucket keyed by the null itself, under any spelling.
    assert not [k for k in m if k.startswith("csr_type_") and "None" in k]
    assert (
        m["csr_type_format_n_criteria"] + m["n_criteria_untyped"]
        == m["n_criteria_graded"]
    )


def test_a_turn_with_no_alignment_label_is_counted_not_dropped():
    """It is in ``csr``, ``isr`` and ``n_turns`` and in neither alignment cell.

    The unlabelled turn carries 3 constraints against the labelled turn's 1, which
    is what gives this test teeth: with one apiece both residuals are 1, so
    reporting either count for both would pass.
    """
    m = _report(
        [
            _judged_session(
                1,
                [
                    ({"1": True}, {"1": "格式约束"}, "align"),
                    (
                        {"1": False, "2": False, "3": False},
                        {"1": "格式约束", "2": "格式约束", "3": "格式约束"},
                        "",
                    ),
                ],
            )
        ]
    )
    assert m["n_turns"] == 2
    assert m["csr"] == pytest.approx(25.0)
    assert "csr_" not in m
    assert m["n_turns_unaligned"] == 1
    assert m["n_criteria_unaligned"] == 3
    # Both alignment axes close to the headline using only reported numbers.
    assert m["isr_align_n_turns"] + m["n_turns_unaligned"] == m["n_turns"]
    assert (
        m["csr_align_n_criteria"] + m["n_criteria_unaligned"] == m["n_criteria_graded"]
    )


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
