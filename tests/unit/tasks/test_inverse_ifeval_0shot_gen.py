"""Unit tests for the Inverse IFEval 0-shot generative task.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from unittest.mock import patch

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.datasets.inverse_ifeval as dataset_module
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import DENOMINATOR_FIELD, DENOMINATOR_REQUESTED
from sieval.datasets.inverse_ifeval import InverseIFEvalDataset
from sieval.tasks.inverse_ifeval_0shot_gen import InverseIFEvalZeroShotGenTask

# The two shipped template shapes: one with a `{prompt}` slot, one without. Five
# of the eight instruction types use the second, whose judge never sees the
# question — upstream's design, and the reason `.format` must tolerate an unused
# keyword.
_TEMPLATE_WITH_PROMPT = (
    "题目：{prompt}\n参考答案：{response_reference}\n模型回复：{response}"
)
_TEMPLATE_WITHOUT_PROMPT = (
    "<标准答案>：\n{response_reference}\n\n<学生答案>：\n{response}"
)
_SYSTEM_PROMPT = "从现在开始你的角色是一名严谨的指令遵循判卷老师。"
_PASS_REPLY = (
    '【评分依据】：符合。\n【评分】：1分\n【JSON】：\n```\n{"answer_score": 1}\n```'
)
_FAIL_REPLY = (
    '【评分依据】：不符合。\n【评分】：0分\n【JSON】：\n```\n{"answer_score": 0}\n```'
)


class _ScriptedChatModel(ChatModel):
    """ChatModel returning a fixed reply, recording what it was called with."""

    def __init__(self, reply: str, model: str = "mock"):
        super().__init__(model=model, api_key="fake")
        self._reply = reply
        self.last_prompt: object = None
        self.last_kwargs: dict[str, object] = {}
        self.calls = 0

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        self.calls += 1
        self.last_prompt = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(
            model=self.meta(),
            texts=[self._reply] * int(kwargs.get("n", 1)),
            finish_reasons=["stop"] * int(kwargs.get("n", 1)),
        )

    async def _alogprobs_impl(
        self, prompt, *, max_tokens=1, logprobs=5, echo=True, temperature=0.0, **kwargs
    ) -> ModelOutput:
        _ = (prompt, max_tokens, logprobs, echo, temperature, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _row(
    *,
    language: str = "english",
    instruction_type: str = "Question Correction",
    template: str = _TEMPLATE_WITH_PROMPT,
    prompt: str = "Answer without any reasoning.",
    reference: str = "The answer must omit all reasoning.",
) -> dict:
    return {
        "instruction_types": instruction_type,
        "prompt": prompt,
        "response_reference": reference,
        "language": language,
        "judge_prompt_template": template,
        "judge_system_prompt": _SYSTEM_PROMPT,
    }


def _dataset(rows: list[dict], *, language: str | None = None) -> InverseIFEvalDataset:
    """Build through the real ``load`` path so ``language`` is recorded on it."""
    hf = HFDatasetDict({"train": HFDataset.from_list(rows)})
    with patch.object(dataset_module, "load_dataset", return_value=hf):
        return InverseIFEvalDataset("m-a-p/Inverse_IFEval", language=language)


def _task(
    grader_reply: str = _PASS_REPLY,
    *,
    rows: list[dict] | None = None,
    language: str | None = None,
    n: int = 1,
    k: int = 1,
):
    dataset = _dataset(rows if rows is not None else [_row()], language=language)
    model = _ScriptedChatModel(reply="4", model="candidate")
    grader = _ScriptedChatModel(reply=grader_reply, model="judge-1")
    task = InverseIFEvalZeroShotGenTask(dataset, model, grader=grader, n=n, k=k)
    return task, model, grader


# --- grader is mandatory; no deterministic fallback ---


def test_build_grader_requires_config():
    with pytest.raises(ValueError, match="requires an LLM judge"):
        InverseIFEvalZeroShotGenTask._build_grader(None)


def test_build_grader_accepts_mapping_and_model():
    built = InverseIFEvalZeroShotGenTask._build_grader(
        {"model": "gpt-4.1", "api_key": "fake"}
    )
    assert isinstance(built, ChatModel)
    existing = _ScriptedChatModel(reply="x")
    assert InverseIFEvalZeroShotGenTask._build_grader(existing) is existing


def test_k_greater_than_n_rejected():
    dataset = _dataset([_row()])
    grader = _ScriptedChatModel(reply=_PASS_REPLY)
    with pytest.raises(ValueError, match="k must be <= n"):
        InverseIFEvalZeroShotGenTask(
            dataset, _ScriptedChatModel(reply="4"), grader=grader, n=1, k=2
        )


# --- preprocess: user turn only, no invented system prompt ---


@pytest.mark.anyio
async def test_preprocess_sends_only_the_user_turn():
    task, _, _ = _task()
    pre = await task.preprocess(_row(), TaskContext(sample_id=0))
    # A system prompt of our own would change what the counter-intuitive
    # instruction competes against, so there must not be one.
    assert pre["prompt"] == [
        {"role": "user", "content": "Answer without any reasoning."}
    ]
    assert pre["reference"] == "The answer must omit all reasoning."
    assert pre["extra"]["language"] == "english"
    assert pre["extra"]["instruction_types"] == "Question Correction"


@pytest.mark.anyio
async def test_infer_forwards_only_n():
    task, model, _ = _task(n=3)
    await task.infer(
        {"prompt": [{"role": "user", "content": "q"}]}, TaskContext(sample_id=0)
    )
    # Decoding is model-layer; the task passes n and nothing else.
    assert model.last_kwargs == {"n": 3}


@pytest.mark.anyio
async def test_postprocess_blanks_normalize_to_none():
    task, model, _ = _task()
    inferred = ModelOutput(model=model.meta(), texts=["ok", "  ", ""])
    post = await task.postprocess(inferred, TaskContext(sample_id=0))
    assert [r.get("prediction") for r in post["rollouts"]] == ["ok", None, None]


# --- feedback: the sample's OWN judge prompt, rendered verbatim ---


@pytest.mark.anyio
async def test_judge_receives_the_sample_system_prompt_and_rendered_template():
    task, _, grader = _task()
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    await task.feedback(build_prediction_record(["42"]), ctx)

    assert grader.last_prompt == [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "题目：Answer without any reasoning.\n"
                "参考答案：The answer must omit all reasoning.\n"
                "模型回复：42"
            ),
        },
    ]


@pytest.mark.anyio
async def test_template_without_a_prompt_slot_omits_the_question():
    # Five of eight types grade against the reference alone; passing `prompt=` to
    # `.format` must not raise, and must not smuggle the question in.
    task, _, grader = _task(rows=[_row(template=_TEMPLATE_WITHOUT_PROMPT)])
    ctx = TaskContext(sample_id=0, raw_sample=_row(template=_TEMPLATE_WITHOUT_PROMPT))
    await task.feedback(build_prediction_record(["42"]), ctx)

    user_turn = grader.last_prompt[1]["content"]
    assert user_turn == (
        "<标准答案>：\nThe answer must omit all reasoning.\n\n<学生答案>：\n42"
    )
    assert "Answer without any reasoning." not in user_turn


@pytest.mark.anyio
async def test_braces_in_the_content_survive_rendering():
    # A third of the code samples carry `{`; `str.format` must not re-interpret
    # braces that arrive inside the substituted values.
    raw = _row(
        prompt="Explain: SELECT {a} FROM t",
        reference="Must not add comments; keep {} intact.",
    )
    task, _, grader = _task(rows=[raw])
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    await task.feedback(build_prediction_record(['{"x": {"y": 1}}']), ctx)

    user_turn = grader.last_prompt[1]["content"]
    assert "SELECT {a} FROM t" in user_turn
    assert "keep {} intact." in user_turn
    assert '{"x": {"y": 1}}' in user_turn


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("reply", "expected_score", "expected_correct"),
    [(_PASS_REPLY, 1, True), (_FAIL_REPLY, 0, False)],
)
async def test_feedback_records_the_verdict(
    reply: str, expected_score: int, expected_correct: bool
):
    task, _, _ = _task(grader_reply=reply)
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    finalize, judgement = await task.feedback(build_prediction_record(["42"]), ctx)

    assert finalize is True
    verdict = judgement["rollouts"][0]
    assert verdict["correct"] is expected_correct
    assert verdict["extra"]["answer_score"] == expected_score
    assert verdict["extra"]["judge_parsed"] is True
    assert judgement["reference"] == "The answer must omit all reasoning."
    # Language and type travel on the judgement so report() need not reread
    # raw_sample.
    assert judgement["extra"]["language"] == "english"
    assert judgement["extra"]["instruction_types"] == "Question Correction"
    # The judge's WHOLE ModelOutput, not hand-picked fields.
    grader_output = verdict["extra"]["grader_output"]
    assert grader_output["texts"] == [reply]
    assert grader_output["model"]["model"] == "judge-1"
    assert grader_output["finish_reasons"] == ["stop"]


@pytest.mark.anyio
async def test_empty_judge_reply_is_a_failure_not_a_pass():
    """The self-parse hazard, at the task level.

    Every shipped ``judge_system_prompt`` ends with a worked example scoring 1,
    so a parser pointed at the prompt — or a fallback that reached for it when the
    reply was empty — would return PASS and inflate the score invisibly. An empty
    reply must stay unparsed and incorrect.
    """
    task, _, _ = _task(grader_reply="")
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    _, judgement = await task.feedback(build_prediction_record(["42"]), ctx)

    verdict = judgement["rollouts"][0]
    assert verdict["correct"] is False
    assert verdict["extra"]["answer_score"] is None
    assert verdict["extra"]["judge_parsed"] is False


@pytest.mark.anyio
async def test_off_rubric_judge_score_is_flagged_and_kept():
    task, _, _ = _task(grader_reply='【JSON】：\n```\n{"answer_score": 85}\n```')
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    _, judgement = await task.feedback(build_prediction_record(["42"]), ctx)

    verdict = judgement["rollouts"][0]
    assert verdict["correct"] is False
    assert verdict["extra"]["judge_parsed"] is False
    # The token behind the violation, so a 0-100 judge is diagnosable.
    assert verdict["extra"]["answer_score_raw"] == "85"


@pytest.mark.anyio
async def test_every_rollout_is_graded_separately():
    task, _, grader = _task(n=3)
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    _, judgement = await task.feedback(build_prediction_record(["a", "b", "c"]), ctx)
    assert grader.calls == 3
    assert [v["index"] for v in judgement["rollouts"]] == [0, 1, 2]


# --- report ---


def _finals(verdicts: list[tuple[str, str, list[bool]]]) -> list[TaskContext]:
    """One context per sample: (language, instruction_type, per-rollout correct)."""
    return [
        TaskContext(
            sample_id=i,
            feedback_result=build_judgement_record(
                "",
                [
                    build_rollout_judgement(
                        j, c, extra={"judge_parsed": True, "answer_score": int(c)}
                    )
                    for j, c in enumerate(correct)
                ],
            )
            | {"extra": {"language": language, "instruction_types": itype}},
        )
        for i, (language, itype, correct) in enumerate(verdicts)
    ]


@pytest.mark.anyio
async def test_report_score_is_the_pooled_mean_with_fails_in_the_denominator():
    task, _, _ = _task()
    finals = _finals(
        [
            ("english", "Question Correction", [True]),
            ("english", "Question Correction", [False]),
            ("chinese", "Code without Comments", [True]),
        ]
    )
    report = await task.report(finals, [TaskContext(sample_id=99)])

    # 2 correct over (3 finals + 1 fail) = 50.0. Scoring only the graded three
    # would give 66.67, so this discriminates.
    assert report["score"] == pytest.approx(50.0)
    assert report["pass@1"] == report["score"]
    assert report["fails"] == 1
    assert report["n_graded"] == 3
    assert report[DENOMINATOR_FIELD] == DENOMINATOR_REQUESTED
    assert report["score_key"] == "pass@1"


@pytest.mark.anyio
async def test_report_breaks_down_by_language_and_type():
    task, _, _ = _task()
    finals = _finals(
        [
            ("english", "Question Correction", [True]),
            ("english", "Code without Comments", [False]),
            ("chinese", "Question Correction", [True]),
            ("chinese", "Question Correction", [True]),
        ]
    )
    report = await task.report(finals, [])

    assert report["score_english"] == pytest.approx(50.0)
    assert report["score_chinese"] == pytest.approx(100.0)
    assert report["n_english"] == 2.0
    assert report["score_question_correction"] == pytest.approx(100.0)
    assert report["score_code_without_comments"] == pytest.approx(0.0)
    # Pooled overall, not the macro-average over the two types (which is 50.0).
    assert report["score"] == pytest.approx(75.0)


@pytest.mark.anyio
async def test_report_counts_unparsed_judge_replies():
    task, _, _ = _task()
    finals = [
        TaskContext(
            sample_id=0,
            feedback_result=build_judgement_record(
                "",
                [build_rollout_judgement(0, False, extra={"judge_parsed": False})],
            )
            | {
                "extra": {
                    "language": "english",
                    "instruction_types": "Question Correction",
                }
            },
        )
    ]
    report = await task.report(finals, [])
    # Still scored (0) and still in the denominator — this is judge health, not a
    # missing sample.
    assert report["judge_unparsed"] == 1
    assert report["n_graded"] == 1
    assert report["score"] == pytest.approx(0.0)


@pytest.mark.anyio
async def test_report_echoes_the_language_subset():
    task, _, _ = _task(rows=[_row(language="chinese")], language="chinese")
    report = await task.report(
        _finals([("chinese", "Question Correction", [True])]), []
    )
    assert report["language"] == "chinese"

    both, _, _ = _task()
    assert (
        await both.report(_finals([("english", "Question Correction", [True])]), [])
    )["language"] == "both"


@pytest.mark.anyio
async def test_report_omits_sampling_block_at_n_1_and_adds_it_above():
    single, _, _ = _task(n=1)
    report = await single.report(
        _finals([("english", "Question Correction", [True])]), []
    )
    # At n=1 the rest of the block only restates pass@1.
    assert "avg@n" not in report

    multi, _, _ = _task(n=2, k=2)
    report = await multi.report(
        _finals([("english", "Question Correction", [True, False])]), []
    )
    # The shared block reports percentages, the same scale as `score`.
    assert report["avg@n"] == pytest.approx(50.0)
    assert report["n"] == 2
    assert report["k"] == 2
    # No majority vote over free-form prose.
    assert "maj@k" not in report


@pytest.mark.anyio
async def test_report_of_an_all_failed_run_still_has_the_key_set():
    task, _, _ = _task()
    report = await task.report([], [TaskContext(sample_id=0)])
    assert report["score"] == pytest.approx(0.0)
    assert report["fails"] == 1
    assert report["n_graded"] == 0
