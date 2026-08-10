"""Unit tests for the ComplexConstraints 0-shot generative task.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from dataclasses import replace

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.datasets.complex_constraints import (
    ComplexConstraintsDataset,
    ComplexConstraintsDatasetSample,
)
from sieval.tasks.complex_constraints_0shot_gen import (
    ComplexConstraintsZeroShotGenTask,
)


class _ScriptedChatModel(ChatModel):
    """ChatModel returning a fixed reply, recording every prompt it was sent."""

    def __init__(self, reply: str, model: str = "mock"):
        super().__init__(model=model, api_key="fake")
        self._reply = reply
        self.prompts: list[object] = []
        self.last_kwargs: dict[str, object] = {}

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        self.prompts.append(prompt)
        self.last_kwargs = dict(kwargs)
        return ModelOutput(
            model=self.meta(), texts=[self._reply], finish_reasons=["stop"]
        )

    async def _alogprobs_impl(
        self, prompt, *, max_tokens=1, logprobs=5, echo=True, temperature=0.0, **kwargs
    ) -> ModelOutput:
        _ = (prompt, max_tokens, logprobs, echo, temperature, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _sample(n_criteria: int = 3) -> ComplexConstraintsDatasetSample:
    return {
        "benchmark_id": "CIF-001",
        "prompt": "Write a rota for the week.",
        "use_case": "Logistics, Scheduling & Event Planning",
        "instruction_type": "Negative",
        "prompt_style": "Context prompting",
        "criteria": [
            f"The response should satisfy rule {i}." for i in range(n_criteria)
        ],
    }


def _task(
    answer_reply: str = "here is the rota",
    grader_reply: str = "1: PASS\n2: PASS\n3: PASS",
    n: int = 1,
):
    sample = _sample()
    dataset = ComplexConstraintsDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(sample)])})
    )
    model = _ScriptedChatModel(reply=answer_reply, model="candidate")
    grader = _ScriptedChatModel(reply=grader_reply, model="gpt-5-mini")
    task = ComplexConstraintsZeroShotGenTask(dataset, model, grader=grader, n=n)
    return task, model, grader


# --- grader is mandatory; the rubric is natural language, no fallback exists ---


def test_build_grader_requires_config():
    with pytest.raises(ValueError, match="requires an LLM grader"):
        ComplexConstraintsZeroShotGenTask._build_grader(None)


def test_build_grader_accepts_mapping_and_model():
    built = ComplexConstraintsZeroShotGenTask._build_grader(
        {"model": "gpt-5-mini", "api_key": "fake"}
    )
    assert isinstance(built, ChatModel)
    existing = _ScriptedChatModel(reply="1: PASS")
    assert ComplexConstraintsZeroShotGenTask._build_grader(existing) is existing


# --- preprocess: the rubric is a procedure, so it goes to extra, not reference ---


@pytest.mark.anyio
async def test_preprocess_puts_rubric_in_extra_not_reference():
    task, _, _ = _task()
    sample = _sample()
    pre = await task.preprocess(sample, TaskContext(sample_id=0, raw_sample=sample))

    assert pre["prompt"] == [{"role": "user", "content": "Write a rota for the week."}]
    # A rubric is a procedure, not a value -- `reference` must stay absent so a
    # reader cannot mistake the criteria list for a gold answer.
    assert "reference" not in pre
    assert pre["extra"]["criteria"] == sample["criteria"]
    assert pre["extra"]["benchmark_id"] == "CIF-001"
    assert pre["extra"]["instruction_type"] == "Negative"


@pytest.mark.anyio
async def test_infer_forwards_n():
    task, model, _ = _task(n=3)
    await task.infer(
        {"prompt": [{"role": "user", "content": "q"}]}, TaskContext(sample_id=0)
    )
    assert model.last_kwargs.get("n") == 3


@pytest.mark.anyio
@pytest.mark.parametrize("blank", ["", "   ", "\n\n"])
async def test_postprocess_normalizes_blank_to_none(blank: str):
    task, model, _ = _task()
    out = ModelOutput(model=model.meta(), texts=[blank])
    post = await task.postprocess(out, TaskContext(sample_id=0))
    assert post["rollouts"][0]["extracted"] is False


# --- feedback: one judge call per rollout, verdicts index-aligned ---


@pytest.mark.anyio
async def test_feedback_all_criteria_pass_is_a_task_pass():
    task, _, grader = _task(grader_reply="1: PASS\n2: PASS\n3: PASS")
    sample = _sample()
    ctx = TaskContext(sample_id=0, raw_sample=sample)
    finalize, judgement = await task.feedback(build_prediction_record(["answer"]), ctx)

    assert finalize is True
    fb = judgement["rollouts"][0]
    assert fb["correct"] is True
    assert fb["metrics"]["task_pass"] is True
    assert fb["metrics"]["criterion_pass_rate"] == pytest.approx(1.0)
    assert fb["extra"]["criterion_verdicts"] == [True, True, True]
    assert fb["extra"]["n_satisfied"] == 3
    assert fb["extra"]["n_criteria"] == 3
    # One call, carrying the rubric and the response together.
    assert len(grader.prompts) == 1
    assert "1. The response should satisfy rule 0." in grader.prompts[0]
    assert "answer" in grader.prompts[0]


@pytest.mark.anyio
async def test_feedback_one_failed_criterion_sinks_the_task_pass():
    # The headline is all-or-nothing: 2/3 criteria is a task failure that still
    # carries 0.667 partial credit. Collapsing the two would lose the paper's
    # distinction between its two metrics.
    task, _, _ = _task(grader_reply="1: PASS\n2: FAIL\n3: PASS")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, judgement = await task.feedback(build_prediction_record(["answer"]), ctx)

    fb = judgement["rollouts"][0]
    assert fb["correct"] is False
    assert fb["metrics"]["task_pass"] is False
    assert fb["score"] == pytest.approx(2 / 3)
    assert fb["extra"]["criterion_verdicts"] == [True, False, True]


@pytest.mark.anyio
async def test_feedback_unparsed_verdict_scores_unsatisfied_but_is_counted():
    # An unreadable verdict must not inflate the score, and must stay
    # distinguishable from a criterion the judge actually failed.
    task, _, _ = _task(grader_reply="1: PASS\n3: PASS")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, judgement = await task.feedback(build_prediction_record(["answer"]), ctx)

    fb = judgement["rollouts"][0]
    assert fb["extra"]["criterion_verdicts"] == [True, None, True]
    assert fb["extra"]["n_satisfied"] == 2
    assert fb["extra"]["n_unparsed"] == 1
    assert fb["correct"] is False


@pytest.mark.anyio
async def test_feedback_persists_the_whole_judge_output():
    reply = "Let me check each one.\n\n1: PASS\n2: PASS\n3: PASS"
    task, _, _ = _task(grader_reply=reply)
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, judgement = await task.feedback(build_prediction_record(["answer"]), ctx)

    grader_output = judgement["rollouts"][0]["extra"]["grader_output"]
    # The whole ModelOutput, not hand-picked fields: the reply is the only
    # durable evidence of a verdict set a re-grade need not reproduce.
    assert grader_output["texts"] == [reply]
    assert grader_output["finish_reasons"] == ["stop"]
    assert grader_output["model"]["model"] == "gpt-5-mini"


@pytest.mark.anyio
async def test_feedback_empty_response_scores_zero_without_calling_the_judge():
    # A judge that would pass everything cannot rescue an empty response.
    task, _, grader = _task(grader_reply="1: PASS\n2: PASS\n3: PASS")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    post = build_prediction_record([None])
    _, judgement = await task.feedback(post, ctx)

    fb = judgement["rollouts"][0]
    assert fb["correct"] is False
    assert fb["extra"]["n_satisfied"] == 0
    assert fb["extra"]["n_criteria"] == 3
    assert grader.prompts == []
    # No call, so no grader output and no verdict list -- absence is the signal.
    assert "grader_output" not in fb["extra"]
    assert "criterion_verdicts" not in fb["extra"]


@pytest.mark.anyio
async def test_feedback_grades_each_rollout_and_records_sample_level_score():
    task, _, grader = _task(grader_reply="1: PASS\n2: FAIL\n3: PASS")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, judgement = await task.feedback(
        build_prediction_record(["first", "second"]), ctx
    )

    assert judgement["n_rollouts"] == 2
    assert judgement["n_correct"] == 0
    assert len(grader.prompts) == 2
    # Sample-level score is the mean criterion pass rate, genuine partial credit
    # rather than a mirror of n_correct/n_rollouts (which is 0 here).
    assert judgement["score"] == pytest.approx(2 / 3)
    # The rubric is a procedure, so there is no reference *value*: the builder
    # keeps the key None in memory and serialization drops it, leaving the row
    # without a gold answer a reader could mistake the criteria for.
    assert judgement["reference"] is None
    assert judgement["extra"]["n_criteria"] == 3
    assert judgement["extra"]["benchmark_id"] == "CIF-001"


@pytest.mark.anyio
async def test_feedback_short_circuit_does_not_inherit_a_prior_verdict():
    # With n>1 the rollouts share one loop; an ungraded attempt must not be
    # attributed the graded attempt's verdicts.
    task, _, _ = _task(grader_reply="1: PASS\n2: PASS\n3: PASS")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, judgement = await task.feedback(build_prediction_record(["answer", None]), ctx)

    graded, skipped = judgement["rollouts"]
    assert graded["extra"]["criterion_verdicts"] == [True, True, True]
    assert "criterion_verdicts" not in skipped["extra"]
    assert "grader_output" not in skipped["extra"]
    assert skipped["extra"]["n_satisfied"] == 0


# --- report: the three published rates over graded + failed samples ---


def _final(
    sample_id: int, satisfied: int, n_criteria: int, n_unparsed: int = 0
) -> TaskContext:
    return TaskContext(
        sample_id=sample_id,
        feedback_result=build_judgement_record(
            None,
            [
                build_rollout_judgement(
                    0,
                    satisfied == n_criteria,
                    score=satisfied / n_criteria,
                    metrics={
                        "task_pass": satisfied == n_criteria,
                        "criterion_pass_rate": satisfied / n_criteria,
                    },
                    extra={
                        "n_criteria": n_criteria,
                        "n_satisfied": satisfied,
                        "n_unparsed": n_unparsed,
                    },
                )
            ],
        ),
    )


@pytest.mark.anyio
async def test_report_headline_is_the_task_pass_rate():
    task, _, _ = _task()
    finals = [_final(0, 10, 10), _final(1, 9, 10), _final(2, 40, 40), _final(3, 0, 10)]
    report = await task.report(finals, fails=[])

    # 2 of 4 responses satisfied every criterion.
    assert report["task_pass_rate"] == pytest.approx(50.0)
    assert report["score"] == report["task_pass_rate"]
    assert report["n_graded"] == 4
    assert report["n_criteria_graded"] == 70
    assert report["n_unparsed"] == 0
    assert report["fails"] == 0


@pytest.mark.anyio
async def test_report_macro_and_micro_criterion_rates_both_reported():
    task, _, _ = _task()
    # 5/10 and 30/40: macro = mean(0.5, 0.75) = 0.625; micro = 35/50 = 0.70.
    finals = [_final(0, 5, 10), _final(1, 30, 40)]
    report = await task.report(finals, fails=[])

    assert report["criterion_pass_rate_macro"] == pytest.approx(62.5)
    assert report["criterion_pass_rate_micro"] == pytest.approx(70.0)


@pytest.mark.anyio
async def test_report_counts_fails_as_zero_criteria_satisfied():
    # Failed samples must dilute all three rates (full-set metric). Excluding
    # them would report 100% here.
    task, _, _ = _task()
    finals = [_final(0, 10, 10)]
    fails = [TaskContext(sample_id=1, raw_sample=_sample(n_criteria=10))]
    report = await task.report(finals, fails)

    assert report["task_pass_rate"] == pytest.approx(50.0)
    assert report["criterion_pass_rate_macro"] == pytest.approx(50.0)
    # The failed sample's rubric size is recovered from its raw sample, so it
    # reaches the pooled denominator too: 10 / 20.
    assert report["criterion_pass_rate_micro"] == pytest.approx(50.0)
    assert report["fails"] == 1


@pytest.mark.anyio
async def test_report_fails_weighted_by_n():
    task, _, _ = _task(n=2)
    finals = [_final(0, 10, 10)]
    fails = [TaskContext(sample_id=1, raw_sample=_sample(n_criteria=10))]
    report = await task.report(finals, fails)

    # 1 passing rollout + n*1 = 2 failed attempts => 3 units, 33.3%.
    # An unweighted count would give 50%.
    assert report["task_pass_rate"] == pytest.approx(100 / 3)


@pytest.mark.anyio
async def test_report_fail_without_raw_sample_still_counts_as_a_failure():
    # A context that never got a raw sample has no known rubric size; it must
    # still dilute the task-pass and macro rates rather than vanish.
    task, _, _ = _task()
    finals = [_final(0, 10, 10)]
    report = await task.report(finals, fails=[TaskContext(sample_id=1)])

    assert report["task_pass_rate"] == pytest.approx(50.0)
    assert report["criterion_pass_rate_macro"] == pytest.approx(50.0)
    # Unknown rubric size adds nothing to the pooled denominator.
    assert report["criterion_pass_rate_micro"] == pytest.approx(100.0)


@pytest.mark.anyio
async def test_report_surfaces_unparsed_verdicts():
    # Judge format drift must be visible in the report, not buried inside the
    # rates it silently depresses.
    task, _, _ = _task()
    report = await task.report([_final(0, 8, 10, n_unparsed=2)], fails=[])
    assert report["n_unparsed"] == 2
    # The two unreadable verdicts are already inside the 8/10 -- surfacing them
    # separately is what lets a reader tell drift from a real rubric failure.
    assert report["criterion_pass_rate_micro"] == pytest.approx(80.0)


@pytest.mark.anyio
async def test_report_empty_is_zero():
    task, _, _ = _task()
    report = await task.report([], fails=[])
    assert report["score"] == 0.0
    assert report["n_graded"] == 0


@pytest.mark.anyio
async def test_report_separates_an_empty_response_from_a_failed_rubric():
    # Two rollouts scoring 0/3, reached two different ways: one the judge graded
    # and failed, one that never had a response to grade. `n_graded` counts both,
    # so without `n_unextracted` the report cannot tell them apart -- and "the
    # model returned nothing" and "the model failed every criterion" want
    # different responses from whoever reads the run.
    task, _, _ = _task()
    graded_and_failed = _final(0, 0, 3)
    empty_response = replace(
        _final(1, 0, 3), postprocess_result=build_prediction_record([None])
    )
    report = await task.report([graded_and_failed, empty_response], fails=[])

    assert report["task_pass_rate"] == pytest.approx(0.0)
    assert report["n_graded"] == 2
    assert report["n_unextracted"] == 1
