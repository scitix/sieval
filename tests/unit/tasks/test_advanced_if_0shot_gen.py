"""Unit tests for the AdvancedIF 0-shot generative task.

The upstream judge prompts are CC-BY-NC-4.0 and are never redistributed, so
``compose_judge_prompt`` is stubbed here; :mod:`tests.unit.community.test_advanced_if`
covers the real assembly against stand-in templates.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import json

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community import advanced_if as community_advanced_if
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    GRADER_OUTPUT_KEY,
    TaskContext,
    build_judgement_record,
    build_rollout_judgement,
    iter_grader_outputs,
)
from sieval.datasets.advanced_if import AdvancedIFDataset, AdvancedIFDatasetSample
from sieval.tasks import advanced_if_0shot_gen
from sieval.tasks.advanced_if_0shot_gen import AdvancedIFZeroShotGenTask

COMPLEX = "complex_if_single_turn_v5"
STEERABILITY = "system_steerability_v2"


class _ScriptedChatModel(ChatModel):
    """ChatModel returning a fixed reply, recording the last prompt it saw."""

    def __init__(self, reply: str, model: str = "mock"):
        super().__init__(model=model, api_key="fake")
        self._reply = reply
        self.last_prompt = None

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        self.last_prompt = prompt
        return ModelOutput(
            model=self.meta(),
            texts=[self._reply],
            usage={"input_tokens": 40, "output_tokens": 3, "total_tokens": 43},
        )

    async def _alogprobs_impl(
        self, prompt, *, max_tokens=1, logprobs=5, echo=True, temperature=0.0, **kwargs
    ) -> ModelOutput:
        _ = (prompt, max_tokens, logprobs, echo, temperature, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _sample(
    benchmark_name: str = COMPLEX,
    rubrics: tuple[str, ...] = ("Is it two paragraphs?", "Are there two metaphors?"),
) -> AdvancedIFDatasetSample:
    return {
        "conversation_history": json.dumps(
            [
                {"role": "user", "content": "Write a story."},
                {"role": "assistant", "content": "Once upon a time."},
                {"role": "user", "content": "Now make it rhyme."},
            ]
        ),
        "benchmark_name": benchmark_name,
        "prompt_metadata": json.dumps({"rubrics": list(rubrics)}),
    }


def _judge_reply(checks: dict, satisfied: str) -> str:
    return json.dumps(
        {"rubrics_check": checks, "SATISFIED_ALL_REQUIREMENTS": satisfied}
    )


def _task(answer: str = "A rhyming story.", grader_reply: str = "{}"):
    sample = _sample()
    dataset = AdvancedIFDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(sample)])})
    )
    model = _ScriptedChatModel(reply=answer, model="candidate")
    grader = _ScriptedChatModel(reply=grader_reply, model="o3-mini")
    return AdvancedIFZeroShotGenTask(dataset, model, grader=grader), grader


async def _run_to_feedback(task, sample):
    """Drive preprocess -> infer -> postprocess -> feedback for one sample."""
    ctx = TaskContext(sample_id=0, raw_sample=sample)
    pre = await task.preprocess(sample, ctx)
    ctx = TaskContext(sample_id=0, raw_sample=sample, preprocess_result=pre)
    inferred = await task.infer(pre, ctx)
    post = await task.postprocess(inferred, ctx)
    return await task.feedback(post, ctx)


@pytest.fixture(autouse=True)
def stub_upstream_prompts(monkeypatch):
    """Stand in for the non-redistributable judge prompts and their assembly.

    ``load_judge_prompts`` is stubbed as well as the assembly because
    construction validates the checkout, which no test machine has staged.
    """
    monkeypatch.setattr(
        advanced_if_0shot_gen,
        "compose_judge_prompt",
        lambda benchmark_name, messages, response, rubrics: (
            f"JUDGE[{benchmark_name}] resp={response} rubrics={len(rubrics)}"
        ),
    )
    monkeypatch.setattr(advanced_if_0shot_gen, "load_judge_prompts", lambda: None)


# --- grader is mandatory; rubric grading is the only scorer AdvancedIF has ---


def test_build_grader_requires_config():
    with pytest.raises(ValueError, match="requires an LLM grader"):
        AdvancedIFZeroShotGenTask._build_grader(None)


def test_construction_fails_without_the_upstream_checkout(monkeypatch):
    """The prompts are validated at construction, not at the first grade.

    Left to ``feedback``, a missing or drifted checkout costs a whole generation
    pass to discover: every grade raises, every sample fails, and the run still
    reports 0.0 -- a floor that reads as a score. The real loader runs here, so
    the check cannot be satisfied by a stub that never touches the environment.
    """
    monkeypatch.setattr(
        advanced_if_0shot_gen,
        "load_judge_prompts",
        community_advanced_if.load_judge_prompts,
    )
    monkeypatch.delenv("SIEVAL_ADVANCED_IF_SRC", raising=False)
    community_advanced_if.load_judge_prompts.cache_clear()

    with pytest.raises(RuntimeError, match="SIEVAL_ADVANCED_IF_SRC"):
        _task()


def test_build_grader_accepts_mapping_and_model():
    built = AdvancedIFZeroShotGenTask._build_grader(
        {"model": "o3-mini", "api_key": "fake"}
    )
    assert isinstance(built, ChatModel)
    existing = _ScriptedChatModel(reply="{}")
    assert AdvancedIFZeroShotGenTask._build_grader(existing) is existing


# --- preprocess ---


@pytest.mark.anyio
async def test_preprocess_sends_the_conversation_as_messages():
    task, _ = _task()
    sample = _sample()
    record = await task.preprocess(sample, TaskContext(sample_id=0, raw_sample=sample))
    assert record["prompt"] == [
        {"role": "user", "content": "Write a story."},
        {"role": "assistant", "content": "Once upon a time."},
        {"role": "user", "content": "Now make it rhyme."},
    ]


@pytest.mark.anyio
async def test_preprocess_omits_reference_and_carries_the_rubric():
    """The ground truth is a rubric -- a procedure, not a value to compare."""
    task, _ = _task()
    sample = _sample()
    record = await task.preprocess(sample, TaskContext(sample_id=0, raw_sample=sample))
    assert "reference" not in record
    assert record["extra"]["benchmark_name"] == COMPLEX
    assert record["extra"]["rubrics"] == [
        "Is it two paragraphs?",
        "Are there two metaphors?",
    ]


# --- postprocess ---


@pytest.mark.anyio
async def test_postprocess_normalizes_a_blank_response_to_none():
    task, _ = _task(answer="   ")
    ctx = TaskContext(sample_id=0)
    inferred = ModelOutput(model=task.model.meta(), texts=["   "])
    record = await task.postprocess(inferred, ctx)
    assert record["rollouts"][0].get("prediction") is None
    assert record["rollouts"][0]["extracted"] is False


# --- feedback ---


@pytest.mark.anyio
async def test_feedback_scores_a_fully_satisfied_response():
    task, grader = _task(
        grader_reply=_judge_reply({"question_1": "Yes", "question_2": "Yes"}, "Yes")
    )
    ok, judgement = await _run_to_feedback(task, _sample())

    assert ok is True
    rollout = judgement["rollouts"][0]
    assert rollout["correct"] is True
    assert rollout["score"] == 1.0
    assert rollout["metrics"] == {
        "satisfied_all": True,
        "rubric_level_pass_rate": 1.0,
    }
    assert judgement["reference"] is None
    assert judgement["extra"] == {"benchmark_name": COMPLEX, "n_rubrics": 2}
    # The grader saw the composed judge prompt, not the raw conversation.
    assert grader.last_prompt.startswith(f"JUDGE[{COMPLEX}]")


@pytest.mark.anyio
async def test_feedback_records_partial_credit_and_raw_counts():
    task, _ = _task(
        grader_reply=_judge_reply({"question_1": "Yes", "question_2": "No"}, "No")
    )
    _, judgement = await _run_to_feedback(task, _sample())

    rollout = judgement["rollouts"][0]
    assert rollout["correct"] is False
    assert rollout["score"] == 0.5
    # Raw counts, because a per-sample rate cannot reconstruct a pooled one.
    assert rollout["extra"]["n_checks"] == 2
    assert rollout["extra"]["n_checks_passed"] == 1
    assert rollout["extra"]["judge_parsed"] is True
    assert rollout["extra"]["rubrics_check"] == {
        "question_1": "Yes",
        "question_2": "No",
    }


@pytest.mark.anyio
async def test_feedback_persists_the_whole_grader_output():
    task, _ = _task(grader_reply=_judge_reply({"question_1": "Yes"}, "No"))
    _, judgement = await _run_to_feedback(task, _sample())

    grader_output = judgement["rollouts"][0]["extra"][GRADER_OUTPUT_KEY]
    assert grader_output["texts"] == [_judge_reply({"question_1": "Yes"}, "No")]
    assert grader_output["usage"]["total_tokens"] == 43


@pytest.mark.anyio
async def test_grader_spend_reaches_the_profiler():
    """One batched grader call per rollout, stored as a mapping -- not a list.

    ``iter_grader_outputs`` skips a list, so fanning the rubric out into one
    judge call per criterion would make the grader's tokens vanish from
    profile.json. The whole rubric goes in a single indexed call instead.
    """
    task, _ = _task(grader_reply=_judge_reply({"question_1": "Yes"}, "Yes"))
    _, judgement = await _run_to_feedback(task, _sample())

    outputs = iter_grader_outputs(judgement)
    assert len(outputs) == 1
    assert outputs[0]["usage"]["total_tokens"] == 43


@pytest.mark.anyio
async def test_feedback_treats_an_unparseable_reply_as_a_failed_row():
    """Upstream's failed-row path: no pass, and no rubrics into the micro rate."""
    task, _ = _task(grader_reply="the judge rambled without emitting JSON")
    _, judgement = await _run_to_feedback(task, _sample())

    rollout = judgement["rollouts"][0]
    assert rollout["correct"] is False
    assert rollout["score"] == 0.0
    assert rollout["extra"]["judge_parsed"] is False
    assert rollout["extra"]["n_checks"] == 0
    assert rollout["extra"]["n_checks_passed"] == 0
    # The reply is still on disk -- the only evidence of what the grader did.
    assert rollout["extra"][GRADER_OUTPUT_KEY]["texts"] == [
        "the judge rambled without emitting JSON"
    ]


# --- report ---


def _final(
    benchmark_name: str,
    satisfied: bool,
    n_checks: int,
    n_passed: int,
    judge_parsed: bool = True,
):
    rate = n_passed / n_checks if n_checks else 0.0
    judgement = build_judgement_record(
        None,
        [
            build_rollout_judgement(
                0,
                satisfied,
                score=rate,
                metrics={"satisfied_all": satisfied, "rubric_level_pass_rate": rate},
                extra={
                    "n_checks": n_checks,
                    "n_checks_passed": n_passed,
                    "judge_parsed": judge_parsed,
                },
            )
        ],
        extra={"benchmark_name": benchmark_name, "n_rubrics": n_checks},
    )
    return TaskContext(sample_id=0, feedback_result=judgement)


@pytest.mark.anyio
async def test_report_pools_both_published_rates():
    task, _ = _task()
    finals = [
        _final(COMPLEX, True, 4, 4),
        _final(COMPLEX, False, 4, 2),
    ]
    report = await task.report(finals, [])

    assert report["score"] == pytest.approx(50.0)
    assert report["overall_pass_rate"] == pytest.approx(50.0)
    assert report["micro_pass_rate"] == pytest.approx(75.0)
    assert report["macro_pass_rate"] == pytest.approx(75.0)
    assert report["n_graded"] == 2.0
    assert report["n_rubric_checks"] == 8.0
    assert report["n_judge_unparsed"] == 0.0
    assert report["fails"] == 0


@pytest.mark.anyio
async def test_report_macro_rate_reaches_the_report():
    """The per-rollout rubric rate is pooled, not left in the shard data.

    Equal rubric counts make micro == macro, so the two are separated with
    samples of different widths.
    """
    task, _ = _task()
    report = await task.report(
        [_final(COMPLEX, True, 1, 1), _final(COMPLEX, False, 9, 1)], []
    )

    assert report["micro_pass_rate"] == pytest.approx(20.0)
    assert report["macro_pass_rate"] == pytest.approx((1.0 + 1 / 9) / 2 * 100)


@pytest.mark.anyio
async def test_report_breaks_down_by_aspect():
    task, _ = _task()
    finals = [
        _final(COMPLEX, True, 2, 2),
        _final(STEERABILITY, False, 2, 1),
        _final(STEERABILITY, False, 2, 0),
    ]
    report = await task.report(finals, [])

    assert report[f"{COMPLEX}_pass_rate"] == pytest.approx(100.0)
    assert report[f"{COMPLEX}_n_graded"] == 1.0
    assert report[f"{STEERABILITY}_pass_rate"] == pytest.approx(0.0)
    assert report[f"{STEERABILITY}_micro_pass_rate"] == pytest.approx(25.0)
    assert report[f"{STEERABILITY}_macro_pass_rate"] == pytest.approx(25.0)
    assert report[f"{STEERABILITY}_n_graded"] == 2.0


@pytest.mark.anyio
async def test_report_separates_a_broken_grader_from_a_failing_model():
    """Two runs with identical rates, one of them graded by a broken grader.

    Every rate scores an unparseable reply exactly as it scores a response that
    missed every rubric, so no pass rate tells the two apart -- overall or per
    aspect, and a grader tends to break format on the long rows first.
    """
    task, _ = _task()
    failing_model = await task.report(
        [_final(COMPLEX, False, 2, 0), _final(STEERABILITY, False, 2, 0)], []
    )
    broken_grader = await task.report(
        [
            _final(COMPLEX, False, 0, 0, judge_parsed=False),
            _final(STEERABILITY, False, 0, 0, judge_parsed=False),
        ],
        [],
    )

    assert failing_model["score"] == broken_grader["score"] == 0.0
    assert failing_model["macro_pass_rate"] == broken_grader["macro_pass_rate"] == 0.0
    assert failing_model["micro_pass_rate"] == broken_grader["micro_pass_rate"] == 0.0
    assert failing_model["n_judge_unparsed"] == 0.0
    assert broken_grader["n_judge_unparsed"] == 2.0
    assert broken_grader[f"{COMPLEX}_n_judge_unparsed"] == 1.0
    assert broken_grader[f"{STEERABILITY}_n_judge_unparsed"] == 1.0


@pytest.mark.anyio
async def test_report_counts_pipeline_failures_as_non_passes():
    """A sample that never produced an answer still spans the requested set."""
    task, _ = _task()
    finals = [_final(COMPLEX, True, 2, 2)]
    report = await task.report(finals, [TaskContext(sample_id=1)])

    assert report["overall_pass_rate"] == pytest.approx(50.0)
    # A failure contributes no rubrics, so the micro rate is unaffected.
    assert report["micro_pass_rate"] == pytest.approx(100.0)
    # The macro rate does see it, as a 0.0 sample.
    assert report["macro_pass_rate"] == pytest.approx(50.0)
    assert report["n_graded"] == 1.0
    assert report["fails"] == 1
    # A sample that never reached the grader is not a grader-parse failure.
    assert report["n_judge_unparsed"] == 0.0
    # It has no aspect to attribute to, so the breakdown covers graded rollouts.
    assert report[f"{COMPLEX}_n_graded"] == 1.0
