"""Unit tests for the AGIEval 0-shot generative task.

The load-bearing behaviours here are the two-stage inference (what stage 2 is
prompted with, and that both calls survive into the stage value) and the report's
group macros.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import cast

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.agieval.dataset_loader import MATH_SUBSETS
from sieval.community.agieval.evaluation import LEADERBOARD_EN_MCQ_SUBSETS
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_JUDGED,
    SCORE_KEY_FIELD,
)
from sieval.datasets.agieval import AGIEvalDataset, AGIEvalDatasetSample
from sieval.tasks.agieval_0shot_gen import AGIEvalZeroShotGenTask


class _ScriptedChatModel(ChatModel):
    """Replies with ``reply``, recording each prompt and each resolved kwarg set."""

    def __init__(self, name: str = "mock-chat", reply: str = "The answer is D", **args):
        super().__init__(model=name, api_key="fake", **args)
        self.reply = reply
        self.prompts: list = []
        self.resolved: list[dict] = []

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        self.prompts.append(prompt)
        # The merge ChatModel._agenerate_impl performs: configured model args
        # first, call-site kwargs last, so the call site wins.
        resolved = {**self._kwargs, **kwargs}
        self.resolved.append(resolved)
        return ModelOutput(model=self.meta(), texts=[self.reply] * resolved.get("n", 1))

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _sample(subset: str = "sat-math", **overrides) -> AGIEvalDatasetSample:
    sample: dict = {
        "subset": subset,
        "passage": None,
        "question": "Q?",
        "options": ["(A)1", "(B)2", "(C)3", "(D)4"],
        "label": "D",
        "answer": None,
        "other": {},
    }
    sample.update(overrides)
    return cast(AGIEvalDatasetSample, sample)


def _task(model=None, extractor="self", **kwargs) -> AGIEvalZeroShotGenTask:
    """Build the task with the one-model configuration (``extractor="self"``).

    There is no default extractor — it decides the score — so every test states
    which of the two configurations it is exercising. Left unannotated so the
    rejection tests can hand it something the signature forbids.
    """
    dataset = AGIEvalDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    return AGIEvalZeroShotGenTask(
        dataset, model or _ScriptedChatModel(), extractor=extractor, **kwargs
    )


@pytest.mark.anyio
async def test_preprocess_carries_subset_and_reference():
    task = _task()
    pre = await task.preprocess(_sample(), TaskContext(sample_id=0))

    assert pre["extra"]["subset"] == "sat-math"
    assert pre["reference"] == "D"
    # Upstream's system turn is part of the measured prompt: it is prepended
    # inside openai_api.query_azure_openai_chat, which both stages route through.
    assert pre["prompt"][0] == {
        "role": "system",
        "content": "You are a helpful AI assistant.",
    }
    assert pre["prompt"][1]["content"].startswith("Q: Q? Answer Choices:")


@pytest.mark.anyio
async def test_preprocess_uses_the_cloze_answer_as_reference():
    task = _task()
    sample = _sample("math", options=[], label=None, answer="42")
    pre = await task.preprocess(sample, TaskContext(sample_id=0))

    assert pre["reference"] == "42"


@pytest.mark.anyio
async def test_infer_runs_two_stages_and_returns_both_outputs():
    model = _ScriptedChatModel(reply="I think it is D")
    task = _task(model)
    pre = await task.preprocess(_sample(), TaskContext(sample_id=0))

    outputs = await task.infer(pre, TaskContext(sample_id=0))

    # Both calls come back, so the runner profiles both and the first stage's
    # reasoning stays on disk.
    assert len(outputs) == 2
    assert len(model.prompts) == 2
    # Stage 2 re-sends the stage-1 prompt + reply + the family extraction cue,
    # under the same system turn upstream sends on both calls.
    assert [m["role"] for m in model.prompts[1]] == ["system", "user"]
    assert model.prompts[1][0]["content"] == "You are a helpful AI assistant."
    stage2 = model.prompts[1][1]["content"]
    # The QUESTION, not the system turn: stage 2's context is selected by role.
    assert stage2.startswith("Q: Q? Answer Choices:")
    assert "I think it is D" in stage2
    assert stage2.endswith("Therefore, among A through E, the answer is")


@pytest.mark.anyio
async def test_infer_routes_stage_two_to_the_extractor_when_given():
    answerer = _ScriptedChatModel("answerer", reply="reasoning...")
    extractor = _ScriptedChatModel("extractor", reply=" D")
    task = _task(answerer, extractor=extractor)
    pre = await task.preprocess(_sample(), TaskContext(sample_id=0))

    outputs = await task.infer(pre, TaskContext(sample_id=0))

    assert len(answerer.prompts) == 1 and len(extractor.prompts) == 1
    assert outputs[1].texts == [" D"]


@pytest.mark.anyio
async def test_infer_carries_an_empty_first_stage_reply_into_stage_two():
    class _Empty(_ScriptedChatModel):
        async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
            _ = kwargs
            self.prompts.append(prompt)
            return ModelOutput(model=self.meta(), texts=[])

    model = _Empty()
    task = _task(model)
    pre = await task.preprocess(_sample(), TaskContext(sample_id=0))

    # Upstream feeds "" onward rather than dropping the sample.
    outputs = await task.infer(pre, TaskContext(sample_id=0))
    assert len(outputs) == 2
    assert model.prompts[1][1]["content"].endswith(
        "\n\nTherefore, among A through E, the answer is"
    )


@pytest.mark.anyio
async def test_feedback_survives_a_record_whose_prediction_was_dropped():
    """A resumed run reads records back from disk, where `prediction` is absent.

    `post_process` returns None whenever extraction fails — routine on the cloze
    subsets — and `prediction` is NotRequired, so serialization drops the key.
    Indexing it made resume raise KeyError on exactly the samples a fresh run
    scored 0, so the bug could not show up in a single clean pass.
    """
    task = _task()
    sample = _sample("gaokao-mathcloze", options=[], label=None, answer="2")
    ctx = TaskContext(sample_id=0, raw_sample=sample)

    record = build_prediction_record([None])
    # What a round trip through JSON leaves behind.
    for rollout in record["rollouts"]:
        rollout.pop("prediction", None)
    assert "prediction" not in record["rollouts"][0]

    finalize, fb = await task.feedback(record, ctx)

    assert finalize is True
    assert fb["rollouts"][0]["correct"] is False


@pytest.mark.anyio
async def test_infer_pins_n_to_one_on_both_calls():
    """A stray `n` in the model args must not reach the backend.

    `n` is meaningless here — upstream samples each problem once and only
    rollout 0 is ever scored — but a model config shared with the sampling
    tasks carries one, and call-site kwargs win the model layer's merge.
    Passing `n=1` is what turns such a config into a no-op instead of n x 2
    calls per problem and a run that fails every sample.
    """
    model = _ScriptedChatModel(n=4)
    task = _task(model)
    pre = await task.preprocess(_sample(), TaskContext(sample_id=0))

    outputs = await task.infer(pre, TaskContext(sample_id=0))

    assert [resolved["n"] for resolved in model.resolved] == [1, 1]
    assert [len(out.texts) for out in outputs] == [1, 1]


@pytest.mark.anyio
async def test_infer_rejects_a_backend_that_ignored_n():
    """More than one rollout is not scoreable: only rollout 0 reaches post_process.

    Unreachable by configuration now that both calls pass `n=1`, so this is the
    backstop for a backend that returns several anyway. Raising on stage 1 also
    stops stage 2 from billing for rollouts that scoring would drop.
    """

    class _IgnoresN(_ScriptedChatModel):
        async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
            _ = kwargs
            self.prompts.append(prompt)
            return ModelOutput(model=self.meta(), texts=["A", "B", "C", "D"])

    model = _IgnoresN()
    task = _task(model)
    pre = await task.preprocess(_sample(), TaskContext(sample_id=0))

    with pytest.raises(ValueError, match="backend ignored `n=1`"):
        await task.infer(pre, TaskContext(sample_id=0))

    # Raised on stage 1, so stage 2 was never called.
    assert len(model.prompts) == 1


def test_extractor_is_required():
    """Stage 2's model decides the score, so no default is comparable."""
    with pytest.raises(ValueError, match="requires an `extractor`"):
        _task(extractor=None)


def test_extractor_arg_rejects_a_non_model():
    with pytest.raises(ValueError, match="Got 42"):
        _task(extractor=42)


def test_extractor_rejects_a_string_that_is_not_the_sentinel():
    # "self" is the only accepted string. A near-miss must not fall back to
    # self-extraction, or a typo would silently choose the more expensive of two
    # non-comparable measurements.
    with pytest.raises(ValueError, match="Got 'itself'"):
        _task(extractor="itself")


def test_extractor_self_runs_stage_two_on_the_model_under_test():
    model = _ScriptedChatModel()
    task = _task(model, extractor="self")

    assert task._extractor is model


@pytest.mark.anyio
async def test_postprocess_parses_the_second_stage_reply():
    task = _task()
    inf = [
        ModelOutput(model=None, texts=["long reasoning about A and B"]),  # type: ignore[invalid-argument-type]
        ModelOutput(model=None, texts=[" D."]),  # type: ignore[invalid-argument-type]
    ]
    post = await task.postprocess(inf, TaskContext(sample_id=0, raw_sample=_sample()))

    # Parsed from stage 2 only — parsing stage 1 would have returned "A".
    assert post["rollouts"][0]["prediction"] == "D"
    assert post["rollouts"][0]["extracted"] is True


@pytest.mark.anyio
async def test_feedback_scores_per_subset_rule():
    task = _task()
    ctx = TaskContext(sample_id=0, raw_sample=_sample("jec-qa-kd", label="B"))
    finalize, fb = await task.feedback(build_prediction_record([["B"]]), ctx)

    assert finalize is True
    assert fb["reference"] == "B"
    assert fb["rollouts"][0]["correct"] is True
    assert fb["extra"]["subset"] == "jec-qa-kd"


@pytest.mark.anyio
async def test_feedback_uses_math_equivalence_for_cloze():
    task = _task()
    sample = _sample("math", options=[], label=None, answer="0.5")
    ctx = TaskContext(sample_id=0, raw_sample=sample)
    _, fb = await task.feedback(build_prediction_record(["\\frac{1}{2}"]), ctx)

    assert fb["rollouts"][0]["correct"] is True


def _final(subset: str, correct: bool) -> TaskContext:
    return TaskContext(
        sample_id=0,
        raw_sample=_sample(subset),
        feedback_result=build_judgement_record(
            "D", [build_rollout_judgement(0, correct)], extra={"subset": subset}
        ),
    )


@pytest.mark.anyio
async def test_report_macro_averages_over_subsets_not_samples():
    task = _task()
    # sat-math: 1/2 correct, aqua-rat: 1/1 -> macro 75.0, micro would be 66.7.
    finals = [
        _final("sat-math", True),
        _final("sat-math", False),
        _final("aqua-rat", True),
    ]
    metrics = await task.report(finals, [])

    assert metrics["score"] == pytest.approx(75.0)
    assert metrics["score_sat_math"] == pytest.approx(50.0)
    assert metrics["score_aqua_rat"] == pytest.approx(100.0)
    assert metrics["fails"] == 0.0


@pytest.mark.anyio
async def test_report_omits_group_macros_until_the_whole_group_ran():
    task = _task()
    partial = [_final(subset, True) for subset in MATH_SUBSETS[:-1]]
    assert "macro_math" not in await task.report(partial, [])

    full = [_final(subset, True) for subset in MATH_SUBSETS]
    metrics = await task.report(full, [])
    assert metrics["macro_math"] == pytest.approx(100.0)
    # The MATH subset's own accuracy keeps its own key.
    assert metrics["score_math"] == pytest.approx(100.0)
    assert "macro_en_mcq" not in metrics


@pytest.mark.anyio
async def test_report_emits_the_leaderboard_macro_when_the_group_is_complete():
    task = _task()
    finals = [_final(subset, True) for subset in LEADERBOARD_EN_MCQ_SUBSETS]
    finals.append(_final("lsat-ar", False))

    metrics = await task.report(finals, [1])

    # lsat-ar is 1/2 -> the 8-subset macro is (7*100 + 50) / 8.
    assert metrics["macro_en_mcq"] == pytest.approx(93.75)
    assert metrics["fails"] == 1.0


@pytest.mark.anyio
async def test_report_on_an_all_failed_run_is_zero_not_a_crash():
    task = _task()
    metrics = await task.report([], [1, 2])

    assert metrics["score"] == 0.0
    assert metrics["fails"] == 2.0


@pytest.mark.anyio
async def test_report_declares_which_key_the_headline_came_from():
    task = _task()
    metrics = await task.report([_final("sat-math", True)], [])

    assert metrics[SCORE_KEY_FIELD] == "score"
    # Failures live in `fails` and are absent from every per-subset accuracy, so
    # the headline is averaged over the judged samples, not the requested ones.
    assert metrics[DENOMINATOR_FIELD] == DENOMINATOR_JUDGED
