"""Import-discipline and few-shot-pool tests for the DROP k-shot task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import interval_declaration_problems
from sieval.datasets.drop import DROPDataset, DROPDatasetSample
from sieval.tasks.drop_kshot_gen import DROPFewShotGenTask
from tests.conftest import HandlerTransport


class _StubChatModel(ChatModel):
    def __init__(self):
        super().__init__(model="mock-chat", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        return Response(texts=("Answer: 1",) * req.sampling.n)


def _sample(tag: str) -> DROPDatasetSample:
    return {
        "context": f"ctx-{tag}",
        "completion": f"ans-{tag}",
        "ref_text": f"ans-{tag}",
    }


def _dataset(train: list[DROPDatasetSample] | None) -> DROPDataset:
    splits = {"test": HFDataset.from_list([dict(_sample("eval"))])}
    if train is not None:
        splits["train"] = HFDataset.from_list([dict(s) for s in train])
    return DROPDataset(_hf_dict=HFDatasetDict(splits))


# --- the few-shot pool must supply every shot meta.json records ------------


@pytest.mark.anyio
async def test_setup_aborts_when_train_split_is_shorter_than_n_shot():
    """A short pool used to truncate silently while meta.json still said n_shot.

    The abort belongs in setup() so it lands before any inference spend.
    """
    task = DROPFewShotGenTask(_dataset([_sample("t0")]), _StubChatModel(), n_shot=3)
    assert task.n_shot == 3
    with pytest.raises(ValueError, match="requires at least 3 examples"):
        await task.setup()


@pytest.mark.anyio
async def test_setup_aborts_when_fewshot_split_is_absent():
    task = DROPFewShotGenTask(_dataset(None), _StubChatModel(), n_shot=3)
    with pytest.raises(ValueError, match="requires a 'train' split"):
        await task.setup()


@pytest.mark.anyio
async def test_exactly_enough_examples_is_accepted():
    train = [_sample("t0"), _sample("t1"), _sample("t2")]
    task = DROPFewShotGenTask(_dataset(train), _StubChatModel(), n_shot=3)
    await task.setup()
    raw = _sample("eval")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    # All three exemplars reached the prompt — the count is not merely declared.
    content = pre["prompt"][0]["content"]
    for tag in ("t0", "t1", "t2"):
        assert f"ctx-{tag}" in content


@pytest.mark.anyio
async def test_zero_shot_needs_no_train_split():
    """n_shot=0 renders an empty block, so a missing pool is not an error."""
    task = DROPFewShotGenTask(_dataset(None), _StubChatModel(), n_shot=0)
    await task.setup()
    raw = _sample("eval")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    assert "ctx-t0" not in pre["prompt"][0]["content"]


def test_negative_n_shot_rejected():
    with pytest.raises(ValueError, match="n_shot must be >= 0"):
        DROPFewShotGenTask(_dataset([_sample("t0")]), _StubChatModel(), n_shot=-1)


# --- hoisting the draw into setup() must not move the prompt ---------------


@pytest.mark.anyio
async def test_prompt_is_identical_with_and_without_setup():
    """The exemplar draw is seeded, so caching it in setup() is a pure hoist.

    preprocess() keeps a lazy fallback for callers that skip setup(); both paths
    must render the same bytes, or the hoist changed scores.
    """
    train = [_sample(f"t{i}") for i in range(5)]
    raw = _sample("eval")
    ctx = TaskContext(sample_id=0, raw_sample=raw)

    cached = DROPFewShotGenTask(_dataset(train), _StubChatModel(), n_shot=3)
    await cached.setup()
    from_setup = await cached.preprocess(raw, ctx)

    lazy = DROPFewShotGenTask(_dataset(train), _StubChatModel(), n_shot=3)
    from_fallback = await lazy.preprocess(raw, ctx)

    assert from_setup == from_fallback


@pytest.mark.anyio
async def test_repeated_preprocess_is_stable():
    train = [_sample(f"t{i}") for i in range(5)]
    task = DROPFewShotGenTask(_dataset(train), _StubChatModel(), n_shot=3)
    await task.setup()
    raw = _sample("eval")
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    first = await task.preprocess(raw, ctx)
    second = await task.preprocess(raw, ctx)
    assert first == second


# --- the headline interval rides the per-question F1, rescaled to 0-1 -------


@pytest.mark.anyio
async def test_report_interval_rides_per_question_f1_on_the_right_scale():
    """`f1` is stored on 0-100, so it must be divided by 100 going in.

    Passed raw, every value would be 100x over, `p` would exceed 1.0, and
    `wilson_interval` would take its saturated Clopper-Pearson branch -- which
    returns a bound near the top of the range regardless of the data. The upper
    bound staying below 100 is what fails if the rescale is dropped.
    """
    task = DROPFewShotGenTask(_dataset(None), _StubChatModel(), n_shot=0)

    def _final(sample_id: int, *, f1: float, em: float) -> TaskContext:
        metrics: dict[str, bool | float] = {"em": em, "f1": f1}
        return TaskContext(
            sample_id=sample_id,
            feedback_result=build_judgement_record(
                ["ans"],
                [build_rollout_judgement(0, em == 1.0, metrics=metrics)],
                metrics=metrics,
            ),
        )

    # Partial credit, which is the shape DROP's mean-of-per-question-F1 exists
    # for: 100.0 and 50.0 on the stored 0-100 scale.
    finals = [_final(0, f1=100.0, em=1.0), _final(1, f1=50.0, em=0.0)]
    report = await task.report(finals, [])

    assert report["f1"] == pytest.approx(75.0)
    assert report["score"] == report["f1"]
    assert report["em"] == pytest.approx(50.0)
    assert report["n_problems"] == 2
    interval = report["score_ci95"]
    assert isinstance(interval, list)
    lo, hi = interval
    assert lo < report["score"] < hi
    # On the right scale the mean is 0.75, not 75. Unrescaled it saturates and
    # the upper bound pins to the top of the range.
    assert hi < 100.0
    # `f1` is `score` under its own name, so it repeats the headline bounds; `em`
    # is the co-equal column and gets its OWN, over its own per-question values.
    # The two differ here -- 0.5 against 0.75 -- so `em` borrowing the headline's
    # is what these assertions catch.
    assert report["f1_ci95"] == [lo, hi]
    em_interval = report["em_ci95"]
    assert isinstance(em_interval, list)
    assert em_interval != interval
    assert em_interval[0] < report["em"] < em_interval[1]
    assert report["ci95_units"] == {
        "score": "n_problems",
        "f1": "n_problems",
        "em": "n_problems",
    }
    # The task tests call report() directly, so the runner's finalizer never sees
    # this dict -- run the validator here or a missing declaration ships.
    assert interval_declaration_problems(report) == []
