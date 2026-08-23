"""Unit tests for the GSM1k 0-shot chat task (the GSM8K-paired half).

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.deepseek_math import is_correct
from sieval.core.models import ModelOutput, Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_rollout_judgement,
)
from sieval.core.utils.offload import GRADE_TIMEOUT
from sieval.datasets.gsm1k import GSM1KDataset, GSM1KDatasetSample
from sieval.tasks import gsm1k_0shot_gen as module
from sieval.tasks.gsm1k_0shot_gen import COT_INSTRUCTION, GSM1KZeroShotGenTask
from tests.conftest import HandlerTransport


class _CapturingChatModel(ChatModel):
    def __init__(self, text: str):
        self.last_req: Request | None = None
        self._text = text
        super().__init__(model="mock-chat", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        self.last_req = req
        return Response(texts=(self._text,))


def _sample(answer: str = "42") -> GSM1KDatasetSample:
    return {"question": "What is 40 + 2?", "answer": answer}


def _task(text: str) -> tuple[GSM1KZeroShotGenTask, _CapturingChatModel]:
    dataset = GSM1KDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _CapturingChatModel(text=text)
    return GSM1KZeroShotGenTask(dataset, model), model


@pytest.mark.anyio
async def test_prompt_is_the_deepseek_cot_user_turn():
    task, _ = _task("")

    pre = await task.preprocess(_sample(), TaskContext(sample_id=0))

    assert pre["prompt"] == [
        {"role": "user", "content": "What is 40 + 2?" + COT_INSTRUCTION}
    ]
    # GSM1k's `answer` already IS the bare final answer — no `####` split, which
    # is the one deviation from the gsm8k_0shot_gen sibling this pairs with.
    assert pre["reference"] == "42"


@pytest.mark.anyio
async def test_boxed_answer_is_extracted_and_scored_correct():
    task, model = _task("Reasoning...\n\\boxed{42}")
    raw = _sample()
    inferred = ModelOutput(model=model.meta(), texts=["Reasoning...\n\\boxed{42}"])
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inferred)

    post = await task.postprocess(inferred, ctx)
    finalize, feedback = await task.feedback(post, ctx)

    assert finalize is True
    assert post["rollouts"][0]["prediction"] == "42"
    assert feedback["reference"] == "42"
    assert feedback["rollouts"][0]["correct"] is True


@pytest.mark.anyio
async def test_unextractable_response_is_none_and_scores_wrong():
    task, model = _task("I decline to answer.")
    raw = _sample()
    inferred = ModelOutput(model=model.meta(), texts=["I decline to answer."])
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inferred)

    post = await task.postprocess(inferred, ctx)
    _, feedback = await task.feedback(post, ctx)

    assert post["rollouts"][0]["prediction"] is None
    assert post["rollouts"][0]["extracted"] is False
    assert feedback["rollouts"][0]["correct"] is False


@pytest.mark.anyio
async def test_report_counts_pipeline_failures_as_wrong():
    task, model = _task("\\boxed{42}")
    raw = _sample()
    inferred = ModelOutput(model=model.meta(), texts=["\\boxed{42}"])
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inferred)
    post = await task.postprocess(inferred, ctx)
    _, feedback = await task.feedback(post, ctx)

    report = await task.report(
        [TaskContext(sample_id=0, raw_sample=raw, feedback_result=feedback)],
        [TaskContext(sample_id=1, raw_sample=raw)],
    )

    # Same denominator rule as gsm8k_0shot_gen, so both sides of the paired diff
    # treat a pipeline failure identically.
    assert report["fails"] == 1
    assert report["score"] == report["accuracy"] == 50.0


@pytest.mark.anyio
async def test_report_interval_is_quoted_over_the_requested_population():
    task, _ = _task("")
    raw = _sample()

    def _final(sample_id: int, *, correct: bool) -> TaskContext:
        return TaskContext(
            sample_id=sample_id,
            raw_sample=raw,
            feedback_result=build_judgement_record(
                "42", [build_rollout_judgement(0, correct)]
            ),
        )

    report = await task.report(
        [_final(0, correct=True), _final(1, correct=False)],
        [TaskContext(sample_id=2, raw_sample=raw)],
    )

    # REQUESTED, so the fail is inside the population the interval is quoted
    # over: 3, not the 2 samples that produced a verdict. Reading `len(finals)`
    # here would be the denominator artifact the paired diff exists to avoid.
    assert report["n_problems"] == 3
    interval = report["score_ci95"]
    assert isinstance(interval, list)
    lo, hi = interval
    assert lo < report["score"] < hi


# --- grading is offloaded, and a timeout scores wrong rather than failing ---


@pytest.mark.anyio
async def test_grading_is_bounded_in_a_worker_process(monkeypatch):
    """The mechanism, not the verdict — an inline grade scores identically, so
    reverting the offload keeps every other test in this file passing. Why a
    process: `is_correct` reaches `math_equal` with its default `timeout=False`,
    so `simplify` runs with no bound of its own (criterion 2 in
    `core/utils/offload.py`), on the one event loop the whole session shares.
    """
    seen: dict[str, object] = {}

    async def _spy(func, *args, timeout=None):
        seen.update(func=func, args=args, timeout=timeout)
        return func(*args)

    monkeypatch.setattr(module, "run_cpu_bound", _spy)

    task, model = _task("Reasoning...\n\\boxed{42}")
    raw = _sample()
    inferred = ModelOutput(model=model.meta(), texts=["Reasoning...\n\\boxed{42}"])
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inferred)
    post = await task.postprocess(inferred, ctx)
    _, feedback = await task.feedback(post, ctx)

    # The spy must actually have been reached — a green suite proves nothing if
    # the patched name is no longer the one `feedback` calls.
    assert seen["func"] is is_correct
    assert seen["args"] == ({"prediction": "42", "answer": "42"},)
    assert seen["timeout"] == GRADE_TIMEOUT
    assert feedback["rollouts"][0]["correct"] is True


@pytest.mark.anyio
async def test_a_grading_timeout_scores_wrong_rather_than_failing_the_sample(
    monkeypatch,
):
    # An answer that cannot be graded is a wrong answer, not an infrastructure
    # failure: letting TimeoutError propagate would land the sample in `fails`,
    # which is one of the signals a run is promoted on.
    async def _raise_timeout(_func, *_args, **_kwargs):
        raise TimeoutError("grading took too long")

    monkeypatch.setattr(module, "run_cpu_bound", _raise_timeout)

    task, model = _task("Reasoning...\n\\boxed{42}")
    raw = _sample()
    inferred = ModelOutput(model=model.meta(), texts=["Reasoning...\n\\boxed{42}"])
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inferred)
    post = await task.postprocess(inferred, ctx)

    finalize, feedback = await task.feedback(post, ctx)

    assert finalize is True
    assert feedback["rollouts"][0]["correct"] is False
    assert feedback["reference"] == "42"


@pytest.mark.anyio
async def test_report_on_empty_set_reports_zero():
    task, _ = _task("")

    report = await task.report([], [])

    # The declarations ride on the empty report too, and `requested` here has to
    # match `gsm8k_0shot_gen`'s, or the paired diff is a denominator artifact.
    assert report == {
        "score": 0.0,
        "fails": 0,
        "accuracy": 0.0,
        "score_key": "accuracy",
        "denominator_policy": "requested",
        # Extraction health rides on both halves of the pair, so a diff can never
        # be confused with a difference in how often extraction failed.
        "n_unextracted": 0.0,
    }
