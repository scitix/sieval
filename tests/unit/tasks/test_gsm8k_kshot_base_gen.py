"""Unit tests for the GSM8K k-shot base generative task.

AI-Generated Code - GPT-5-Codex (OpenAI)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput, Request, Response, SamplingParams
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext
from sieval.core.tasks.metrics import interval_declaration_problems
from sieval.datasets.gsm8k import GSM8KDataset, GSM8KDatasetSample
from sieval.tasks.gsm8k_kshot_base_gen import (
    STOP_SEQUENCES,
    GSM8KFewShotBaseGenTask,
    _extract_answer,
    _extract_flexible_match,
)
from tests.conftest import HandlerTransport


class _CapturingGenModel(GenModel):
    def __init__(self):
        self.last_req: Request | None = None
        super().__init__(model="mock-gen", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_completions")

    async def _stub_arun(self, req: Request) -> Response:
        self.last_req = req
        return Response(texts=(" Work shown.\n#### 42",))


def _sample(answer: str = "Solution.\n#### 42") -> GSM8KDatasetSample:
    return {"question": "What is 40 + 2?", "answer": answer}


def _task() -> tuple[GSM8KFewShotBaseGenTask, _CapturingGenModel]:
    dataset = GSM8KDataset(
        _hf_dict=HFDatasetDict(
            {
                "train": HFDataset.from_list([dict(_sample())]),
                "test": HFDataset.from_list([dict(_sample())]),
            }
        )
    )
    model = _CapturingGenModel()
    return GSM8KFewShotBaseGenTask(dataset, model, n_shot=0), model


def test_strict_and_flexible_extractors_are_distinct():
    assert _extract_answer("Therefore the answer is 42.") == ("", "none")
    assert _extract_flexible_match("Therefore the answer is 42.") == (
        "42",
        "flexible-extract",
    )
    assert _extract_answer("Final.\n#### 1,234.") == ("1234", "strict-match")


@pytest.mark.anyio
async def test_infer_only_forwards_prompt_coupled_stop():
    task, model = _task()

    await task.infer(
        {"prompt": "prompt"}, TaskContext(sample_id=0, raw_sample=_sample())
    )

    req = model.last_req
    assert req is not None
    # `n` rides along because it is the sampling budget rather than a decoding
    # param; `stop` is prompt-coupled and everything else stays the caller's.
    assert req.sampling == SamplingParams(n=1, stop=STOP_SEQUENCES)
    assert req.dialect_options is None


@pytest.mark.anyio
async def test_feedback_and_report_include_flexible_secondary_metric():
    task, model = _task()
    raw = _sample()
    inferred = ModelOutput(
        model=model.meta(),
        texts=["No strict delimiter, but the final sentence says 42."],
    )

    post = await task.postprocess(
        inferred,
        TaskContext(sample_id=0, raw_sample=raw, infer_result=inferred),
    )
    finalize, feedback = await task.feedback(
        post,
        TaskContext(sample_id=0, raw_sample=raw, infer_result=inferred),
    )
    report = await task.report(
        [TaskContext(sample_id=0, raw_sample=raw, feedback_result=feedback)],
        [],
    )

    assert finalize is True
    # Co-equal metrics: both readings named, headline derived from the strict one.
    assert feedback["metrics"]["exact_match"] is False
    assert feedback["metrics"]["flexible_exact_match"] is True
    assert feedback["rollouts"][0]["correct"] is False
    assert report["score"] == 0.0
    assert report["exact_match"] == 0.0
    assert report["flexible_exact_match"] == 100.0


@pytest.mark.anyio
async def test_report_brackets_both_extraction_rules_separately():
    """Each rule gets its own bounds, over the values its own rate is a mean of.

    The two are built to disagree: both samples are flexible-correct and
    strict-wrong, so a flexible interval copied onto the strict rule lands at the
    opposite end of the range.
    """
    task, _ = _task()
    raw = _sample()

    def _final(sample_id: int) -> TaskContext:
        metrics: dict[str, bool | float] = {
            "exact_match": False,
            "flexible_exact_match": True,
        }
        return TaskContext(
            sample_id=sample_id,
            raw_sample=raw,
            feedback_result={
                "reference": "42",
                "rollouts": [{"index": 0, "correct": False, "metrics": metrics}],
                "metrics": metrics,
            },
        )

    report = await task.report([_final(0), _final(1)], [])

    assert report["score"] == report["exact_match"] == 0.0
    assert report["flexible_exact_match"] == 100.0
    assert report["n_problems"] == 2
    lo, hi = report["score_ci95"]
    # `exact_match` is `score` under its own name, so it repeats the headline
    # bounds; the flexible rule gets its own at the other end of the range.
    assert report["exact_match_ci95"] == [lo, hi]
    flexible = report["flexible_exact_match_ci95"]
    # Saturated at ONE, where the headline is saturated at zero: an upper limit of
    # exactly 100 and a lower limit above 0. Borrowing the headline's fails both.
    assert flexible[1] == 100.0
    assert flexible[0] > 0.0
    assert hi < 100.0
    assert report["ci95_units"] == {
        "score": "n_problems",
        "exact_match": "n_problems",
        "flexible_exact_match": "n_problems",
    }
    # The task tests call report() directly, so the runner's finalizer never sees
    # this dict -- run the validator here or a missing declaration ships.
    assert interval_declaration_problems(report) == []
