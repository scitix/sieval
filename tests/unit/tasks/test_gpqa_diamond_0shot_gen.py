"""Unit tests for the GPQA-Diamond 0-shot generative task.

This task is where the interval's clustering is load-bearing: it repeats its
split ``n_repeats`` times (4 by default, matching simple-evals), so every
question arrives as four samples and reading those as four independent problems
would narrow the interval by ``sqrt(4)`` with nothing in the report to say so.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ChatModel, Request, Response
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import (
    PROBLEM_COUNT_FIELD,
    SCORE_CI_FIELD,
    rollout_view,
    wilson_interval,
)
from sieval.datasets.gpqa_diamond import GPQADiamondDataset
from sieval.tasks.gpqa_diamond_0shot_gen import GPQADiamondZeroShotGenTask
from tests.conftest import HandlerTransport

N_REPEATS = 4
# 2 of 5 questions solved: dispersion between problems, and a `p` strictly
# inside (0, 1), which is what `wilson_interval` needs to emit anything.
SOLVED = 2
QUESTIONS = 5


class _StubChatModel(ChatModel):
    def __init__(self):
        super().__init__(model="mock-chat", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        return Response(texts=("Answer: A",) * req.sampling.n)


def _question(index: int) -> dict[str, str]:
    return {
        "Question": f"q{index}?",
        "Correct Answer": "right",
        "Incorrect Answer 1": "wrong-1",
        "Incorrect Answer 2": "wrong-2",
        "Incorrect Answer 3": "wrong-3",
    }


def _dataset(questions: int) -> GPQADiamondDataset:
    rows = HFDataset.from_list([_question(i) for i in range(questions)])
    return GPQADiamondDataset(_hf_dict=HFDatasetDict({"test": rows}))


def _final(sample_id: int, raw: dict[str, str], *, correct: bool) -> TaskContext:
    return TaskContext(
        sample_id=sample_id,
        raw_sample=raw,
        feedback_result=build_judgement_record(
            "A", [build_rollout_judgement(0, correct)]
        ),
        postprocess_result=build_prediction_record(["A" if correct else "B"]),
    )


@pytest.mark.anyio
async def test_report_counts_questions_not_copies():
    # `n_repeats=4` puts four copies of every question in the split. They are
    # four samples of ONE problem: counting them as four problems narrows the
    # interval by up to sqrt(4) -- on a real 198-question run (792 samples), by
    # 1.54x: +/-3.48pp reported where +/-5.36pp is true -- and no key in the
    # report would disagree.
    dataset = _dataset(QUESTIONS)
    task = GPQADiamondZeroShotGenTask(dataset, _StubChatModel(), n_repeats=N_REPEATS)

    # The premise: the task's own split is 4x the question count. Asserted, not
    # assumed, so this test cannot pass by the repeat silently going away.
    test_set = task.dataset.test_set
    assert test_set is not None
    assert len(test_set) == QUESTIONS * N_REPEATS

    # Copy-major, matching `Dataset.repeat`: row `i` is question `i % QUESTIONS`.
    finals = [
        _final(i, _question(i % QUESTIONS), correct=(i % QUESTIONS) < SOLVED)
        for i in range(QUESTIONS * N_REPEATS)
    ]
    report = await task.report(finals, [])

    assert report[PROBLEM_COUNT_FIELD] == QUESTIONS
    assert report["score"] == pytest.approx(100 * SOLVED / QUESTIONS)

    interval = report[SCORE_CI_FIELD]
    assert isinstance(interval, list)
    lo, hi = interval
    assert lo < report["score"] < hi

    # And the collapsing is what widened it: the very same verdicts, read as 20
    # independent problems, produce a strictly narrower interval. Read back
    # through `rollout_view` so this cannot drift from what the report scored.
    unclustered = wilson_interval(
        [1.0 if rollout_view(final)[0][0] else 0.0 for final in finals],
        len(finals),
    )
    assert unclustered is not None
    assert hi - lo > unclustered[1] - unclustered[0]
