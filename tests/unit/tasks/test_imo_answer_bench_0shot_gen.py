"""The grader self-check `imo_answer_bench` needs and its siblings do not.

Every other member of the pass@k math family grades through
`_math_verify.verify_answer`, which has no handler of its own, so a broken
`math_verify` raises and the family's `except TimeoutError` contract turns it
into a `fails` entry. This task grades through the vendored
`community/imo_bench.verify_math_answer`, which catches `Exception` itself and
falls back to `gold.strip().lower() == pred.strip().lower()` -- upstream's
behaviour, which stays. A broken LaTeX backend therefore never reaches the call
site at all: it silently regrades the run by string equality, scoring every
expression answer wrong while `fails` stays 0.

`_ensure_grader_healthy` is what closes that half, so these assert the probe
itself: it fires on a definite negative, stays out of the way otherwise, runs
once rather than per sample, and treats a timeout as inconclusive rather than as
a failure -- a loaded box must not read as a broken one.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ChatModel, Request, Response
from sieval.core.tasks import TaskContext, build_prediction_record
from sieval.datasets.imo_answer_bench import IMOAnswerBenchDataset
from sieval.tasks import imo_answer_bench_0shot_gen as task_mod
from sieval.tasks.imo_answer_bench_0shot_gen import (
    GRADER_CANARY,
    IMOAnswerBenchZeroShotGenTask,
)
from tests.conftest import HandlerTransport

PROBLEM = "What is 6 times 7?"
ANSWER = "42"


class _StubChatModel(ChatModel):
    def __init__(self):
        super().__init__(model="mock-chat", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        return Response(texts=(rf"\boxed{{{ANSWER}}}",) * req.sampling.n)


def _build() -> IMOAnswerBenchZeroShotGenTask:
    rows = HFDataset.from_list([{"problem": PROBLEM, "answer": ANSWER}])
    dataset = IMOAnswerBenchDataset(
        _hf_dict=HFDatasetDict({"train": rows, "test": rows})
    )
    return IMOAnswerBenchZeroShotGenTask(dataset, _StubChatModel(), k=1, n=1)


class _Grader:
    """A `run_cpu_bound` stand-in separating the PROBE call from a real grade.

    The probe is the one whose positional args are `GRADER_CANARY`; everything
    else is a sample being graded. Both are counted, so a test cannot pass
    because the probe was never reached -- which is what moving the call out of
    `feedback` would otherwise look like.
    """

    def __init__(self, *, probe_result: object = True, grade_result: bool = True):
        self._probe_result = probe_result
        self._grade_result = grade_result
        self.probe_calls = 0
        self.grade_calls = 0

    async def __call__(self, _fn, *args, **_kwargs):
        if args == GRADER_CANARY:
            self.probe_calls += 1
            if isinstance(self._probe_result, type) and issubclass(
                self._probe_result, BaseException
            ):
                raise self._probe_result("probe stub")
            return self._probe_result
        self.grade_calls += 1
        return self._grade_result


async def _feedback(task, sample_id: int = 0):
    post = build_prediction_record([ANSWER])
    return await task.feedback(
        post,
        TaskContext(
            sample_id=sample_id,
            raw_sample={"problem": PROBLEM, "answer": ANSWER},
            postprocess_result=post,
        ),
    )


@pytest.mark.anyio
async def test_a_dead_latex_backend_fails_the_run_instead_of_the_model(monkeypatch):
    # The failure this task cannot otherwise see: `verify_math_answer` returns a
    # verdict rather than raising, so without the probe the run would finish with
    # a depressed score and `fails = 0`.
    task = _build()
    grader = _Grader(probe_result=False)
    monkeypatch.setattr(task_mod, "run_cpu_bound", grader)

    with pytest.raises(RuntimeError, match="string equality"):
        await _feedback(task)
    assert grader.probe_calls == 1
    # It never got as far as grading anything -- the point of failing early.
    assert grader.grade_calls == 0


@pytest.mark.anyio
async def test_a_healthy_grader_is_probed_and_then_stays_out_of_the_way(monkeypatch):
    task = _build()
    grader = _Grader(probe_result=True, grade_result=True)
    monkeypatch.setattr(task_mod, "run_cpu_bound", grader)

    _, judgement = await _feedback(task)

    assert judgement["rollouts"][0]["correct"] is True
    assert grader.probe_calls == 1
    assert grader.grade_calls == 1


@pytest.mark.anyio
async def test_the_probe_runs_once_per_task_not_once_per_sample(monkeypatch):
    # A probe is a worker round trip; paying it per sample would be a real cost
    # on a 400-problem benchmark.
    task = _build()
    grader = _Grader(probe_result=True)
    monkeypatch.setattr(task_mod, "run_cpu_bound", grader)

    for sample_id in range(3):
        await _feedback(task, sample_id=sample_id)

    assert grader.probe_calls == 1
    assert grader.grade_calls == 3


@pytest.mark.anyio
async def test_a_probe_timeout_is_inconclusive_rather_than_a_failed_run(monkeypatch):
    # A timeout says the box is loaded, not that the backend is dead: grading
    # must continue, and the probe must stop retrying rather than pay a 30s
    # timeout per remaining sample.
    task = _build()
    grader = _Grader(probe_result=TimeoutError)
    monkeypatch.setattr(task_mod, "run_cpu_bound", grader)

    _, judgement = await _feedback(task)
    await _feedback(task, sample_id=1)

    assert judgement["rollouts"][0]["correct"] is True
    assert grader.probe_calls == 1
    assert grader.grade_calls == 2
