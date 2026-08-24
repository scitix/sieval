"""
Unit tests for the ARC-Challenge few-shot conditional-log-prob task (options).

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import (
    CompletionInput,
    Request,
    Response,
    TokenLogprob,
    TopKEntry,
)
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import EvalMode, TaskContext
from sieval.core.tasks.meta import get_task_meta
from sieval.core.tasks.metrics import interval_declaration_problems
from sieval.datasets.arc_challenge import (
    ARCChallengeDataset,
    ARCChallengeDatasetSample,
)
from sieval.tasks.arc._base import arc_judgement_record
from sieval.tasks.arc.arc_challenge_kshot_clp import ARCChallengeFewShotClpTask
from tests.conftest import HandlerTransport


class _TopLogprobsGenModel(GenModel):
    """Returns a fixed next-token top_logprobs map; records the prompt + echo."""

    def __init__(self, top: dict[str, float]):
        self._top = top
        self.prompts: list[str] = []
        self.echo_flags: list[bool] = []
        super().__init__(model="mock-gen", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_completions")

    async def _stub_arun(self, req: Request) -> Response:
        if not (req.scoring.sampled_logprobs or req.scoring.input_scoring):
            return Response(texts=("",))
        assert isinstance(req.input, CompletionInput)
        self.prompts.append(req.input.text)
        self.echo_flags.append(req.scoring.input_scoring)
        return Response(
            texts=("A",),
            logprobs=(TokenLogprob(token="A", logprob=-0.1),),
            top_logprobs=(
                tuple(TopKEntry(token=t, logprob=lp) for t, lp in self._top.items()),
            ),
        )


def _train() -> list[ARCChallengeDatasetSample]:
    return [{"question": "1+1?", "choices": ["1", "2", "3"], "answer": 1}]


def _sample() -> ARCChallengeDatasetSample:
    return {
        "question": "Which material is a conductor?",
        "choices": ["copper", "rubber", "wood"],
        "answer": 0,
    }


def _task(
    top: dict[str, float], *, n_shot: int = 0, logprobs: int = 100
) -> tuple[ARCChallengeFewShotClpTask, _TopLogprobsGenModel]:
    dataset = ARCChallengeDataset(
        _hf_dict=HFDatasetDict(
            {
                "train": HFDataset.from_list([dict(s) for s in _train()]),
                "test": HFDataset.from_list([dict(_sample())]),
            }
        )
    )
    model = _TopLogprobsGenModel(top)
    return (
        ARCChallengeFewShotClpTask(dataset, model, n_shot=n_shot, logprobs=logprobs),
        model,
    )


@pytest.mark.anyio
async def test_preprocess_lists_options_with_letters():
    task, _model = _task({}, n_shot=1)
    raw = _sample()

    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert pre["prompt"] == (
        "Question: 1+1?\n"
        "A. 1\n"
        "B. 2\n"
        "C. 3\n"
        "Answer: B\n\n"  # exemplar answer is the LETTER
        "Question: Which material is a conductor?\n"
        "A. copper\n"
        "B. rubber\n"
        "C. wood\n"
        "Answer:"
    )


@pytest.mark.anyio
async def test_single_call_echo_false():
    task, model = _task({" A": -0.1, " B": -2.0, " C": -3.0})
    raw = _sample()
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)

    await task.infer(pre, ctx)

    assert len(model.prompts) == 1  # one inference per sample
    assert model.echo_flags == [False]


@pytest.mark.anyio
async def test_argmax_over_option_letters():
    # Favour " A" (gold index 0); " B"/" C" lower. Token has a leading space
    # (" A") — the scorer strips it to "A".
    task, _model = _task({" A": -0.1, " B": -2.0, " C": -3.0, " the": -0.5})
    raw = _sample()
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)
    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, ctx)
    _finalize, feedback = await task.feedback(post, ctx)
    report = await task.report(
        [TaskContext(sample_id=0, raw_sample=raw, feedback_result=feedback)], []
    )

    assert post["rollouts"][0]["prediction"] == 0
    assert feedback["rollouts"][0]["correct"] is True
    assert report == {
        "score": 100.0,
        "acc": 100.0,
        "fails": 0,
        # `arc_report` names the column the headline came from and the population
        # it is over; ARC excludes pipeline failures (MCQ-family convention).
        "score_key": "acc",
        "denominator_policy": "judged",
    }


@pytest.mark.anyio
async def test_shared_report_pairs_the_interval_with_its_problem_count():
    """`arc_report` is shared by all four leaves, and reached through a leaf here.

    The grouping cannot be read inside a free function, so the leaf resolves it
    and passes it in; this exercises that call path rather than the helper alone.
    """
    task, _model = _task({" A": -0.1, " B": -2.0, " C": -3.0})
    raw = _sample()

    def _final(sample_id: int, *, correct: bool) -> TaskContext:
        return TaskContext(
            sample_id=sample_id,
            raw_sample=raw,
            feedback_result=arc_judgement_record(0 if correct else 1, raw),
        )

    report = await task.report(
        [_final(0, correct=True), _final(1, correct=False)],
        [TaskContext(sample_id=2, raw_sample=raw)],
    )

    assert report["score"] == report["acc"] == 50.0
    # JUDGED: the fail is outside the population, so 2 -- not 3.
    assert report["n_problems"] == 2
    interval = report["score_ci95"]
    assert isinstance(interval, list)
    lo, hi = interval
    assert lo < report["score"] < hi
    # `acc` is `score` under its own name -- `score_key` says so -- so it carries
    # the same bounds. Without them, a reader keyed on the column ARC actually
    # publishes finds no interval at all.
    assert report["acc_ci95"] == [lo, hi]
    # Both names declared over the one population. The task tests call report()
    # directly, so the runner's finalizer never sees this dict -- run the
    # validator here or a missing declaration ships.
    assert report["ci95_units"] == {"score": "n_problems", "acc": "n_problems"}
    assert interval_declaration_problems(report) == []


@pytest.mark.anyio
async def test_missing_option_letter_fails_loud():
    # top-k omits "C" (a 3-option sample needs A/B/C) → RuntimeError, not a guess.
    task, _model = _task({"A": -0.1, "B": -2.0})
    raw = _sample()
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)
    inf = await task.infer(pre, ctx)
    with pytest.raises(RuntimeError, match="missing option token"):
        await task.postprocess(inf, ctx)


def test_negative_k_rejected():
    dataset = ARCChallengeDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    with pytest.raises(ValueError, match="n_shot must be >= 0"):
        ARCChallengeFewShotClpTask(dataset, _TopLogprobsGenModel({}), n_shot=-1)


def test_task_meta_points_to_arc_challenge_dataset():
    meta = get_task_meta(ARCChallengeFewShotClpTask)

    assert meta.name == "arc_challenge_kshot_clp"
    assert meta.dataset == "arc_challenge"
    assert meta.model_type == "gen"
    assert meta.n_shot == 25
    assert meta.eval_mode == EvalMode.CLP
