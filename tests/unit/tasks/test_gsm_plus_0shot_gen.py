"""Unit tests for the GSM-Plus 0-shot CoT task.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.gsm_plus import (
    extract_gold_ans,
    extract_pred_ans,
    extract_pred_ans_none,
    is_equivalent,
)
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
)
from sieval.core.utils.offload import GRADE_TIMEOUT
from sieval.datasets.gsm_plus import GSMPlusDataset, GSMPlusDatasetSample
from sieval.tasks import gsm_plus_0shot_gen as module
from sieval.tasks.gsm_plus_0shot_gen import (
    SYSTEM_INSTRUCTION,
    GSMPlusZeroShotGenTask,
    _metric_key,
    _user_turn,
)


class _CapturingChatModel(ChatModel):
    def __init__(self, text: str):
        super().__init__(model="mock-chat", api_key="fake")
        self.last_kwargs: dict[str, object] = {}
        self._text = text

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(model=self.meta(), texts=[self._text])

    async def _alogprobs_impl(
        self,
        prompt,
        *,
        max_tokens: int = 1,
        logprobs: int = 5,
        echo: bool = True,
        temperature: float = 0.0,
        **kwargs,
    ) -> ModelOutput:
        _ = (prompt, max_tokens, logprobs, echo, temperature, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _sample(
    perturbation_type: str = "numerical substitution",
    answer: str = "27",
) -> GSMPlusDatasetSample:
    return {
        "question": "What is 25 + 2?",
        "solution": f"Work.\n#### {answer}",
        "answer": answer,
        "perturbation_type": perturbation_type,
        "seed_question": "What is 20 + 2?",
        "seed_solution": "Work.\n#### 22",
        "seed_answer": "22",
    }


def _task(text: str = "x"):
    dataset = GSMPlusDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _CapturingChatModel(text=text)
    return GSMPlusZeroShotGenTask(dataset, model), model


# --- Pinning: prompt is byte-for-byte cot_prompt_map_func(question) ---


def test_system_instruction_pinned():
    assert SYSTEM_INSTRUCTION == (
        "Your task is to solve a series of math word problems by providing the "
        "final answer. Use the format #### [value] to highlight your answer. "
        "For example, if the answer is 560, you should write #### 560. Make "
        "sure to carefully read and understand each problem before providing "
        "your answer."
    )


def test_user_turn_pinned():
    assert _user_turn("Q?") == "Question:\nQ?\nAnswer:\nLet's think step by step."


@pytest.mark.anyio
async def test_preprocess_builds_system_then_user_turn():
    task, _ = _task()
    raw = _sample()
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    messages = pre["prompt"]
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": SYSTEM_INSTRUCTION}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == _user_turn("What is 25 + 2?")
    # The gold reaches disk from preprocess; raw_sample is never serialized.
    assert pre["reference"] == "27"
    assert pre["extra"]["perturbation_type"] == "numerical substitution"


# --- Gold comes from `solution` (upstream get_gsmplus), not `answer` ---


def test_extract_gold_ans_reads_hash_segment():
    assert extract_gold_ans("Work.\n#### 1,000") == "1000"
    assert extract_gold_ans("Work.\n#### 4.33") == "4.33"
    assert extract_gold_ans("Work.\n#### 3/5") == "3/5"


def test_extract_gold_ans_keeps_none_for_critical_thinking():
    assert extract_gold_ans("We don't know how many eggs.\n#### None") == "None"


def test_extract_gold_ans_rejects_unparseable_gold():
    # Upstream drops into pdb.set_trace() here; a library must raise instead.
    with pytest.raises(ValueError, match="neither '####' nor"):
        extract_gold_ans("no marker at all")


# --- Numeric extraction: #### segment wins, last-number fallback ---


def test_extract_pred_ans_prefers_hash_segment():
    assert extract_pred_ans("First 12, then 30.\n#### 27") == "27"


def test_extract_pred_ans_last_number_fallback():
    assert extract_pred_ans("first 12 then finally 30") == "30"


def test_extract_pred_ans_ignores_hash_segment_without_digits():
    # `#### None` has no digit, so upstream falls back to the last number.
    assert extract_pred_ans("I computed 9 eggs.\n#### None") == "9"


def test_extract_pred_ans_empty_when_no_number():
    assert extract_pred_ans("cannot be determined") == ""


# --- critical thinking: refusal phrasing, not a number ---


def test_extract_pred_ans_none_detects_refusal():
    assert (
        extract_pred_ans_none("The problem does not specify how many eggs.\n#### 5")
        == "None"
    )


def test_extract_pred_ans_none_rejects_confident_number():
    assert extract_pred_ans_none("She makes 18 dollars.\n#### 18") == ""


def test_extract_pred_ans_none_credits_missing_hash_marker():
    # Upstream leniency, kept verbatim and load-bearing for its published
    # numbers: no "####" in the response scores "None" (correct) regardless.
    assert extract_pred_ans_none("She makes 18 dollars.") == "None"


# --- Scoring: normalize_final_answer + check_sympy_equivalence ---


def test_is_equivalent_matches_fraction_and_decimal():
    # sieval's [math] extra pins the ANTLR runtime parse_latex needs, so the
    # vendored sympy branch is live where upstream's own env fell back to string
    # equality and scored these wrong.
    assert is_equivalent("4.5", "9/2") is True
    assert is_equivalent("7/20", "0.35") is True
    assert is_equivalent("3", "3/1") is True


def test_is_equivalent_rejects_different_numbers():
    assert is_equivalent("27", "18") is False


def test_is_equivalent_handles_none_answer():
    assert is_equivalent("None", "None") is True
    assert is_equivalent("None", "") is False


# --- postprocess / feedback wiring ---


@pytest.mark.anyio
async def test_postprocess_dispatches_on_perturbation_type():
    task, model = _task()
    raw = _sample(perturbation_type="critical thinking", answer="None")
    inf = ModelOutput(
        model=model.meta(), texts=["The problem does not provide the egg count."]
    )
    post = await task.postprocess(inf, TaskContext(sample_id=0, raw_sample=raw))
    assert post["rollouts"][0]["prediction"] == "None"


@pytest.mark.anyio
async def test_postprocess_reports_none_when_nothing_extracted():
    task, model = _task()
    raw = _sample()
    inf = ModelOutput(model=model.meta(), texts=["no digits here"])
    post = await task.postprocess(inf, TaskContext(sample_id=0, raw_sample=raw))
    # `None` is the protocol's "could not extract" — distinct from the string
    # "None", which is a real answer on a critical-thinking row.
    assert post["rollouts"][0]["prediction"] is None


@pytest.mark.anyio
async def test_feedback_scores_correct_answer():
    task, model = _task()
    raw = _sample(answer="1,000")
    inf = ModelOutput(model=model.meta(), texts=["Work.\n#### 1000"])
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inf)
    post = await task.postprocess(inf, ctx)
    finalize, fb = await task.feedback(post, ctx)
    assert finalize is True
    assert fb["reference"] == "1000"
    assert fb["rollouts"][0]["correct"] is True
    assert fb["extra"]["perturbation_type"] == "numerical substitution"


@pytest.mark.anyio
async def test_feedback_scores_wrong_answer():
    task, model = _task()
    raw = _sample(answer="27")
    inf = ModelOutput(model=model.meta(), texts=["Work.\n#### 18"])
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inf)
    post = await task.postprocess(inf, ctx)
    _, fb = await task.feedback(post, ctx)
    assert fb["rollouts"][0]["correct"] is False


@pytest.mark.anyio
async def test_feedback_credits_recognized_unanswerable():
    task, model = _task()
    raw = _sample(perturbation_type="critical thinking", answer="None")
    inf = ModelOutput(model=model.meta(), texts=["It does not specify.\n#### 5"])
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inf)
    post = await task.postprocess(inf, ctx)
    _, fb = await task.feedback(post, ctx)
    assert fb["reference"] == "None"
    assert fb["rollouts"][0]["correct"] is True


@pytest.mark.anyio
async def test_feedback_reads_a_record_whose_prediction_key_was_omitted():
    # A None `prediction` is DROPPED on write, so the fresh path carries the key
    # and the resumed path does not. Both must score the same and neither may
    # raise -- this is the routine `critical thinking` case where the response
    # held no refusal phrase, 487 of 10552 on a real run.
    task, _ = _task()
    raw = _sample(perturbation_type="critical thinking", answer="None")
    ctx = TaskContext(sample_id=0, raw_sample=raw)

    as_persisted = {"rollouts": [{"index": 0, "extracted": False}]}
    fresh = {"rollouts": [{"index": 0, "prediction": None, "extracted": False}]}

    _, from_disk = await task.feedback(as_persisted, ctx)
    _, in_memory = await task.feedback(fresh, ctx)
    assert from_disk == in_memory
    assert from_disk["rollouts"][0]["correct"] is False


# --- grading is offloaded, and a timeout scores wrong rather than failing ---


@pytest.mark.anyio
async def test_grading_is_bounded_in_a_worker_process(monkeypatch):
    """The mechanism, not the verdict — an inline grade scores identically, so
    reverting the offload keeps every other test in this file passing. Why a
    process: `check_sympy_equivalence` reaches `simplify` with no bound of its
    own, criterion 2 in `core/utils/offload.py`.
    """
    seen: dict[str, object] = {}

    async def _spy(func, *args, timeout=None):
        seen.update(func=func, args=args, timeout=timeout)
        return func(*args)

    monkeypatch.setattr(module, "run_cpu_bound", _spy)

    task, model = _task()
    raw = _sample(answer="27")
    inf = ModelOutput(model=model.meta(), texts=["Work.\n#### 27"])
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inf)
    post = await task.postprocess(inf, ctx)
    _, fb = await task.feedback(post, ctx)

    # The spy must actually have been reached — a green suite proves nothing if
    # the patched name is no longer the one `feedback` calls.
    assert seen["func"] is is_equivalent
    assert seen["args"] == ("27", "27")
    assert seen["timeout"] == GRADE_TIMEOUT
    assert fb["rollouts"][0]["correct"] is True


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

    task, model = _task()
    raw = _sample(answer="27")
    inf = ModelOutput(model=model.meta(), texts=["Work.\n#### 27"])
    ctx = TaskContext(sample_id=0, raw_sample=raw, infer_result=inf)
    post = await task.postprocess(inf, ctx)

    finalize, fb = await task.feedback(post, ctx)

    assert finalize is True
    assert fb["rollouts"][0]["correct"] is False
    assert fb["reference"] == "27"


# --- report: overall, per-perturbation, wo_critical_thinking ---


def test_metric_key_slugifies_perturbation_type():
    assert (
        _metric_key("integer-decimal-fraction conversion")
        == "integer_decimal_fraction_conversion"
    )
    assert _metric_key("critical thinking") == "critical_thinking"


def _final(sample_id: int, perturbation_type: str, correct: bool) -> TaskContext:
    return TaskContext(
        sample_id=sample_id,
        raw_sample=_sample(perturbation_type=perturbation_type),
        feedback_result=build_judgement_record(
            "27",
            [build_rollout_judgement(0, correct)],
            extra={"perturbation_type": perturbation_type},
        ),
    )


@pytest.mark.anyio
async def test_report_breaks_down_by_perturbation_type():
    task, _ = _task()
    finals = [
        _final(0, "numerical substitution", True),
        _final(1, "numerical substitution", False),
        _final(2, "critical thinking", True),
        _final(3, "critical thinking", True),
    ]
    report = await task.report(finals, [])
    assert report["score"] == 75.0  # 3 of 4
    assert report["accuracy"] == 75.0
    assert report["score_numerical_substitution"] == 50.0
    assert report["score_critical_thinking"] == 100.0
    # upstream's gsmplus_wo_ncr: drops the one cell where refusal is the answer,
    # so the two critical-thinking hits stop propping the headline up
    assert report["score_wo_critical_thinking"] == 50.0
    assert report["fails"] == 0


@pytest.mark.anyio
async def test_report_empty_finals():
    task, _ = _task()
    report = await task.report([], [])
    # The two declarations are present on the empty-run path too: a report saying
    # nothing about its denominator is as unreadable at 0 samples as at 10552.
    assert report == {
        "score": 0.0,
        "accuracy": 0.0,
        "score_wo_critical_thinking": 0.0,
        "fails": 0,
        "n_unextracted": 0.0,
        SCORE_KEY_FIELD: "accuracy",
        DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
    }


@pytest.mark.anyio
async def test_report_counts_fails_in_overall_and_per_type_denominators():
    # A pipeline failure counts as wrong, matching upstream's denominator (every
    # item in the prediction file) and the gsm8k/math-0shot-gen family.
    task, _ = _task()
    finals = [_final(0, "numerical substitution", True)]
    fails = [
        TaskContext(
            sample_id=1, raw_sample=_sample(perturbation_type="numerical substitution")
        )
    ]
    report = await task.report(finals, fails)
    assert report["score"] == 50.0
    assert report["score_numerical_substitution"] == 50.0
    assert report["fails"] == 1


@pytest.mark.anyio
async def test_report_tolerates_fail_without_raw_sample():
    # A context that failed before its sample was attached has no perturbation
    # type, so it cannot land in a per-type cell.
    task, _ = _task()
    finals = [_final(0, "numerical substitution", True)]
    report = await task.report(finals, [TaskContext(sample_id=1)])
    assert report["score"] == 50.0
    assert report["score_numerical_substitution"] == 100.0
    # ...but it MUST still land in `score_wo_critical_thinking`, which declares
    # the same `requested` denominator as `score`. Left out, this co-headline read
    # 100.0 on a run whose `score` was 50.0.
    assert report["score_wo_critical_thinking"] == 50.0


@pytest.mark.anyio
async def test_infer_injects_no_decode_params():
    task, model = _task()
    raw = _sample()
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)
    await task.infer(pre, ctx)
    for forbidden in ("temperature", "top_p", "max_tokens", "n", "stop"):
        assert forbidden not in model.last_kwargs
