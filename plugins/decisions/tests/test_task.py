"""Question metrics distinguish refusals, binary and multiclass calibration.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import json
from dataclasses import replace

import pytest
from sieval_decisions import (
    ChoiceAnswer,
    ChoiceProbability,
    DecisionResponse,
    DecisionResult,
    PredicateAnswer,
    RefusalAnswer,
    ScoreAnswer,
)
from sieval_decisions.dataset import DecisionJSONLDataset
from sieval_decisions.task import DecisionJSONLTask


def task_and_context(tmp_path, **kwargs):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "input": "ticket",
                "questions": [
                    {"type": "predicate", "id": "p", "instructions": "yes?"},
                    {
                        "type": "choice",
                        "id": "c",
                        "instructions": "which?",
                        "options": [{"value": "a"}, {"value": "b"}],
                    },
                    {
                        "type": "score",
                        "id": "s",
                        "instructions": "rate?",
                        "levels": ["low", "high"],
                    },
                ],
                "reference": {"p": True, "c": "a", "s": 1},
            }
        )
    )
    task = DecisionJSONLTask(
        DecisionJSONLDataset(str(path)),
        None,
        provider="openrouter",
        model_name="m",
        **kwargs,
    )
    return task, task.make_context(0)


@pytest.mark.anyio
async def test_mixed_question_metrics_do_not_merge_binary_and_multiclass_brier(
    tmp_path,
):
    task, ctx = task_and_context(tmp_path)
    output = DecisionResponse(
        DecisionResult(
            (
                PredicateAnswer("p", 0.8),
                ChoiceAnswer(
                    "c", "a", (ChoiceProbability("a", 0.8), ChoiceProbability("b", 0.2))
                ),
                ScoreAnswer("s", 0.75),
            )
        ),
        "m",
    )
    finalize, feedback = await task.feedback(output, ctx)
    assert finalize
    ctx = replace(ctx, feedback_result=feedback)
    report = await task.report([ctx], [])
    assert report["accuracy"] == 1
    assert report["predicate_brier_score"] == pytest.approx(0.04)
    assert report["choice_brier_score"] == pytest.approx(0.08)
    assert report["mean_absolute_error"] == pytest.approx(0.25)
    assert report["n_questions"] == 3


@pytest.mark.anyio
async def test_refusals_are_incorrect_and_have_no_fabricated_calibration(tmp_path):
    task, ctx = task_and_context(tmp_path)
    output = DecisionResponse(
        DecisionResult(tuple(RefusalAnswer(q) for q in ("p", "c", "s"))), "m"
    )
    _, feedback = await task.feedback(output, ctx)
    ctx = replace(ctx, feedback_result=feedback)
    report = await task.report([ctx], [])
    assert report["accuracy"] == 0
    assert report["refusal_rate"] == 1
    assert report["n_refused"] == 3
    assert "answered_accuracy" not in report
    assert "predicate_brier_score" not in report
    assert "choice_brier_score" not in report


@pytest.mark.anyio
async def test_failed_samples_are_reported_without_a_false_accuracy(tmp_path):
    task, ctx = task_and_context(tmp_path)
    report = await task.report([], [ctx])
    assert report["n_failed"] == 1
    assert report["n_questions"] == 0
    assert "accuracy" not in report


@pytest.mark.parametrize(
    "kwargs",
    [
        {"predicate_threshold": -0.1},
        {"predicate_threshold": True},
        {"score_tolerance": -1},
        {"score_tolerance": float("nan")},
    ],
)
def test_invalid_grading_settings_fail_before_inference(tmp_path, kwargs):
    with pytest.raises((ValueError, TypeError)):
        task_and_context(tmp_path, **kwargs)


@pytest.mark.parametrize("name", ["require_probabilities", "require_confidence"])
@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_invalid_decision_requirements_fail_during_task_construction(
    tmp_path, name, value
):
    with pytest.raises(TypeError, match=f"{name} must be a boolean"):
        task_and_context(tmp_path, **{name: value})


@pytest.mark.anyio
@pytest.mark.parametrize("require_probabilities", [False, True])
@pytest.mark.parametrize("require_confidence", [False, True])
async def test_boolean_decision_requirements_reach_request_unchanged(
    tmp_path, require_probabilities, require_confidence
):
    task, ctx = task_and_context(
        tmp_path,
        require_probabilities=require_probabilities,
        require_confidence=require_confidence,
    )
    try:
        request = await task.preprocess(ctx.raw_sample, ctx)
        assert request.decisions.require_probabilities is require_probabilities
        assert request.decisions.require_confidence is require_confidence
        assert task.inference_args() == {
            "require_probabilities": require_probabilities,
            "require_confidence": require_confidence,
        }
    finally:
        await task.shutdown()
