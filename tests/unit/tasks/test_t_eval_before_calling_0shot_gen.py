"""Tests for the t_eval before-calling task: import discipline + scored axes.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sieval.core.models import ChatModel
from sieval.core.tasks import build_prediction_record
from sieval.tasks.t_eval_before_calling_0shot_gen import (
    TEvalBeforeCallingZeroShotGenTask,
)


def _task(**kwargs):
    """A real instance -- these tests exercise the scoring methods, not mocks."""
    model = Mock(spec=ChatModel)
    model.dialect_id = "openai_chat"
    model.runtime_plan = None
    return TEvalBeforeCallingZeroShotGenTask(Mock(), model, **kwargs)


def _call(thought="search it", name="search", args=None):
    return json.dumps(
        {
            "thought": thought,
            "name": name,
            "args": {"query": "cats"} if args is None else args,
        }
    )


def _ctx(gold):
    return SimpleNamespace(
        raw_sample={"template": "", "ground_truth": gold, "meta_data": {}}
    )


class TestMetricKeys:
    """The single source of truth for which axes this configuration scores.

    `_evaluate` always returns all six axes, pre-seeded to 0, so that its mapping
    stays rectangular. A 0 on an axis nobody measured is a hole, not a
    measurement — and anything that treats it as one corrupts both `metrics` and
    the `correct` derived from it.
    """

    def test_thought_is_excluded_when_thought_scoring_is_off(self):
        # eval_thought defaults to False: the axis costs an embedding call.
        assert _task()._metric_keys() == [
            "name",
            "args_precision",
            "args_recall",
            "args_f1_score",
            "parse_rate",
        ]

    def test_thought_is_scored_when_enabled(self, monkeypatch):
        # Enabling the axis builds the embedding client in `__init__`, which
        # requires a key -- so the axis is not configurable without one. Nothing
        # is called here; only `_metric_keys`.
        monkeypatch.setenv("SIEVAL_EMBED_API_KEY", "not-a-real-key")
        assert _task(eval_thought=True)._metric_keys() == [
            "thought",
            "name",
            "args_precision",
            "args_recall",
            "args_f1_score",
            "parse_rate",
        ]

    @pytest.mark.parametrize("value", [None, ""])
    def test_thought_scoring_without_a_key_names_the_variable_that_helps(
        self, monkeypatch, value
    ):
        # The endpoint is our embedding service, so the OpenAI client's own
        # `Missing credentials` error -- which names OPENAI_API_KEY -- points at a
        # variable that would not help. Both an unset and an empty key must reach
        # sieval's message instead; "" is the reachable case, since it is what an
        # exported-but-blank variable gives.
        if value is None:
            monkeypatch.delenv("SIEVAL_EMBED_API_KEY", raising=False)
        else:
            monkeypatch.setenv("SIEVAL_EMBED_API_KEY", value)
        with pytest.raises(ValueError, match="SIEVAL_EMBED_API_KEY") as excinfo:
            _task(eval_thought=True)
        assert "OPENAI_API_KEY" not in str(excinfo.value)
        # The way out has to be in the message: the axis is optional.
        assert "eval_thought" in str(excinfo.value)

    @pytest.mark.parametrize(
        ("eval_type", "expected"),
        [
            # str mode returns one bare field, so only the axis matching
            # eval_type is populated at all. `thought` is still gated on
            # eval_thought, which is why "reason" collapses to parse_rate alone.
            ("reason", ["parse_rate"]),
            ("retrieve", ["name", "parse_rate"]),
            (
                "understand",
                ["args_precision", "args_recall", "args_f1_score", "parse_rate"],
            ),
        ],
    )
    def test_str_mode_scores_only_the_axis_matching_eval_type(
        self, eval_type, expected
    ):
        task = _task(default_prompt_type="str", eval_type=eval_type)

        assert task._metric_keys() == expected

    def test_unsupported_prompt_type_is_rejected(self):
        # Previously fell through with `metric_keys` unbound → NameError deep in
        # the macro-average, long after the bad config was accepted.
        with pytest.raises(NotImplementedError, match="json and str"):
            _task(default_prompt_type="ReWOO")._metric_keys()


class TestCorrectDerivation:
    """`correct` must be reachable under the DEFAULT configuration.

    It is derived as "every scored axis came out at 1.0". Deriving it over the
    unscored axes too pinned it to False on every sample of every default-config
    run, which silently emptied the one axis the stage-output protocol exists to
    make comparable across tasks.
    """

    def test_perfect_answer_is_correct_under_the_default_config(self):
        gold = _call()
        task = _task()

        _, record = asyncio.run(
            task.feedback(build_prediction_record([gold]), _ctx(gold))
        )

        assert record["rollouts"][0]["correct"] is True
        assert record["n_correct"] == 1

    def test_unscored_axis_is_absent_from_metrics_not_zero(self):
        gold = _call()

        _, record = asyncio.run(
            _task().feedback(build_prediction_record([gold]), _ctx(gold))
        )

        assert "thought" not in record["metrics"]
        assert set(record["metrics"]) == {
            "name",
            "args_precision",
            "args_recall",
            "args_f1_score",
            "parse_rate",
        }

    def test_wrong_tool_name_is_not_correct(self):
        gold = _call()
        wrong = _call(name="WRONG")

        _, record = asyncio.run(
            _task().feedback(build_prediction_record([wrong]), _ctx(gold))
        )

        # Discriminating: only `name` moves, so a `correct` that ignored it — or
        # one derived from parse_rate — would still report True here.
        assert record["metrics"]["name"] == 0.0
        assert record["metrics"]["parse_rate"] == 1.0
        assert record["rollouts"][0]["correct"] is False

    def test_unparseable_answer_is_not_correct(self):
        gold = _call()

        _, record = asyncio.run(
            _task().feedback(build_prediction_record(["not json at all"]), _ctx(gold))
        )

        assert record["metrics"]["parse_rate"] == 0.0
        assert record["rollouts"][0]["correct"] is False
        assert record["extra"]["parse_error"] == 1


class TestReport:
    """The macro-average reads exactly the axes `feedback` recorded."""

    def test_json_mode_macro_averages_each_axis(self):
        gold = _call()
        task = _task()
        records = [
            asyncio.run(task.feedback(build_prediction_record([pred]), _ctx(gold)))[1]
            for pred in (gold, _call(name="WRONG"))
        ]
        finals = [SimpleNamespace(feedback_result=r) for r in records]

        result = asyncio.run(task.report(finals, []))

        assert result["name"] == 50.0
        assert result["args_f1_score"] == 100.0
        assert result["parse_rate"] == 100.0
        assert result["fails"] == 0

    def test_str_mode_reports_args_parsed_without_the_args_axes_recorded(self):
        """The *_parsed variants are reported in every mode, scored or not.

        In str/reason mode `metrics` legitimately holds only `parse_rate`, so
        these three have to read defensively. Pre-fix they were 0.0 because
        `_evaluate` pre-seeded them and everything was recorded; 0.0 is therefore
        also the parity-preserving answer.
        """
        task = _task(default_prompt_type="str", eval_type="reason")
        finals = [
            SimpleNamespace(feedback_result={"metrics": {"parse_rate": 1.0}}),
            SimpleNamespace(feedback_result={"metrics": {"parse_rate": 1.0}}),
        ]

        result = asyncio.run(task.report(finals, []))

        assert result["parse_rate"] == 100.0
        assert result["args_precision_parsed"] == 0.0
        assert result["args_recall_parsed"] == 0.0
        assert result["args_f1_score_parsed"] == 0.0
