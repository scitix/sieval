"""Tests for the t_eval before-calling task: import discipline + scored axes.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sieval.core.models import DEFAULT_REQUEST_TIMEOUT, ChatModel
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


class TestEmbeddingClient:
    def test_client_declares_the_shared_request_timeout(self, monkeypatch):
        """The thought axis reaches a model, so it owes the declared bound too.

        Built here rather than through ``connection_factory``, with the SDK's
        default as its silent fallback.
        """
        monkeypatch.setenv("SIEVAL_EMBED_API_KEY", "not-a-real-key")
        monkeypatch.setenv("SIEVAL_EMBED_API", "https://embed.example/model-api")

        with patch(
            "sieval.tasks.t_eval_before_calling_0shot_gen.AsyncOpenAI"
        ) as client_factory:
            _task(eval_thought=True)

        client_factory.assert_called_once_with(
            base_url="https://embed.example/model-api",
            api_key="not-a-real-key",
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )


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

    def test_str_mode_omits_the_parsed_axes_this_mode_never_scores(self):
        """A mode that does not score args reports no args column, suffixed or not.

        str/reason scores only `parse_rate`, so the un-suffixed axes are already
        absent; their `*_parsed` twins must go with them. Publishing
        `args_precision_parsed = 0.0` beside an omitted `args_precision` calls
        one axis both unmeasured and zero, off a `.get` default the evaluator
        never produced.
        """
        task = _task(default_prompt_type="str", eval_type="reason")
        finals = [
            SimpleNamespace(feedback_result={"metrics": {"parse_rate": 1.0}}),
            SimpleNamespace(feedback_result={"metrics": {"parse_rate": 1.0}}),
        ]

        result = asyncio.run(task.report(finals, []))

        assert result["parse_rate"] == 100.0
        # The denominator still reports -- it is a fact about the run whether or
        # not an axis was averaged over it.
        assert result["n_parsed"] == 2.0
        for axis in ("args_precision", "args_recall", "args_f1_score"):
            assert axis not in result
            assert f"{axis}_parsed" not in result

    def test_str_understand_mode_keeps_the_parsed_axes_it_does_score(self):
        """The other half of the gate: str/understand DOES score args.

        Discriminates against a fix that drops the triple in every str mode
        rather than only where the axis is unscored.
        """
        task = _task(default_prompt_type="str", eval_type="understand")
        finals = [
            SimpleNamespace(
                feedback_result={
                    "metrics": {
                        "args_precision": 1.0,
                        "args_recall": 0.0,
                        "args_f1_score": 0.0,
                        "parse_rate": 1.0,
                    }
                }
            )
        ]

        result = asyncio.run(task.report(finals, []))

        assert result["args_precision"] == 100.0
        assert result["args_precision_parsed"] == 100.0
        assert result["args_recall_parsed"] == 0.0

    def test_no_judged_samples_omits_every_axis_instead_of_averaging_nothing(self):
        """`np.mean([])` is nan, and a nan reaches report.json as `null`.

        A `null` cannot be told apart from a measured value that failed to
        serialise, so the axis is omitted and its denominator reported instead.
        """
        result = asyncio.run(_task().report([], []))

        for axis in ("name", "args_precision", "args_f1_score", "parse_rate"):
            assert axis not in result
        for axis in ("args_precision_parsed", "args_recall_parsed"):
            assert axis not in result
        assert result["n_graded"] == 0.0
        assert result["n_parsed"] == 0.0
        # The declarations survive the empty path (`.claude/rules/tasks.md`).
        assert result["denominator_policy"] == "judged"

    def test_zero_parseable_answers_omits_only_the_parsed_axes(self):
        """Samples graded, not one of them parseable -- the two zeros differ.

        The macro axes were measured over those samples and are a real zero; the
        `*_parsed` triple has an empty denominator and is not. Reporting the
        second as 0.0 claims zero precision on calls the model never emitted.
        """
        task = _task()
        finals = [
            SimpleNamespace(
                feedback_result={
                    "metrics": {
                        "name": 0.0,
                        "args_precision": 0.0,
                        "args_recall": 0.0,
                        "args_f1_score": 0.0,
                        "parse_rate": 0.0,
                    }
                }
            )
            for _ in range(5)
        ]

        result = asyncio.run(task.report(finals, []))

        assert result["parse_rate"] == 0.0
        assert result["args_precision"] == 0.0
        # Core claim first: a later `KeyError` on a denominator must not pre-empt
        # the omission this test exists to pin.
        for axis in (
            "args_precision_parsed",
            "args_recall_parsed",
            "args_f1_score_parsed",
        ):
            assert axis not in result
        assert result["n_graded"] == 5.0
        assert result["n_parsed"] == 0.0

    def test_every_report_path_is_strict_json(self):
        """No path may emit a nan: `allow_nan=False` is the reader's contract."""
        task = _task()
        unparseable = SimpleNamespace(
            feedback_result={"metrics": dict.fromkeys(task._metric_keys(), 0.0)}
        )

        for finals in ([], [unparseable], [unparseable, unparseable]):
            json.dumps(asyncio.run(task.report(finals, [])), allow_nan=False)
