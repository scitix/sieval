"""Tests for the stage-output protocol records.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest

from sieval.core.tasks import TaskStageOutput
from sieval.core.tasks.records import (
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
    is_judgement_record,
    is_prediction_record,
    iter_grader_outputs,
)
from sieval.core.utils.serialization import dict_to_obj, obj_to_dict


class TestPromptRecord:
    def test_carries_prompt_and_omits_absent_optionals(self):
        record = build_prompt_record([{"role": "user", "content": "q"}])
        assert record == {"prompt": [{"role": "user", "content": "q"}]}

    def test_keeps_reference_and_extra_when_given(self):
        record = build_prompt_record(
            "q", reference="D", extra={"permutation": [2, 0, 1, 3]}
        )
        assert record["reference"] == "D"
        assert record["extra"] == {"permutation": [2, 0, 1, 3]}

    def test_empty_extra_is_omitted_not_stored(self):
        assert "extra" not in build_prompt_record("q", extra={})


class TestPredictionRecord:
    def test_indexes_rollouts_in_order(self):
        record = build_prediction_record(["a", "b", "c"])
        assert [r["index"] for r in record["rollouts"]] == [0, 1, 2]
        assert [r["prediction"] for r in record["rollouts"]] == ["a", "b", "c"]

    def test_extracted_tracks_none_only(self):
        record = build_prediction_record(["a", None, ""])
        assert [r["extracted"] for r in record["rollouts"]] == [True, False, True]

    def test_no_rollouts_yields_empty_list(self):
        assert build_prediction_record([]) == {"rollouts": []}

    def test_per_rollout_extras_land_on_their_own_rollout(self):
        # A second extraction RULE over the same response (GSM8K's flexible
        # match) is per-rollout detail. In the sample-level slot it silently
        # means "rollout 0's" as soon as n > 1.
        record = build_prediction_record(
            ["a", "b"], extras=[{"rule": "strict"}, {"rule": "flexible"}]
        )
        assert [r["extra"] for r in record["rollouts"]] == [
            {"rule": "strict"},
            {"rule": "flexible"},
        ]
        assert "extra" not in record

    def test_an_empty_per_rollout_extra_is_omitted_not_stored(self):
        record = build_prediction_record(["a", "b"], extras=[None, {}])
        assert all("extra" not in r for r in record["rollouts"])

    def test_misaligned_extras_are_rejected_rather_than_zipped(self):
        # Silently truncating would attach rollout 1's detail to nothing and
        # rollout 2's to rollout 1 -- wrong data, not missing data.
        with pytest.raises(ValueError, match="one entry per rollout"):
            build_prediction_record(["a", "b", "c"], extras=[{"rule": "strict"}])

    def test_sample_level_extra_still_works_alongside(self):
        record = build_prediction_record(
            ["a"], extras=[{"rule": "strict"}], extra={"subject": "algebra"}
        )
        assert record["extra"] == {"subject": "algebra"}
        assert record["rollouts"][0]["extra"] == {"rule": "strict"}


class TestJudgementRecord:
    def test_derives_counts_from_rollouts(self):
        record = build_judgement_record(
            "42",
            [
                build_rollout_judgement(0, True),
                build_rollout_judgement(1, False),
                build_rollout_judgement(2, True),
            ],
        )
        assert record["n_rollouts"] == 3
        assert record["n_correct"] == 2

    def test_counts_are_zero_for_no_rollouts(self):
        record = build_judgement_record("42", [])
        assert record["n_rollouts"] == 0
        assert record["n_correct"] == 0

    def test_reference_may_be_none_for_procedural_ground_truth(self):
        record = build_judgement_record(None, [build_rollout_judgement(0, True)])
        assert record["reference"] is None

    def test_scores_and_extras_are_optional(self):
        bare = build_judgement_record("x", [build_rollout_judgement(0, True)])
        assert "score" not in bare
        assert "extra" not in bare
        assert "score" not in bare["rollouts"][0]

        scored = build_judgement_record(
            "x",
            [build_rollout_judgement(0, True, score=0.5, extra={"why": "partial"})],
            score=0.5,
            extra={"grader": "llm"},
        )
        assert scored["score"] == 0.5
        assert scored["rollouts"][0]["score"] == 0.5
        assert scored["rollouts"][0]["extra"] == {"why": "partial"}
        assert scored["extra"] == {"grader": "llm"}

    def test_zero_score_is_kept_not_treated_as_absent(self):
        record = build_judgement_record(
            "x", [build_rollout_judgement(0, False, score=0.0)], score=0.0
        )
        assert record["score"] == 0.0
        assert record["rollouts"][0]["score"] == 0.0

    def test_accepts_any_sequence_of_rollouts(self):
        # Builders take a Sequence, so a tuple must work as well as a list, and the
        # stored value must be a list so it round-trips through JSON unchanged.
        record = build_judgement_record("x", (build_rollout_judgement(0, True),))
        assert isinstance(record["rollouts"], list)
        assert record["n_rollouts"] == 1


class TestCoEqualMetrics:
    """`metrics` is what keeps a non-headline metric readable without task knowledge."""

    # IFEval's real shape: strict and loose are both published, strict is merely
    # the reading the headline points at.
    _METRICS: dict[str, bool | float] = {
        "strict_follow_all": False,
        "strict_instruction_level": 0.5,
        "loose_follow_all": True,
        "loose_instruction_level": 1.0,
    }

    def test_records_every_metric_not_only_the_one_the_headline_points_at(self):
        record = build_judgement_record(
            ["a", "b"],
            [build_rollout_judgement(0, False, score=0.5, metrics=self._METRICS)],
            score=0.5,
            metrics=self._METRICS,
        )
        # Enumerable without knowing the task; loose is not lost behind `correct`.
        assert set(record["metrics"]) == set(self._METRICS)
        assert set(record["rollouts"][0]["metrics"]) == set(self._METRICS)
        assert record["metrics"]["loose_follow_all"] is True
        # ...while `correct`/`n_correct` stay the single cross-task axis.
        assert record["rollouts"][0]["correct"] is False
        assert record["n_correct"] == 0

    def test_metrics_are_omitted_when_absent(self):
        record = build_judgement_record("x", [build_rollout_judgement(0, True)])
        assert "metrics" not in record
        assert "metrics" not in record["rollouts"][0]

    def test_empty_metrics_are_omitted_like_every_other_optional_field(self):
        record = build_judgement_record(
            "x", [build_rollout_judgement(0, True, metrics={})], metrics={}
        )
        assert "metrics" not in record
        assert "metrics" not in record["rollouts"][0]

    def test_a_none_metric_is_rejected_rather_than_silently_dropped(self):
        # A None metric would be absent on disk: "not measured" would read as
        # "never existed". Held in a bare dict because it is deliberately
        # ill-typed -- the guard is for values that reach a task past the checker.
        none_valued: dict = {"acc_norm": None}
        with pytest.raises(ValueError, match="unmeasured|None"):
            build_rollout_judgement(0, True, metrics=none_valued)
        with pytest.raises(ValueError, match="unmeasured|None"):
            build_judgement_record(
                "x", [build_rollout_judgement(0, True)], metrics=none_valued
            )

    def test_a_structured_metric_is_rejected_and_belongs_in_extra(self):
        structured: dict = {"follow_instruction_list": [True, False]}
        with pytest.raises(ValueError, match="bool/number"):
            build_rollout_judgement(0, True, metrics=structured)

    def test_false_and_zero_metrics_survive_the_wire(self):
        # Falsy but not None, so they must reach disk -- otherwise a failed metric
        # is indistinguishable from an unrecorded one.
        record = build_judgement_record(
            "x",
            [
                build_rollout_judgement(
                    0, False, metrics={"acc": False, "acc_norm": 0.0}
                )
            ],
            metrics={"acc": False, "acc_norm": 0.0},
        )
        restored = dict_to_obj(obj_to_dict(record, False), {})
        assert restored["metrics"] == {"acc": False, "acc_norm": 0.0}
        assert restored["rollouts"][0]["metrics"] == {"acc": False, "acc_norm": 0.0}

    def test_metrics_is_a_copy_so_a_shared_dict_cannot_be_mutated_through(self):
        source: dict[str, bool | float] = {"acc": True}
        judgement = build_rollout_judgement(0, True, metrics=source)
        source["acc"] = False
        assert judgement["metrics"] == {"acc": True}


class TestRecordSniffing:
    def test_recognizes_protocol_records(self):
        assert is_prediction_record(build_prediction_record(["a"]))
        assert is_judgement_record(
            build_judgement_record("x", [build_rollout_judgement(0, True)])
        )

    def test_recognizes_empty_rollout_records(self):
        assert is_prediction_record(build_prediction_record([]))
        # n_rollouts is materialized even at zero, so an empty judgement is still
        # recognized as one — and not mistaken for a prediction.
        empty_judgement = build_judgement_record("x", [])
        assert is_judgement_record(empty_judgement)
        assert not is_prediction_record(empty_judgement)

    def test_sniffs_discriminate_prediction_from_judgement(self):
        # Both carry `rollouts`, so a check on that alone cannot tell them apart.
        # The discriminator is `n_rollouts` (judgement-only), so neither sniff
        # may fire on the other's record.
        prediction = build_prediction_record(["a"])
        judgement = build_judgement_record("x", [build_rollout_judgement(0, True)])
        assert not is_judgement_record(prediction)
        assert not is_prediction_record(judgement)

    @pytest.mark.parametrize(
        "legacy",
        [
            "C",
            ["C"],
            None,
            {"correct": True, "answer": "D"},
            [{"correct": True}],
            42,
        ],
    )
    def test_rejects_legacy_stage_values(self, legacy):
        assert not is_prediction_record(legacy)
        assert not is_judgement_record(legacy)

    def test_does_not_see_through_a_stage_output_box(self):
        # Callers that accept a box must unwrap first; the sniff deliberately does
        # not do it for them, so a boxed record cannot be mistaken for a bare one.
        boxed = TaskStageOutput(value=build_prediction_record(["a"]))
        assert not is_prediction_record(boxed)


class TestSerializationRoundTrip:
    """Records are plain dicts, so they must survive persistence untyped.

    The ``None``-dropping behaviour asserted here is why ``extracted`` and the
    ``n_*`` counts are explicit fields: they are what a reader can rely on after
    a round trip.
    """

    @staticmethod
    def _round_trip(record):
        return dict_to_obj(obj_to_dict(record, True), {})

    def test_no_type_marker_is_embedded(self):
        wire = obj_to_dict(build_prediction_record(["a"]), True)
        assert "__sieval_cls__" not in wire
        assert "__sieval_mod__" not in wire

    def test_none_prediction_is_absent_but_extracted_survives(self):
        restored = self._round_trip(build_prediction_record([None]))
        rollout = restored["rollouts"][0]
        assert "prediction" not in rollout
        assert rollout["extracted"] is False
        assert rollout.get("prediction") is None

    def test_none_reference_is_absent_but_counts_survive(self):
        restored = self._round_trip(
            build_judgement_record(None, [build_rollout_judgement(0, False)])
        )
        assert "reference" not in restored
        assert restored["n_rollouts"] == 1
        assert restored["n_correct"] == 0

    def test_false_and_zero_survive(self):
        restored = self._round_trip(
            build_judgement_record(
                "x", [build_rollout_judgement(0, False, score=0.0)], score=0.0
            )
        )
        assert restored["rollouts"][0]["correct"] is False
        assert restored["rollouts"][0]["score"] == 0.0
        assert restored["score"] == 0.0

    def test_populated_record_round_trips_unchanged(self):
        record = build_judgement_record(
            "42",
            [build_rollout_judgement(0, True, score=1.0, extra={"grade": "CORRECT"})],
            score=1.0,
            extra={"category": "physics"},
        )
        assert self._round_trip(record) == record

    def test_restored_record_is_still_recognized(self):
        restored = self._round_trip(build_prediction_record([None, "b"]))
        assert is_prediction_record(restored)


class TestIterGraderOutputs:
    """The runner reads grader calls back off a judgement record.

    `feedback` returns a record rather than a ModelOutput, so a grader's spend is
    invisible to the profiler unless it is recovered from the record itself.
    """

    @staticmethod
    def _judged(*outputs):
        return build_judgement_record(
            "gold",
            [
                build_rollout_judgement(
                    index, True, extra={"grader_output": output} if output else {}
                )
                for index, output in enumerate(outputs)
            ],
        )

    def test_returns_one_output_per_judged_rollout_in_order(self):
        record = self._judged(
            {"model": {"model": "judge"}, "usage": {"input_tokens": 1}},
            {"model": {"model": "judge"}, "usage": {"input_tokens": 2}},
        )
        assert [o["usage"]["input_tokens"] for o in iter_grader_outputs(record)] == [
            1,
            2,
        ]

    def test_rollout_without_a_grader_contributes_nothing(self):
        # aa_lcr's empty-candidate short-circuit never calls the checker, so that
        # rollout has no grader_output at all -- and must not fabricate a call.
        record = self._judged({"model": {"model": "judge"}}, None)
        assert len(iter_grader_outputs(record)) == 1

    def test_judgement_with_no_grader_at_all_is_empty(self):
        # Most tasks: a string compare or a test suite has no grader.
        record = build_judgement_record("gold", [build_rollout_judgement(0, True)])
        assert iter_grader_outputs(record) == []

    def test_non_judgement_values_are_empty(self):
        # Runs for EVERY stage of every task, so a non-record must be cheap+silent.
        assert iter_grader_outputs(build_prediction_record(["x"])) == []
        assert iter_grader_outputs("some text") == []
        assert iter_grader_outputs(None) == []

    def test_a_rollout_graded_in_several_calls_reports_all_of_them(self):
        """sysbench grades one session turn by turn and records the list.

        Flattening it here is what keeps the profiler's arithmetic right: a
        five-turn session bills five judge calls, and reading only the first --
        or skipping the rollout for not being a mapping -- loses four of them
        silently, which reads as a cheap grader rather than a miscount.
        """
        record = self._judged(
            [
                {"model": {"model": "judge"}, "usage": {"input_tokens": 1}},
                {"model": {"model": "judge"}, "usage": {"input_tokens": 2}},
            ],
            {"model": {"model": "judge"}, "usage": {"input_tokens": 3}},
        )
        assert [o["usage"]["input_tokens"] for o in iter_grader_outputs(record)] == [
            1,
            2,
            3,
        ]

    def test_a_string_grader_output_is_not_walked_as_a_sequence(self):
        # A str is a Sequence, so the list branch would otherwise yield its
        # characters -- none of them mappings, but the guard states the intent.
        record = self._judged("not an output")
        assert iter_grader_outputs(record) == []

    def test_judgement_whose_rollouts_are_not_mappings_is_empty(self):
        """A malformed `rollouts` must be walked past, not indexed into.

        `n_rollouts` is load-bearing in this fixture: without it the value is not
        a judgement at all, the walk returns at the first guard, and the
        per-rollout check this covers is never reached. A string is the sneaky
        case -- iterable, so it yields characters rather than raising, and
        dropping the guard turns those into `"n".get("extra")`.
        """
        assert iter_grader_outputs({"n_rollouts": 1, "rollouts": "not a list"}) == []
