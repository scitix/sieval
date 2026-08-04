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
