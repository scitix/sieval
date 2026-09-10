"""
Tests for sieval.core.utils.meta — build_model_call_meta, build_stage_meta.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import time

from sieval.core.models import ModelMeta, ModelOutput
from sieval.core.tasks.consts import TaskStage
from sieval.core.utils.meta import (
    build_model_call_meta,
    build_model_call_meta_from_mapping,
    build_stage_meta,
    collect_versions,
    count_scored_rollouts,
    count_truncated_rollouts,
    report_versions,
)


class TestBuildModelCallMeta:
    def test_full_output(self, sample_model_output):
        meta = build_model_call_meta(sample_model_output)
        assert meta["model"] == sample_model_output.model
        assert meta["usage"]["input_tokens"] == 100
        assert meta["request_params"]["temperature"] == 0.7
        assert meta["finish_reasons"] == ["stop"]

    def test_minimal_output(self, sample_model_meta):
        output = ModelOutput(model=sample_model_meta, texts=["hi"])
        meta = build_model_call_meta(output)
        assert meta["model"] == sample_model_meta
        assert "usage" not in meta
        assert "request_params" not in meta
        assert "finish_reasons" not in meta

    def test_with_usage_only(self, sample_model_meta, sample_usage):
        output = ModelOutput(model=sample_model_meta, texts=["hi"], usage=sample_usage)
        meta = build_model_call_meta(output)
        assert meta["usage"]["total_tokens"] == 150
        assert "request_params" not in meta

    def test_request_params_copied(self, sample_model_meta):
        params = {"temperature": 0.5}
        output = ModelOutput(
            model=sample_model_meta, texts=["hi"], request_params=params
        )
        meta = build_model_call_meta(output)
        # Should be a copy, not the same dict
        assert meta["request_params"] == params
        assert meta["request_params"] is not params

    def test_with_response_metadata(self):
        """build_model_call_meta includes response_model and system_fingerprint."""
        output = ModelOutput(
            model={"model": "qwen", "api_base": None, "default_params": {}},
            texts=["hi"],
            response_model="Qwen/Qwen3-4B",
            system_fingerprint="fp_xyz",
        )
        meta = build_model_call_meta(output)
        assert meta["response_model"] == "Qwen/Qwen3-4B"
        assert meta["system_fingerprint"] == "fp_xyz"

    def test_without_response_metadata(self):
        """build_model_call_meta omits None response metadata."""
        output = ModelOutput(
            model={"model": "qwen", "api_base": None, "default_params": {}},
            texts=["hi"],
        )
        meta = build_model_call_meta(output)
        assert "response_model" not in meta
        assert "system_fingerprint" not in meta


class TestBuildStageMeta:
    def test_no_outputs(self):
        meta = build_stage_meta()
        assert "timestamp" in meta
        assert "model_calls" not in meta
        assert "timing_s" not in meta

    def test_with_timing(self):
        meta = build_stage_meta(timing_s=1.5)
        assert meta["timing_s"] == 1.5

    def test_single_output(self, sample_model_output):
        meta = build_stage_meta(sample_model_output)
        assert len(meta["model_calls"]) == 1
        assert meta["model_calls"][0]["model"] == sample_model_output.model

    def test_multiple_outputs(self, sample_model_meta, sample_usage):
        out1 = ModelOutput(model=sample_model_meta, texts=["a"], usage=sample_usage)
        out2 = ModelOutput(model=sample_model_meta, texts=["b"])
        meta = build_stage_meta(out1, out2)
        assert len(meta["model_calls"]) == 2
        assert "usage" in meta["model_calls"][0]
        assert "usage" not in meta["model_calls"][1]

    def test_with_extra(self, sample_model_output):
        meta = build_stage_meta(sample_model_output, extra={"custom_key": "value"})
        assert meta["extra"] == {"custom_key": "value"}

    def test_empty_extra_not_included(self):
        meta = build_stage_meta(extra={})
        assert "extra" not in meta

    def test_timestamp_is_recent(self):
        before = time.time()
        meta = build_stage_meta()
        after = time.time()
        assert "timestamp" in meta, "timestamp key missing from meta"
        assert isinstance(meta["timestamp"], (int, float)), (
            f"timestamp must be numeric, got {type(meta['timestamp'])}"
        )
        assert before <= meta["timestamp"] <= after

    def test_full_combo(self, sample_model_output):
        meta = build_stage_meta(
            sample_model_output, timing_s=2.0, extra={"note": "test"}
        )
        assert meta["timing_s"] == 2.0
        assert len(meta["model_calls"]) == 1
        assert meta["extra"]["note"] == "test"
        assert "timestamp" in meta

    def test_includes_version(self):
        from sieval import __version__

        meta = build_stage_meta()
        assert meta["version"] == __version__

    def test_version_present_with_outputs(self, sample_model_output):
        from sieval import __version__

        meta = build_stage_meta(sample_model_output, timing_s=1.0)
        assert meta["version"] == __version__


class TestCollectVersions:
    def test_empty_input(self):
        assert collect_versions([]) == []

    def test_single_version_deduped(self):
        sm = {"infer": [{"version": "0.6.0"}], "postprocess": [{"version": "0.6.0"}]}
        assert collect_versions([sm]) == ["0.6.0"]

    def test_blended_sorted_semver(self):
        sm1 = {"infer": [{"version": "0.6.10"}]}
        sm2 = {"infer": [{"version": "0.6.2"}]}
        assert collect_versions([sm1, sm2]) == ["0.6.2", "0.6.10"]

    def test_missing_version_ignored(self):
        assert collect_versions([{"infer": [{"timestamp": 1.0}]}]) == []

    def test_unparseable_sorts_after_valid(self):
        sm = {"a": [{"version": "0.6.0"}, {"version": "weird"}]}
        assert collect_versions([sm]) == ["0.6.0", "weird"]


class TestReportVersions:
    def test_all_stamped_no_sentinel(self):
        finals = [{"infer": [{"version": "0.6.0"}]}, {"infer": [{"version": "0.6.0"}]}]
        assert report_versions(finals, []) == ["0.6.0"]

    def test_empty_inputs(self):
        assert report_versions([], []) == []

    def test_unstamped_final_appends_unknown(self):
        finals = [{"infer": [{"version": "0.7.0"}]}, {"feedback": [{"timestamp": 1.0}]}]
        assert report_versions(finals, []) == ["0.7.0", "unknown"]

    def test_fully_legacy_finals_is_unknown_only(self):
        finals = [{}, {"infer": [{"timestamp": 1.0}]}]
        assert report_versions(finals, []) == ["unknown"]

    def test_unstamped_failed_does_not_add_sentinel(self):
        # A FAILED record with no version is legitimate (failed before any
        # stage produced versioned work) — only unstamped FINALs are legacy.
        finals = [{"infer": [{"version": "0.7.0"}]}]
        fails = [{}]
        assert report_versions(finals, fails) == ["0.7.0"]

    def test_blended_sorted_then_unknown_last(self):
        finals = [
            {"infer": [{"version": "0.6.0"}]},
            {"infer": [{"version": "0.6.10"}]},
            {},  # legacy, pre-provenance
        ]
        assert report_versions(finals, []) == ["0.6.0", "0.6.10", "unknown"]


def _model_meta(name: str) -> ModelMeta:
    """A minimally-complete ModelMeta (all three required keys)."""
    return {"model": name, "api_base": None, "default_params": {}}


class TestBuildModelCallMetaFromMapping:
    """Rebuilding a call from a grader's already-flattened ModelOutput.

    This is the only way grader spend reaches the profiler: `feedback` returns a
    judgement record, not a ModelOutput, so the runner cannot derive the call
    from the stage value the way it does for `infer`.
    """

    def test_round_trips_the_fields_a_profiler_reads(self):
        flattened = {
            "model": _model_meta("judge-5.2"),
            "usage": {"input_tokens": 120, "output_tokens": 7},
            "request_params": {"temperature": 0},
            "finish_reasons": ["stop"],
            "response_model": "judge-5.2-2026",
            "system_fingerprint": "fp_1",
            "texts": ["CORRECT"],  # not a call field; must not leak through
        }
        call = build_model_call_meta_from_mapping(flattened)
        assert call == {
            "model": _model_meta("judge-5.2"),
            "usage": {"input_tokens": 120, "output_tokens": 7},
            "request_params": {"temperature": 0},
            "finish_reasons": ["stop"],
            "response_model": "judge-5.2-2026",
            "system_fingerprint": "fp_1",
        }

    def test_mapping_without_model_is_not_a_call(self):
        # `model` is the one field every call has. Without it this is some other
        # dict in `extra`, and admitting it would add a usage-less phantom call.
        assert (
            build_model_call_meta_from_mapping({"usage": {"input_tokens": 1}}) is None
        )

    def test_absent_optional_fields_are_omitted_not_nulled(self):
        call = build_model_call_meta_from_mapping({"model": _model_meta("m")})
        assert call == {"model": _model_meta("m")}


class TestBuildStageMetaModelCalls:
    def test_extra_model_calls_append_after_output_derived_ones(self):
        # A stage can both return a ModelOutput and have called a grader.
        output = ModelOutput(model=_model_meta("candidate"), texts=["x"])
        meta = build_stage_meta(
            output,
            model_calls=[{"model": _model_meta("judge")}],
        )
        assert [c["model"]["model"] for c in meta["model_calls"]] == [
            "candidate",
            "judge",
        ]

    def test_only_extra_model_calls_still_records_them(self):
        # The judge-family case: the stage value is a record, so there is no
        # output to derive from -- the grader call must still be recorded.
        meta = build_stage_meta(model_calls=[{"model": _model_meta("judge")}])
        assert meta["model_calls"] == [{"model": _model_meta("judge")}]

    def test_no_calls_omits_the_key(self):
        assert "model_calls" not in build_stage_meta(timing_s=1.0)


def _inferred(*calls: dict) -> dict:
    """A stage_meta whose INFERRED history is one entry with *calls*."""
    return {TaskStage.INFERRED.value: [{"model_calls": list(calls)}]}


class TestCountTruncatedRollouts:
    """The report-level count of rollouts that ran out of budget.

    Semantics are pinned against `detect_truncated_output`, which reduces the
    same event off the live output rather than the persisted meta: the two must
    agree on what "one truncation" is, or the anomaly file and the report will
    disagree with nothing to catch it.
    """

    def test_no_records(self):
        assert count_truncated_rollouts([]) == 0

    def test_unmeasurable_record_is_not_zero(self):
        # No INFERRED history means the finish reasons were never recorded, not
        # that nothing was truncated. The reachable cause is a resume under
        # `record_meta=False`, which hydrates a final with no stage_meta; the two
        # causes the absence *looks* like -- a ppl/clp task, a sample that failed
        # before infer -- cannot arrive here, since the caller passes finals only
        # and gates on `gen`. Reducing it to 0 would report a clean run.
        assert count_truncated_rollouts([{}]) is None
        assert count_truncated_rollouts([{"final": [{"version": "0.7.0"}]}]) is None

    def test_one_unmeasurable_record_forfeits_the_whole_count(self):
        # A partially-hydrated resume: some finals carry their meta, some do not.
        # A count over just the measurable ones reads low with nothing saying so,
        # which is the failure mode this key exists to remove.
        truncated = _inferred({"finish_reasons": ["length"]})
        assert count_truncated_rollouts([truncated]) == 1
        assert count_truncated_rollouts([truncated, {}]) is None

    def test_natural_stop_is_not_truncation(self):
        assert count_truncated_rollouts([_inferred({"finish_reasons": ["stop"]})]) == 0

    def test_counts_rollouts_not_samples(self):
        # One truncated draw in four is a different fact from four.
        one = _inferred({"finish_reasons": ["stop", "length", "stop", "stop"]})
        four = _inferred({"finish_reasons": ["length"] * 4})
        assert count_truncated_rollouts([one]) == 1
        assert count_truncated_rollouts([one, four]) == 5

    def test_every_provider_spelling_counts(self):
        # The IR keeps finish_reasons provider-verbatim: OpenAI-compatible
        # Chat/Completions and SGLang native say `length`, Responses says
        # `max_output_tokens`, and Anthropic says `max_tokens`. A set holding only
        # one spelling would silently read zero on another provider.
        for reason in (
            "length",
            "max_output_tokens",
            "max_tokens",
            "content_filter",
        ):
            assert (
                count_truncated_rollouts([_inferred({"finish_reasons": [reason]})]) == 1
            )

    def test_same_rollout_truncated_on_two_turns_counts_once(self):
        # A multi-turn stage makes several calls per rollout. Rollout 0 hit the
        # cap on both turns; that is one truncated rollout, not two.
        multi_turn = _inferred(
            {"finish_reasons": ["length", "stop"]},
            {"finish_reasons": ["length", "stop"]},
        )
        assert count_truncated_rollouts([multi_turn]) == 1

    def test_union_across_calls_of_one_stage(self):
        # Rollout 0 truncated on turn 1, rollout 1 on turn 2: two rollouts.
        multi_turn = _inferred(
            {"finish_reasons": ["length", "stop"]},
            {"finish_reasons": ["stop", "length"]},
        )
        assert count_truncated_rollouts([multi_turn]) == 2

    def test_only_the_scored_attempt_is_counted(self):
        # A retried sample keeps its earlier attempts in the stage history, but
        # was scored on the last one. Counting a superseded truncation would
        # report one that no longer affects any number in the report.
        retried = {
            TaskStage.INFERRED.value: [
                {"model_calls": [{"finish_reasons": ["length"]}]},
                {"model_calls": [{"finish_reasons": ["stop"]}]},
            ]
        }
        assert count_truncated_rollouts([retried]) == 0

    def test_grader_truncation_is_not_the_models(self):
        # FEEDBACK carries the *grader's* calls. A judge that hit its own budget
        # is a fact about a different model, already reported separately as
        # `n_grader_unparsed` -- charging it to the candidate would inflate a
        # judged lane's truncation rate with someone else's.
        judged = {
            TaskStage.INFERRED.value: [{"model_calls": [{"finish_reasons": ["stop"]}]}],
            TaskStage.FEEDBACK.value: [
                {"model_calls": [{"finish_reasons": ["length"]}]}
            ],
        }
        assert count_truncated_rollouts([judged]) == 0

    def test_calls_without_finish_reasons_are_skipped(self):
        # `build_model_call_meta` omits the key when the provider sent nothing.
        # The stage still ran, so this is measured-and-zero, not unmeasurable.
        assert count_truncated_rollouts([_inferred({"model": _model_meta("m")})]) == 0
        assert count_truncated_rollouts([{TaskStage.INFERRED.value: [{}]}]) == 0

    def test_empty_stage_history_is_unmeasurable(self):
        # An INFERRED key whose history is empty records nothing about the stage,
        # which is the `{}` case spelled differently -- not a measured zero.
        assert count_truncated_rollouts([{TaskStage.INFERRED.value: []}]) is None


class TestCountScoredRollouts:
    """`n_truncated`'s denominator: the rollouts a report actually scored.

    Without it the numerator is unreadable -- the lanes this was built for
    publish rates plus `fails` and no sample total, so `report.json` carried
    nothing to divide by.
    """

    def test_no_records(self):
        assert count_scored_rollouts([]) == 0

    def test_counts_the_draw_not_the_samples(self):
        four = _inferred({"finish_reasons": ["stop"] * 4})
        assert count_scored_rollouts([four]) == 4
        assert count_scored_rollouts([four, four]) == 8

    def test_multi_turn_counts_each_rollout_once(self):
        # Same union as the numerator: the list dimension is extra calls, the
        # index dimension is rollouts. Two turns of a 2-rollout draw is 2.
        multi_turn = _inferred(
            {"finish_reasons": ["stop", "stop"]},
            {"finish_reasons": ["stop", "length"]},
        )
        assert count_scored_rollouts([multi_turn]) == 2

    def test_observed_draw_not_the_budget(self):
        # A short sample drew fewer rollouts than the budget asked for. The rate
        # a reader wants is over what actually ran, so the base is the observed
        # width -- `n * len(finals)` would understate the truncation share.
        short = _inferred({"finish_reasons": ["length"]})
        full = _inferred({"finish_reasons": ["stop"] * 4})
        assert count_scored_rollouts([short, full]) == 5
        assert count_truncated_rollouts([short, full]) == 1

    def test_only_the_scored_attempt_is_counted(self):
        # Same last-entry rule as the numerator, or a retried sample would
        # inflate the base and dilute the rate.
        retried = {
            TaskStage.INFERRED.value: [
                {"model_calls": [{"finish_reasons": ["stop"] * 4}]},
                {"model_calls": [{"finish_reasons": ["stop"] * 2}]},
            ]
        }
        assert count_scored_rollouts([retried]) == 2

    def test_grader_calls_are_not_part_of_the_base(self):
        judged = {
            TaskStage.INFERRED.value: [{"model_calls": [{"finish_reasons": ["stop"]}]}],
            TaskStage.FEEDBACK.value: [
                {"model_calls": [{"finish_reasons": ["stop"] * 8}]}
            ],
        }
        assert count_scored_rollouts([judged]) == 1

    def test_unmeasurable_record_is_not_zero(self):
        assert count_scored_rollouts([{}]) is None
        assert (
            count_scored_rollouts([_inferred({"finish_reasons": ["stop"]}), {}]) is None
        )

    def test_numerator_never_exceeds_the_base(self):
        # The invariant that makes the pair a fraction: both are reduced off the
        # same index space, so a rate can never come out above 1.
        cases = [
            _inferred({"finish_reasons": ["length", "stop", "max_tokens"]}),
            _inferred(
                {"finish_reasons": ["length", "stop"]},
                {"finish_reasons": ["stop", "content_filter"]},
            ),
            _inferred({"finish_reasons": ["length"] * 4}),
            _inferred({"model": _model_meta("m")}),
        ]
        for case in cases:
            truncated = count_truncated_rollouts([case])
            scored = count_scored_rollouts([case])
            assert truncated is not None and scored is not None
            assert truncated <= scored
