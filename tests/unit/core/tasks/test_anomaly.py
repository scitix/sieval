"""
Tests for sieval.core.tasks.anomaly — detection rules and TaskAnomalyDetector.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

from dataclasses import replace
from pathlib import Path

import orjson
import pytest

from sieval.core.models import ModelOutput
from sieval.core.tasks.anomaly import (
    _DETECTION_RULES,
    TaskAnomalyDetector,
    _model_outputs,
    _rule_applies,
    _unwrap_result,
    detect_empty_infer_gen,
    detect_empty_infer_ppl,
    detect_empty_postprocess,
    detect_extraction_failure,
    detect_truncated_output,
    get_applied_rules,
    get_rules_by_category,
    get_rules_hash,
    get_rules_schema,
    sieval_detection_rule,
)
from sieval.core.tasks.context import TaskContext, TaskStageOutput
from sieval.core.tasks.records import build_prediction_record
from sieval.core.utils.serialization import dict_to_obj, obj_to_dict


@pytest.fixture(autouse=True)
def _isolate_detection_rules():
    """Save and restore _DETECTION_RULES to prevent cross-test pollution."""
    snapshot = dict(_DETECTION_RULES)
    yield
    _DETECTION_RULES.clear()
    _DETECTION_RULES.update(snapshot)


def _make_final_ctx(sample_id: int = 0, iteration: int = 0, **kwargs) -> TaskContext:
    """Create a FINAL-stage context with given fields."""
    ctx = TaskContext(sample_id=sample_id, raw_sample={})
    ctx = replace(ctx, iteration=iteration)
    ctx = ctx.to_preprocessed(kwargs.get("preprocess_result", "pre"))
    ctx = ctx.to_inferred(kwargs.get("infer_result", "inf"))
    ctx = ctx.to_postprocessed(kwargs.get("postprocess_result", "post"))
    ctx = ctx.to_feedback(kwargs.get("feedback_result", "fb"))
    ctx = ctx.to_final()
    return ctx


class TestUnwrapResult:
    def test_unwrap_behaviors(self):
        assert _unwrap_result("hello") == "hello"
        assert _unwrap_result(42) == 42
        assert _unwrap_result(None) is None

        tso = TaskStageOutput(value="answer")
        assert _unwrap_result(tso) == "answer"


class TestModelOutputs:
    """A multi-turn `infer` returns `list[ModelOutput]`; rules must still see it."""

    def test_shapes(self, sample_model_meta):
        one = ModelOutput(model=sample_model_meta, texts=["a"])
        two = ModelOutput(model=sample_model_meta, texts=["b"])
        cases = [
            (one, (one,), "single"),
            ([one, two], (one, two), "list_of_outputs"),
            ([], (), "empty_list"),
            ("not_an_output", (), "wrong_type"),
            ([one, "mixed"], (), "mixed_list_is_not_a_multi_call_stage"),
            ({"rollouts": []}, (), "prediction_record"),
            (None, (), "none"),
        ]
        for result, expected, case_name in cases:
            assert _model_outputs(result) == expected, case_name


class TestDetectEmptyInferGen:
    def test_variants(self, sample_model_meta):
        wrapped_empty = TaskStageOutput(
            value=ModelOutput(model=sample_model_meta, texts=[])
        )
        # (infer_result, expected_indices, case_name)
        cases = [
            (ModelOutput(model=sample_model_meta, texts=[]), {0}, "empty_texts"),
            (ModelOutput(model=sample_model_meta, texts=["hello"]), set(), "non_empty"),
            ("some_string", set(), "non_model_output"),
            (wrapped_empty, {0}, "wrapped_empty"),
            (None, set(), "none"),
        ]
        for infer_result, expected, case_name in cases:
            ctx = _make_final_ctx(infer_result=infer_result)
            assert detect_empty_infer_gen(ctx) == expected, case_name

    def test_multi_turn_reports_a_single_empty_call(self, sample_model_meta):
        """One dead turn out of three is the same signal as a dead single call.

        Before rules read a list, this returned an empty set -- indistinguishable
        from "nothing was wrong".
        """
        answered = ModelOutput(model=sample_model_meta, texts=["ok"])
        blank = ModelOutput(model=sample_model_meta, texts=[])
        cases = [
            ([answered, answered, answered], set(), "every_turn_answered"),
            ([answered, blank, answered], {0}, "middle_turn_empty"),
            ([blank, blank, blank], {0}, "every_turn_empty"),
            ([answered, answered], set(), "two_turn_conversation"),
        ]
        for infer_result, expected, case_name in cases:
            ctx = _make_final_ctx(infer_result=infer_result)
            assert detect_empty_infer_gen(ctx) == expected, case_name


class TestDetectEmptyInferPpl:
    def test_logprobs_variants(self, sample_model_meta):
        def _make_output(logprobs, logprobs_tokens):
            return ModelOutput(
                model=sample_model_meta,
                texts=["x"],
                logprobs=logprobs,
                logprobs_tokens=logprobs_tokens,
            )

        cases = [
            (_make_output([], ["a"]), {0}, "empty_logprobs"),
            (_make_output([-0.5], []), {0}, "empty_logprobs_tokens"),
            (_make_output([-0.5], ["a"]), set(), "non_empty_ppl_fields"),
            (_make_output(None, None), set(), "no_ppl_fields"),
            ("not_model_output", set(), "non_model_output"),
            (None, set(), "none"),
        ]
        for infer_result, expected, case_name in cases:
            ctx = _make_final_ctx(infer_result=infer_result)
            assert detect_empty_infer_ppl(ctx) == expected, case_name

    def test_multi_call_reports_any_empty_candidate(self, sample_model_meta):
        """One call per candidate is the only shape a ppl task ships.

        Reading only a lone `ModelOutput` returned an empty set for every ppl
        sample. It matters more than for the `gen` siblings: `postprocess` takes an
        argmax across candidates, and an empty one sums to 0.0 where a real one is
        negative, so it wins and changes the answer instead of failing.
        """

        def cand(logprobs, logprobs_tokens):
            return ModelOutput(
                model=sample_model_meta,
                texts=["x"],
                logprobs=logprobs,
                logprobs_tokens=logprobs_tokens,
            )

        ok = cand([-0.5], ["a"])
        empty = cand([], ["a"])
        empty_tokens = cand([-0.5], [])
        unscored = cand(None, None)
        cases = [
            ([ok, ok, ok, ok], set(), "every_candidate_scored"),
            ([ok, empty, ok], {0}, "middle_candidate_empty"),
            ([ok, ok, empty_tokens], {0}, "last_candidate_empty_tokens"),
            ([empty, ok], {0}, "first_candidate_empty"),
            # A candidate carrying neither field is not an empty one: `ppl` only
            # claims a verdict where the fields exist, as the single-output arm
            # has always done.
            ([unscored, ok], set(), "candidate_without_ppl_fields"),
            ([], set(), "no_calls"),
        ]
        for infer_result, expected, case_name in cases:
            ctx = _make_final_ctx(infer_result=infer_result)
            assert detect_empty_infer_ppl(ctx) == expected, case_name


class TestDetectTruncatedOutput:
    def test_prompt_block_is_never_output_truncation(self, sample_model_meta):
        output = ModelOutput(
            model=sample_model_meta,
            texts=[""],
            finish_reasons=["content_filter"],
            finish_reason_sources=["promptFeedback.blockReason"],
        )
        assert detect_truncated_output(_make_final_ctx(infer_result=output)) == set()

    def test_google_candidate_filter_not_prompt_block(self, sample_model_meta):
        for reason in ("SAFETY", "LANGUAGE"):
            for source, expected in (
                ("candidate.finishReason", {0}),
                ("promptFeedback.blockReason", set()),
                (None, set()),
            ):
                output = ModelOutput(
                    model=sample_model_meta,
                    texts=[""],
                    finish_reasons=[reason],
                    finish_reason_sources=[source] if source is not None else None,
                )
                assert (
                    detect_truncated_output(_make_final_ctx(infer_result=output))
                    == expected
                ), (reason, source)

    def test_single_sample(self, sample_model_meta):
        """Single-sample (n=1) finish reason variants."""
        cases = [
            (["length"], {0}, "length"),
            (["max_output_tokens"], {0}, "max_output_tokens"),
            (["max_tokens"], {0}, "max_tokens"),
            (
                ["model_context_window_exceeded"],
                {0},
                "model_context_window_exceeded",
            ),
            (["MAX_TOKENS"], {0}, "MAX_TOKENS"),
            (["content_filter"], {0}, "content_filter"),
            (["refusal"], {0}, "refusal"),
            (["end_turn"], set(), "end_turn"),
            (["stop"], set(), "stop"),
            (None, set(), "missing_finish_reasons"),
        ]
        for finish_reasons, expected, case_name in cases:
            output = ModelOutput(
                model=sample_model_meta, texts=["x"], finish_reasons=finish_reasons
            )
            ctx = _make_final_ctx(infer_result=output)
            assert detect_truncated_output(ctx) == expected, case_name

    def test_multi_sample_partial_truncation(self, sample_model_meta):
        """n>1: only the truncated samples are reported by index."""
        output = ModelOutput(
            model=sample_model_meta,
            texts=["a", "b", "c"],
            finish_reasons=["stop", "length", "stop"],
        )
        ctx = _make_final_ctx(infer_result=output)
        assert detect_truncated_output(ctx) == {1}

    def test_multi_sample_all_truncated(self, sample_model_meta):
        output = ModelOutput(
            model=sample_model_meta,
            texts=["a", "b"],
            finish_reasons=["max_tokens", "length"],
        )
        ctx = _make_final_ctx(infer_result=output)
        assert detect_truncated_output(ctx) == {0, 1}

    def test_multi_turn_reports_whichever_turn_hit_the_cap(self, sample_model_meta):
        """A conversation carrying its own history is the likeliest to truncate.

        Before rules read a list this returned an empty set for every turn, so the
        signal was simply absent for multi-turn tasks.
        """

        def turn(reason):
            return ModelOutput(
                model=sample_model_meta, texts=["x"], finish_reasons=[reason]
            )

        cases = [
            ([turn("stop"), turn("stop")], set(), "nothing_truncated"),
            ([turn("stop"), turn("length"), turn("stop")], {0}, "middle_turn"),
            ([turn("stop"), turn("stop"), turn("max_tokens")], {0}, "last_turn"),
            ([turn("content_filter"), turn("stop")], {0}, "filtered_first_turn"),
        ]
        for infer_result, expected, case_name in cases:
            ctx = _make_final_ctx(infer_result=infer_result)
            assert detect_truncated_output(ctx) == expected, case_name

    def test_multi_turn_keeps_rollout_indices_not_turn_indices(self, sample_model_meta):
        """`len(indices)` counts rollouts, so the turn dimension must not leak in.

        Two turns, n=2 each, truncated at *different* rollout positions: the union
        is over rollout index, so this is {0, 1} -- not {0, 1, 2} (three turns) and
        not {1} (last turn wins).
        """
        first = ModelOutput(
            model=sample_model_meta,
            texts=["a", "b"],
            finish_reasons=["length", "stop"],
        )
        second = ModelOutput(
            model=sample_model_meta,
            texts=["c", "d"],
            finish_reasons=["stop", "length"],
        )
        ctx = _make_final_ctx(infer_result=[first, second])
        assert detect_truncated_output(ctx) == {0, 1}


class TestDetectEmptyPostprocess:
    def test_postprocess_variants(self):
        cases = [
            ("", {0}, "empty"),
            ("   ", {0}, "whitespace"),
            ("answer", set(), "non_empty"),
            (None, {0}, "none"),
            (0, set(), "int_zero"),
            (0.0, set(), "float_zero"),
            (False, set(), "bool_false"),
            (1, set(), "int_one"),
            ([], {0}, "empty_list"),
            ({}, {0}, "empty_dict"),
        ]
        for postprocess_result, expected, case_name in cases:
            ctx = _make_final_ctx(postprocess_result=postprocess_result)
            assert detect_empty_postprocess(ctx) == expected, case_name

    def test_defers_on_protocol_records(self):
        # A PredictionRecord is a non-empty dict whatever it holds, so this rule
        # must stay silent and let detect_extraction_failure read it properly --
        # otherwise every protocol task double-reports or reports nothing useful.
        for values in ([None], ["C"], []):
            ctx = _make_final_ctx(postprocess_result=build_prediction_record(values))
            assert detect_empty_postprocess(ctx) == set(), values

    def test_defers_on_a_boxed_protocol_record(self):
        ctx = _make_final_ctx(
            postprocess_result=TaskStageOutput(value=build_prediction_record([None]))
        )
        assert detect_empty_postprocess(ctx) == set()


class TestDetectExtractionFailure:
    def test_reports_only_the_rollouts_that_failed(self):
        ctx = _make_final_ctx(
            postprocess_result=build_prediction_record(["a", None, "c", None])
        )
        assert detect_extraction_failure(ctx) == {1, 3}

    def test_silent_when_every_rollout_extracted(self):
        ctx = _make_final_ctx(postprocess_result=build_prediction_record(["a", "b"]))
        assert detect_extraction_failure(ctx) == set()

    def test_reports_all_rollouts_when_all_failed(self):
        ctx = _make_final_ctx(postprocess_result=build_prediction_record([None, None]))
        assert detect_extraction_failure(ctx) == {0, 1}

    def test_a_record_with_no_rollouts_is_itself_an_anomaly(self):
        ctx = _make_final_ctx(postprocess_result=build_prediction_record([]))
        assert detect_extraction_failure(ctx) == {0}

    def test_reads_through_a_stage_output_box(self):
        ctx = _make_final_ctx(
            postprocess_result=TaskStageOutput(
                value=build_prediction_record(["a", None])
            )
        )
        assert detect_extraction_failure(ctx) == {1}

    def test_honours_explicit_rollout_indices_over_position(self):
        # Indices are materialized in the record, so a filtered or reordered
        # rollout list must still report the rollout's own index.
        record = {
            "rollouts": [
                {"index": 5, "prediction": None, "extracted": False},
                {"index": 9, "prediction": "x", "extracted": True},
            ]
        }
        ctx = _make_final_ctx(postprocess_result=record)
        assert detect_extraction_failure(ctx) == {5}

    @pytest.mark.parametrize("legacy", ["", "C", [], ["C"], None])
    def test_ignores_legacy_stage_values(self, legacy):
        # Non-migrated tasks stay the exclusive business of detect_empty_postprocess.
        ctx = _make_final_ctx(postprocess_result=legacy)
        assert detect_extraction_failure(ctx) == set()

    def test_survives_a_round_trip_through_disk(self):
        restored = dict_to_obj(
            obj_to_dict(build_prediction_record([None, "b"]), True), {}
        )
        ctx = _make_final_ctx(postprocess_result=restored)
        assert detect_extraction_failure(ctx) == {0}


class TestDetectionRuleRegistry:
    def test_builtin_registry_and_schema_contents(self):
        rules = get_applied_rules()
        assert "empty_infer_gen" in rules
        assert "empty_infer_ppl" in rules
        assert "truncated_output" in rules
        assert "empty_postprocess" in rules
        assert "extraction_failure" in rules

        schema = get_rules_schema()
        assert schema["version"] == "1.0"
        rule_names = {r["name"] for r in schema["rules"]}
        assert "empty_infer_gen" in rule_names
        assert "empty_infer_ppl" in rule_names
        assert "truncated_output" in rule_names
        assert "empty_postprocess" in rule_names
        assert "extraction_failure" in rule_names

        cats = get_rules_by_category()
        assert "output_quality" in cats
        assert "correctness" in cats

        h1 = get_rules_hash()
        h2 = get_rules_hash()
        assert h1 == h2


class TestTaskAnomalyDetector:
    def test_detect_non_final_returns_empty(self):
        detector = TaskAnomalyDetector(root_dir=__import__("pathlib").Path("/tmp"))
        ctx = TaskContext(sample_id=0, raw_sample={})
        assert detector.detect(ctx, task_tags={"gen"}) == {}
        assert detector.has_anomalies(ctx, task_tags={"gen"}) is False

    def test_detect_returns_rule_to_indices(self, tmp_path, sample_model_meta):
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        output = ModelOutput(model=sample_model_meta, texts=[])
        ctx = _make_final_ctx(infer_result=output, postprocess_result="ok")
        result = detector.detect(ctx, task_tags={"gen", "zero_shot"})
        assert "empty_infer_gen" in result
        assert result["empty_infer_gen"] == {0}
        assert detector.has_anomalies(ctx, task_tags={"gen", "zero_shot"}) is True

    def test_detect_truncated_multi_sample(self, tmp_path, sample_model_meta):
        """detect() maps rule -> specific sample indices for n>1 outputs."""
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        output = ModelOutput(
            model=sample_model_meta,
            texts=["a", "b", "c"],
            finish_reasons=["stop", "length", "max_tokens"],
        )
        ctx = _make_final_ctx(infer_result=output, postprocess_result="ok")
        result = detector.detect(ctx, task_tags={"gen", "zero_shot"})
        assert result["truncated_output"] == {1, 2}


class TestCustomDetectionRule:
    """Custom detection rule registration via @sieval_detection_rule."""

    @staticmethod
    def _register_custom_rule() -> None:
        @sieval_detection_rule(
            description="Postprocess result contains the substring BAD",
            category="correctness",
            rationale=(
                "Answers containing BAD are considered anomalous in this test suite."
            ),
            severity="warning",
            tags=["custom", "bad_answer"],
        )
        def detect_custom_test_bad_answer(ctx: TaskContext) -> set[int]:
            if ctx.postprocess_result is None:
                return set()
            post = _unwrap_result(ctx.postprocess_result)
            return {0} if isinstance(post, str) and "BAD" in post else set()

    def test_custom_rule_lifecycle(self, tmp_path):
        applied = get_applied_rules()
        assert "custom_test_bad_answer" not in applied

        self._register_custom_rule()
        applied = get_applied_rules()
        assert "custom_test_bad_answer" in applied

        schema = get_rules_schema()
        rule_names = [r["name"] for r in schema["rules"]]
        assert "custom_test_bad_answer" in rule_names

        defn = _DETECTION_RULES["custom_test_bad_answer"]["definition"]
        assert defn["category"] == "correctness"
        assert defn["severity"] == "warning"
        assert defn["description"] == "Postprocess result contains the substring BAD"

        detector = TaskAnomalyDetector(root_dir=tmp_path)
        ctx = _make_final_ctx(postprocess_result="BAD_answer")
        result = detector.detect(ctx, task_tags={"gen"})
        assert "custom_test_bad_answer" in result
        assert result["custom_test_bad_answer"] == {0}


class TestDetectionRuleRegistration:
    """What `@sieval_detection_rule` derives, not just that it registers.

    The built-in rules are registered at import, so a test that only reads the
    resulting registry cannot see the decorator's own logic. These exercise it
    directly. The derived name is what `applies_to` and every stored report key
    on, so a change here renames rules across the fleet.
    """

    def _register(self, func_name: str, **kwargs):
        def rule(ctx: TaskContext) -> set[int]:
            # Never fires: these tests are about registration, not detection.
            # The parameter is named and typed to satisfy the DetectFunc
            # protocol, which declares it positionally *and* by keyword.
            del ctx
            return set()

        rule.__name__ = func_name
        sieval_detection_rule(
            description=kwargs.pop("description", "d"),
            category=kwargs.pop("category", "output_quality"),
            rationale=kwargs.pop("rationale", "r"),
            **kwargs,
        )(rule)
        return _DETECTION_RULES

    def test_the_detect_prefix_is_stripped_from_the_rule_name(self):
        assert "empty_thing" in self._register("detect_empty_thing")

    def test_a_private_detect_prefix_is_stripped_too(self):
        assert "empty_thing" in self._register("_detect_empty_thing")

    def test_a_name_without_the_prefix_is_kept_whole(self):
        # Stripping a suffix instead, or matching case-insensitively, would
        # silently rename rules that do not follow the convention.
        assert "custom_rule" in self._register("custom_rule")

    def test_default_tags_come_from_the_rule_name(self):
        rules = self._register("detect_empty_thing")
        assert rules["empty_thing"]["definition"]["tags"] == ["empty thing"]

    def test_explicit_tags_win_over_the_derived_default(self):
        rules = self._register("detect_empty_thing", tags=["explicit"])
        assert rules["empty_thing"]["definition"]["tags"] == ["explicit"]

    def test_an_empty_tag_list_is_respected_not_replaced(self):
        # `is not None`, not truthiness: an explicitly empty list means "no
        # tags", which is different from "derive some for me".
        rules = self._register("detect_empty_thing", tags=[])
        assert rules["empty_thing"]["definition"]["tags"] == []

    def test_severity_defaults_to_warning(self):
        rules = self._register("detect_empty_thing")
        assert rules["empty_thing"]["definition"]["severity"] == "warning"

    def test_severity_can_be_raised(self):
        rules = self._register("detect_empty_thing", severity="error")
        assert rules["empty_thing"]["definition"]["severity"] == "error"

    def test_the_definition_carries_every_declared_field(self):
        # These keys are serialized into `rules_schema` and hashed; a renamed
        # key changes the hash and breaks any consumer reading the report.
        rules = self._register(
            "detect_empty_thing",
            description="a description",
            category="correctness",
            rationale="a rationale",
            applies_to=["gen"],
            threshold=3,
        )
        definition = rules["empty_thing"]["definition"]
        assert definition["description"] == "a description"
        assert definition["category"] == "correctness"
        assert definition["rationale"] == "a rationale"
        assert definition["applies_to"] == ["gen"]
        assert definition["threshold"] == 3

    def test_the_registered_function_is_the_one_that_runs(self):
        rules = self._register("detect_empty_thing")
        assert rules["empty_thing"]["func"].__name__ == "detect_empty_thing"


class TestDetectGating:
    """`detect` runs only the rules that apply, and only on finished samples."""

    def test_a_non_final_sample_is_not_inspected(self):
        # A sample still moving through the pipeline has no settled output to
        # judge; flagging it would report anomalies that resolve themselves.
        ctx = TaskContext(sample_id=0, raw_sample={}).to_preprocessed("pre")
        assert TaskAnomalyDetector(root_dir=Path("/tmp")).detect(ctx, {"gen"}) == {}

    def test_a_failed_sample_is_not_inspected(self):
        ctx = TaskContext(sample_id=0, raw_sample={}).to_preprocessed("pre")
        ctx = ctx.to_failed(None, "error", "msg")
        assert TaskAnomalyDetector(root_dir=Path("/tmp")).detect(ctx, {"gen"}) == {}

    def test_no_tags_means_no_detection(self):
        # Rules select on tags; with none, every rule would either all-apply or
        # none-apply, and neither is a defensible default.
        ctx = _make_final_ctx(postprocess_result="answer")
        assert TaskAnomalyDetector(root_dir=Path("/tmp")).detect(ctx, set()) == {}

    def test_rules_that_do_not_apply_are_skipped(self):
        # A tag no rule declares must yield nothing rather than everything.
        ctx = _make_final_ctx(postprocess_result="answer")
        detector = TaskAnomalyDetector(root_dir=Path("/tmp"))
        assert detector.detect(ctx, {"no_rule_declares_this_tag"}) == {}

    def test_a_rule_reporting_nothing_is_absent_from_the_result(self):
        # Only tripped rules appear; an empty set per rule would make every
        # sample look inspected-and-anomalous in the report's sample map.
        ctx = _make_final_ctx(postprocess_result="a perfectly normal answer")
        result = TaskAnomalyDetector(root_dir=Path("/tmp")).detect(ctx, {"gen"})
        assert all(indices for indices in result.values())

    def test_has_anomalies_agrees_with_detect(self):
        ctx = _make_final_ctx(postprocess_result="answer")
        detector = TaskAnomalyDetector(root_dir=Path("/tmp"))
        assert detector.has_anomalies(ctx, {"gen"}) is bool(
            detector.detect(ctx, {"gen"})
        )
        assert detector.has_anomalies(ctx, set()) is False


class TestAnomalyReport:
    """`anomalies.json` holds per-sample detail; counts are `report.json`'s."""

    RESULTS: dict = {
        # two iterations, one rule each, different numbers of rollouts
        "s1": {1: {"rule_a": [2, 0]}, 0: {"rule_a": [0]}},
        # a second sample tripping a different rule
        "s2": {0: {"rule_b": [5]}},
        # present but clean: not persisted
        "s3": {},
    }

    def test_every_iteration_is_kept_and_keys_are_strings(self, tmp_path):
        report = TaskAnomalyDetector(root_dir=tmp_path).build_report(self.RESULTS, "t")
        assert report["samples"] == {
            "s1": {"0": {"rule_a": [0]}, "1": {"rule_a": [0, 2]}},
            "s2": {"0": {"rule_b": [5]}},
        }
        assert "summary" not in report
        assert report["meta"]["rules_hash"] == get_rules_hash()

    @pytest.mark.anyio
    async def test_write_replaces_atomically_and_returns_the_hash(self, tmp_path):
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        assert await detector.write(self.RESULTS, "first") == get_rules_hash()
        await detector.write({}, "second")
        on_disk = orjson.loads((tmp_path / "anomalies.json").read_bytes())
        assert on_disk["meta"]["task_name"] == "second"
        assert on_disk["samples"] == {}
        assert list(tmp_path.glob("*.tmp")) == []
        assert list(tmp_path.glob("anomalies.*.json")) == []

    @pytest.mark.anyio
    async def test_a_failed_write_is_logged_not_raised(self, tmp_path):
        detector = TaskAnomalyDetector(root_dir=tmp_path / "missing")
        assert await detector.write(self.RESULTS, "t") is None
        assert not (tmp_path / "missing").exists()


class TestRulesHashStability:
    """`rules_hash` is what tells a reader the rule set moved under them."""

    def test_the_hash_is_stable_across_calls(self):
        assert get_rules_hash() == get_rules_hash()

    def test_the_hash_is_pinned_to_the_current_rule_set(self):
        """Changing any rule's prose changes every new `runs[].rules_hash`.

        The hash is computed over the whole rules schema — names, descriptions,
        rationales — so editing a description, not just adding a rule, is a
        visible event in every task directory written after it. If this test
        fails you changed the rule set: update the value here in the same
        commit, so the rotation is in the diff.
        """
        assert get_rules_hash() == "2c440347324b1edb"

    def test_the_rule_set_is_the_expected_size(self):
        # Guards the other direction: a rule silently dropped from the registry
        # stops being detected, and nothing else would notice.
        assert len(get_applied_rules()) == 5

    def test_the_hash_is_short_and_hex(self):
        # Persisted into every report and compared as a string; a change in
        # width or alphabet silently invalidates every stored comparison.
        h = get_rules_hash()
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestDetectionRuleDecorator:
    def test_threshold_stored_in_definition(self):
        """threshold kwarg is persisted in the rule definition."""

        @sieval_detection_rule(
            description="Test rule with threshold",
            category="output_quality",
            rationale="Testing threshold storage",
            threshold=42,
        )
        def detect_threshold_test(ctx: TaskContext) -> set[int]:  # noqa: ARG001
            return set()

        defn = _DETECTION_RULES["threshold_test"]["definition"]
        assert defn["threshold"] == 42

    def test_detect_exception_is_logged_not_raised(self, tmp_path):
        """A rule that raises should not propagate — anomaly is skipped."""

        @sieval_detection_rule(
            description="Buggy rule",
            category="correctness",
            rationale="Testing error handling",
        )
        def detect_buggy_rule(ctx: TaskContext) -> set[int]:  # noqa: ARG001
            raise RuntimeError("rule exploded")

        detector = TaskAnomalyDetector(root_dir=tmp_path)
        ctx = _make_final_ctx(postprocess_result="ok")
        # Should not raise; buggy rule is silently skipped
        result = detector.detect(ctx, task_tags={"gen"})
        assert "buggy_rule" not in result


class TestDetectTruncatedOutputNoneInfer:
    def test_none_infer_result_returns_empty(self):
        """detect_truncated_output returns empty set when infer_result is None."""
        ctx = _make_final_ctx(infer_result=None, postprocess_result="ok")
        assert detect_truncated_output(ctx) == set()


class TestRuleApplies:
    """Unit tests for _rule_applies matching logic."""

    def test_all_tasks_always_matches(self):
        assert _rule_applies(["all_tasks"], set()) is True
        assert _rule_applies(["all_tasks"], {"gen"}) is True
        assert _rule_applies(["all_tasks"], {"ppl"}) is True

    def test_none_tags_never_matches(self):
        """None task_tags is no longer supported — _rule_applies requires set."""
        assert _rule_applies(["gen"], set()) is False
        assert _rule_applies(["ppl"], set()) is False

    def test_single_tag_match(self):
        assert _rule_applies(["gen"], {"gen", "zero_shot"}) is True
        assert _rule_applies(["ppl"], {"ppl", "few_shot"}) is True

    def test_single_tag_no_match(self):
        assert _rule_applies(["gen"], {"ppl", "few_shot"}) is False
        assert _rule_applies(["ppl"], {"gen", "zero_shot"}) is False

    def test_or_semantics(self):
        """applies_to is OR-list: match if ANY entry is in task_tags."""
        assert _rule_applies(["gen", "ppl"], {"gen"}) is True
        assert _rule_applies(["gen", "ppl"], {"ppl"}) is True
        assert _rule_applies(["gen", "ppl"], {"base"}) is False

    def test_empty_applies_to_never_matches(self):
        assert _rule_applies([], {"gen"}) is False
        assert _rule_applies([], set()) is False

    def test_empty_task_tags_no_match_for_specific_rules(self):
        assert _rule_applies(["gen"], set()) is False
        assert _rule_applies(["all_tasks"], set()) is True


class TestDetectWithTaskTags:
    """Verify detect() filters rules by task_tags."""

    def test_gen_rule_skipped_for_ppl_task(self, tmp_path, sample_model_meta):
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        output = ModelOutput(model=sample_model_meta, texts=[])
        ctx = _make_final_ctx(infer_result=output, postprocess_result="ok")
        result = detector.detect(ctx, task_tags={"ppl", "few_shot"})
        assert "empty_infer_gen" not in result

    def test_ppl_rule_skipped_for_gen_task(self, tmp_path, sample_model_meta):
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        output = ModelOutput(
            model=sample_model_meta, texts=["x"], logprobs=[], logprobs_tokens=["a"]
        )
        ctx = _make_final_ctx(infer_result=output, postprocess_result="ok")
        result = detector.detect(ctx, task_tags={"gen", "zero_shot"})
        assert "empty_infer_ppl" not in result

    def test_all_tasks_rule_always_runs(self, tmp_path, sample_model_meta):
        # empty_postprocess is an all_tasks rule: it runs regardless of tags.
        # (truncated_output was the old sentinel but is now scoped to gen.)
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        output = ModelOutput(model=sample_model_meta, texts=["x"])
        ctx = _make_final_ctx(infer_result=output, postprocess_result="")
        result = detector.detect(ctx, task_tags={"ppl"})
        assert "empty_postprocess" in result

    def test_truncated_output_scoped_to_gen(self, tmp_path, sample_model_meta):
        # clp/ppl infer at max_tokens=1 → always finish "length"; truncation is
        # only meaningful for generation, so the rule must not fire for them.
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        output = ModelOutput(
            model=sample_model_meta, texts=["x"], finish_reasons=["length"]
        )
        ctx = _make_final_ctx(infer_result=output, postprocess_result="ok")
        assert "truncated_output" not in detector.detect(ctx, task_tags={"clp"})
        assert "truncated_output" in detector.detect(ctx, task_tags={"gen"})

    def test_empty_tags_skips_detection_with_warning(self, tmp_path, sample_model_meta):
        """Empty tags → skip detection and warn."""
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        output = ModelOutput(model=sample_model_meta, texts=[])
        ctx = _make_final_ctx(infer_result=output, postprocess_result="ok")
        result = detector.detect(ctx, task_tags=set())
        assert result == {}

    def test_has_anomalies_forwards_tags(self, tmp_path, sample_model_meta):
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        output = ModelOutput(model=sample_model_meta, texts=[])
        ctx = _make_final_ctx(infer_result=output, postprocess_result="ok")
        assert detector.has_anomalies(ctx, task_tags={"gen"}) is True
        assert detector.has_anomalies(ctx, task_tags={"ppl"}) is False
