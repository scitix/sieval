"""
Tests for sieval.core.tasks.anomaly — detection rules and TaskAnomalyDetector.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from sieval.core.models.model import ModelOutput
from sieval.core.tasks.anomaly import (
    _DETECTION_RULES,
    TaskAnomalyDetector,
    _rule_applies,
    _unwrap_result,
    detect_empty_infer_gen,
    detect_empty_infer_ppl,
    detect_empty_postprocess,
    detect_truncated_output,
    get_applied_rules,
    get_rules_by_category,
    get_rules_hash,
    get_rules_schema,
    sieval_detection_rule,
)
from sieval.core.tasks.context import TaskContext, TaskStageOutput


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


class TestDetectTruncatedOutput:
    def test_single_sample(self, sample_model_meta):
        """Single-sample (n=1) finish reason variants."""
        cases = [
            (["length"], {0}, "length"),
            (["max_tokens"], {0}, "max_tokens"),
            (["content_filter"], {0}, "content_filter"),
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


class TestDetectionRuleRegistry:
    def test_builtin_registry_and_schema_contents(self):
        rules = get_applied_rules()
        assert "empty_infer_gen" in rules
        assert "empty_infer_ppl" in rules
        assert "truncated_output" in rules
        assert "empty_postprocess" in rules

        schema = get_rules_schema()
        assert schema["version"] == "1.0"
        rule_names = {r["name"] for r in schema["rules"]}
        assert "empty_infer_gen" in rule_names
        assert "empty_infer_ppl" in rule_names
        assert "truncated_output" in rule_names
        assert "empty_postprocess" in rule_names

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

    def test_generate_report_single_iteration(self, tmp_path, sample_model_meta):
        """Flat {sid: ctx} input — report uses iteration=0 key."""
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        output = ModelOutput(model=sample_model_meta, texts=[])
        ctx_anomaly = _make_final_ctx(infer_result=output, postprocess_result="ok")
        ctx_clean = _make_final_ctx(sample_id=1, postprocess_result="answer")
        report = detector.generate_report(
            {0: ctx_anomaly, 1: ctx_clean}, "test_task", task_tags={"gen", "zero_shot"}
        )

        assert report["summary"]["total_samples"] == 2
        assert report["summary"]["final_samples"] == 2
        assert report["summary"]["anomaly_samples"] == 1
        # anomaly_sample_details counts affected samples per rule
        assert report["summary"]["anomaly_sample_details"]["empty_infer_gen"] == 1
        # empty_infer_gen fires once (sentinel index 0), so rollout count is 1
        assert report["summary"]["anomaly_rollout_details"]["empty_infer_gen"] == 1
        # Structure: {sid: {iter: {rule: [indices]}}}
        assert report["samples"] == {"0": {"0": {"empty_infer_gen": [0]}}}

    @pytest.mark.anyio
    async def test_generate_report_multi_iteration(self, tmp_path, sample_model_meta):
        """Multi-iteration anomalies are keyed by iteration in the report."""
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        output_trunc = ModelOutput(
            model=sample_model_meta,
            texts=["a", "b"],
            finish_reasons=["stop", "length"],
        )
        ctx_iter0 = _make_final_ctx(iteration=0, postprocess_result="answer")
        ctx_iter1 = _make_final_ctx(
            iteration=1, infer_result=output_trunc, postprocess_result="ok"
        )

        # Simulate what the runner collects: detect per-iteration, store results
        tags = {"gen", "zero_shot"}
        results_iter0 = detector.detect(ctx_iter0, task_tags=tags)
        results_iter1 = detector.detect(ctx_iter1, task_tags=tags)
        anomaly_results: dict[str | int, dict[int, dict[str, list[int]]]] = {}
        if results_iter0:
            anomaly_results.setdefault(0, {})[0] = {
                r: sorted(i) for r, i in results_iter0.items()
            }
        if results_iter1:
            anomaly_results.setdefault(0, {})[1] = {
                r: sorted(i) for r, i in results_iter1.items()
            }

        report = await detector.generate_and_save_from_results(
            anomaly_results,
            task_name="test_task",
            total_samples=2,
            final_count=2,
            failed_count=0,
            backup_if_changed=False,
        )

        assert report["summary"]["total_samples"] == 2
        assert report["summary"]["anomaly_samples"] == 1
        # Only iteration 1 of sample 0 has an anomaly, at index 1
        assert report["samples"] == {"0": {"1": {"truncated_output": [1]}}}


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


# ===================================================================
# Async save / load / generate_and_save / needs_regeneration
# ===================================================================
class TestTaskAnomalyDetectorAsync:
    @pytest.mark.anyio
    async def test_load_and_save_load_flow(self, tmp_path, sample_model_meta):
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        assert await detector.load() is None

        output = ModelOutput(model=sample_model_meta, texts=[])
        ctx = _make_final_ctx(infer_result=output, postprocess_result="ok")
        report = detector.generate_report({0: ctx}, "test_task", task_tags={"gen"})

        await detector.save(report, backup_if_changed=False)
        loaded = await detector.load()
        assert loaded is not None
        assert loaded["summary"]["total_samples"] == 1

    @pytest.mark.anyio
    async def test_generate_and_save(self, tmp_path, sample_model_meta):
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        output = ModelOutput(model=sample_model_meta, texts=[])
        ctx = _make_final_ctx(infer_result=output, postprocess_result="ok")
        report = await detector.generate_and_save(
            {0: ctx}, "my_task", task_tags={"gen"}
        )
        assert "meta" in report
        assert "summary" in report
        assert (tmp_path / "anomalies.json").exists()

    @pytest.mark.anyio
    async def test_needs_regeneration_transitions(self, tmp_path):
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        assert detector.needs_regeneration() is True

        ctx = _make_final_ctx(postprocess_result="answer")
        report = detector.generate_report({0: ctx}, "t", task_tags={"gen"})
        await detector.save(report, backup_if_changed=False)
        await detector.load()
        assert detector.needs_regeneration() is False

    @pytest.mark.anyio
    async def test_save_backup_when_rules_changed(self, tmp_path, sample_model_meta):
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        ctx = _make_final_ctx(postprocess_result="answer")

        report_v1 = detector.generate_report({0: ctx}, "t", task_tags={"gen"})
        report_v1["meta"]["rules_hash"] = "oldhash"
        await detector.save(report_v1, backup_if_changed=False)

        report_v2 = detector.generate_report({0: ctx}, "t", task_tags={"gen"})
        report_v2["meta"]["rules_hash"] = "newhash"
        await detector.save(report_v2, backup_if_changed=True)

        backups = list(tmp_path.glob("anomalies.*.json"))
        assert len(backups) == 1

    @pytest.mark.anyio
    async def test_generate_report_includes_failed(self, tmp_path):
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        ctx_final = _make_final_ctx(postprocess_result="good")
        ctx_fail = TaskContext(sample_id=1, raw_sample={})
        ctx_fail = ctx_fail.to_preprocessed("pre")
        ctx_fail = ctx_fail.to_failed(None, "error_reason", "error message")
        report = detector.generate_report(
            {0: ctx_final, 1: ctx_fail}, "t", task_tags={"gen"}
        )
        assert report["summary"]["final_samples"] == 1
        assert report["summary"]["failed_samples"] == 1


class TestTaskAnomalyDetectorIOErrors:
    @pytest.mark.anyio
    async def test_load_invalid_json_returns_none(self, tmp_path):
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        (tmp_path / "anomalies.json").write_bytes(b"not valid json at all !!!")
        assert await detector.load() is None

    @pytest.mark.anyio
    async def test_save_io_error_does_not_raise(self, tmp_path):
        from unittest.mock import patch

        detector = TaskAnomalyDetector(root_dir=tmp_path)
        ctx = _make_final_ctx(postprocess_result="ok")
        report = detector.generate_report({0: ctx}, "t", task_tags={"gen"})

        with patch(
            "sieval.core.tasks.anomaly.anyio.open_file",
            side_effect=OSError("disk full"),
        ):
            await detector.save(report, backup_if_changed=False)

    @pytest.mark.anyio
    async def test_backup_io_error_does_not_raise(self, tmp_path):
        """_backup_if_rules_changed swallows exceptions."""
        from unittest.mock import patch

        detector = TaskAnomalyDetector(root_dir=tmp_path)
        ctx = _make_final_ctx(postprocess_result="ok")
        report = detector.generate_report({0: ctx}, "t", task_tags={"gen"})
        # Write an initial report so backup logic is triggered
        await detector.save(report, backup_if_changed=False)

        report2 = detector.generate_report({0: ctx}, "t", task_tags={"gen"})
        report2["meta"]["rules_hash"] = "differenthash"
        with patch(
            "sieval.core.tasks.anomaly.anyio.open_file",
            side_effect=OSError("read error"),
        ):
            # Must not raise even when backup read fails
            await detector.save(report2, backup_if_changed=True)

    @pytest.mark.anyio
    async def test_backup_skipped_when_hashes_match(self, tmp_path):
        """No backup file created when rules hash is unchanged."""
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        ctx = _make_final_ctx(postprocess_result="ok")
        report = detector.generate_report({0: ctx}, "t", task_tags={"gen"})
        await detector.save(report, backup_if_changed=False)

        # Save again with the same hash — no backup should be created
        report2 = detector.generate_report({0: ctx}, "t", task_tags={"gen"})
        await detector.save(report2, backup_if_changed=True)

        backups = list(tmp_path.glob("anomalies.*.json"))
        assert len(backups) == 0

    @pytest.mark.anyio
    async def test_backup_skipped_when_generated_at_missing(self, tmp_path):
        """No backup file created when old report has no generated_at."""
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        ctx = _make_final_ctx(postprocess_result="ok")

        report_v1 = detector.generate_report({0: ctx}, "t", task_tags={"gen"})
        report_v1["meta"]["rules_hash"] = "oldhash"
        del report_v1["meta"]["generated_at"]  # type: ignore[invalid-argument-type]  # intentionally malformed
        await detector.save(report_v1, backup_if_changed=False)

        report_v2 = detector.generate_report({0: ctx}, "t", task_tags={"gen"})
        report_v2["meta"]["rules_hash"] = "newhash"
        await detector.save(report_v2, backup_if_changed=True)

        # No backup because generated_at was missing
        backups = list(tmp_path.glob("anomalies.*.json"))
        assert len(backups) == 0


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


class TestGenerateAndSaveFromResults:
    @pytest.mark.anyio
    async def test_basic_roundtrip(self, tmp_path):
        """generate_and_save_from_results writes a valid report."""
        from sieval.core.tasks.anomaly import TaskAnomalyDetector

        detector = TaskAnomalyDetector(root_dir=tmp_path)
        anomaly_results = {
            0: {0: {"truncated_output": [1, 2]}},
            1: {},  # no anomalies
        }
        report = await detector.generate_and_save_from_results(
            anomaly_results,
            task_name="test_task",
            total_samples=2,
            final_count=2,
            failed_count=0,
            backup_if_changed=False,
        )

        assert report["summary"]["total_samples"] == 2
        assert report["summary"]["final_samples"] == 2
        assert report["summary"]["failed_samples"] == 0
        assert report["summary"]["anomaly_samples"] == 1
        # anomaly_sample_details: 1 sample with truncated_output
        assert report["summary"]["anomaly_sample_details"]["truncated_output"] == 1
        # anomaly_rollout_details: 2 affected rollout indices
        assert report["summary"]["anomaly_rollout_details"]["truncated_output"] == 2
        assert report["samples"] == {"0": {"0": {"truncated_output": [1, 2]}}}
        assert (tmp_path / "anomalies.json").exists()

    @pytest.mark.anyio
    async def test_empty_results(self, tmp_path):
        """Empty anomaly_results produces a clean report."""
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        report = await detector.generate_and_save_from_results(
            {},
            task_name="t",
            total_samples=5,
            final_count=5,
            failed_count=0,
            backup_if_changed=False,
        )
        assert report["summary"]["anomaly_samples"] == 0
        assert report["samples"] == {}

    @pytest.mark.anyio
    async def test_counts_accumulate_and_default_backup_flag_is_forwarded(
        self, tmp_path, monkeypatch
    ):
        """Count aggregation and default save options should be preserved."""
        detector = TaskAnomalyDetector(root_dir=tmp_path)
        save_mock = AsyncMock(return_value=None)
        monkeypatch.setattr(detector, "save", save_mock)

        anomaly_results = {
            0: {0: {"truncated_output": [1], "empty_postprocess": [0]}},
            1: {2: {"truncated_output": [0, 2]}},
        }

        report = await detector.generate_and_save_from_results(
            anomaly_results,
            task_name="t",
            total_samples=2,
            final_count=2,
            failed_count=0,
        )

        assert report["summary"]["anomaly_samples"] == 2
        assert report["summary"]["anomaly_sample_details"] == {
            "truncated_output": 2,
            "empty_postprocess": 1,
        }
        assert report["summary"]["anomaly_rollout_details"] == {
            "truncated_output": 3,
            "empty_postprocess": 1,
        }
        save_mock.assert_awaited_once()
        assert save_mock.await_args is not None
        assert save_mock.await_args.args[0] == report
        assert save_mock.await_args.kwargs["backup_if_changed"] is True


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
