"""Anomaly detection framework: decorator-based rule registration and reporting."""

import contextlib
import json
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NotRequired, Protocol, TypedDict

import anyio
import orjson
import xxhash
from loguru import logger

from sieval.core.models import ModelOutput
from sieval.core.tasks.context import TaskContext, TaskStage, TaskStageOutput


# Schema
class RuleDefinition(TypedDict):
    name: str
    description: str
    category: Literal["output_quality", "performance", "correctness"]
    applies_to: list[str]
    severity: Literal["info", "warning", "error"]
    rationale: str
    tags: list[str]
    threshold: NotRequired[int | float]


class RulesSchema(TypedDict):
    version: str
    rules: list[RuleDefinition]


class AnomalyReportMeta(TypedDict):
    generated_at: str
    task_name: str
    rules_schema: RulesSchema
    rules_hash: str
    note: str


class AnomalyReportSummary(TypedDict):
    total_samples: int
    final_samples: int
    failed_samples: int
    anomaly_samples: int
    anomaly_sample_details: dict[str, int]
    anomaly_rollout_details: dict[str, int]


class AnomalyReport(TypedDict):
    meta: AnomalyReportMeta
    samples: dict[str, dict[str, dict[str, list[int]]]]
    summary: AnomalyReportSummary


class DetectFunc(Protocol):
    __name__: str

    def __call__(self, ctx: TaskContext) -> set[int]: ...


class RegisteredRule(TypedDict):
    func: DetectFunc
    definition: RuleDefinition


# Global registry for detection rules
_DETECTION_RULES: dict[str, RegisteredRule] = {}


def sieval_detection_rule(
    description: str,
    category: Literal["output_quality", "performance", "correctness"],
    rationale: str,
    applies_to: list[str] | None = None,
    severity: Literal["info", "warning", "error"] = "warning",
    tags: list[str] | None = None,
    threshold: int | float | None = None,
) -> Callable[[DetectFunc], DetectFunc]:
    """Register a function as an anomaly detection rule.

    Example::

        @sieval_detection_rule(
            description="Output is empty",
            category="output_quality",
            rationale="Empty outputs indicate failures",
        )
        def detect_empty_output(ctx: TaskContext) -> set[int]:
            return {0} if not ctx.postprocess_result else set()
    """

    def decorator(func: DetectFunc) -> DetectFunc:
        # Extract rule name from function name
        func_name = func.__name__
        rule_name = func_name.removeprefix("_detect_").removeprefix("detect_")

        # Auto-generate tags if not provided
        rule_tags = tags if tags is not None else [rule_name.replace("_", " ")]

        # Build rule definition
        definition: RuleDefinition = {
            "name": rule_name,
            "description": description,
            "category": category,
            "applies_to": applies_to if applies_to is not None else ["all_tasks"],
            "severity": severity,
            "rationale": rationale,
            "tags": rule_tags,
        }
        if threshold is not None:
            definition["threshold"] = threshold

        # Register the rule globally
        _DETECTION_RULES[rule_name] = {
            "func": func,
            "definition": definition,
        }
        return func

    return decorator


# Rule query functions
def get_rules_schema() -> RulesSchema:
    return {
        "version": "1.0",
        "rules": [info["definition"] for info in _DETECTION_RULES.values()],
    }


def get_rules_hash() -> str:
    schema = get_rules_schema()
    schema_str = json.dumps(schema, sort_keys=True)
    return xxhash.xxh3_64(schema_str.encode()).hexdigest()[:16]


def get_applied_rules() -> list[str]:
    return list(_DETECTION_RULES.keys())


def get_rules_by_category() -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for rule_name, rule_info in _DETECTION_RULES.items():
        category = rule_info["definition"].get("category", "uncategorized")
        if category not in categories:
            categories[category] = []
        categories[category].append(rule_name)
    return categories


def _rule_applies(applies_to: list[str], task_tags: AbstractSet[str]) -> bool:
    """Check if a rule should run for a task with the given tags.

    A rule applies when ANY of its ``applies_to`` entries is present in
    *task_tags*.  The sentinel ``"all_tasks"`` makes the rule match
    unconditionally.
    """
    if "all_tasks" in applies_to:
        return True
    return bool(set(applies_to) & task_tags)


class TaskAnomalyDetector:
    """Run detection rules against completed samples and persist reports."""

    def __init__(self, root_dir: Path):
        self._root_dir = root_dir
        self._report_path = root_dir / "anomalies.json"
        self._current_report: AnomalyReport | None = None

    def detect(
        self, ctx: TaskContext, task_tags: AbstractSet[str]
    ) -> dict[str, set[int]]:
        if ctx.stage != TaskStage.FINAL:
            return {}
        if not task_tags:
            logger.warning(
                "Task has no tags; skipping anomaly detection. "
                "Define `tags` on the task class."
            )
            return {}

        anomalies: dict[str, set[int]] = {}
        for rule_name, rule_info in _DETECTION_RULES.items():
            if not _rule_applies(rule_info["definition"]["applies_to"], task_tags):
                continue
            try:
                indices = rule_info["func"](ctx)
                if indices:
                    anomalies[rule_name] = indices
            except Exception as e:
                logger.warning("Rule {} failed: {}", rule_name, e)

        return anomalies

    def has_anomalies(self, ctx: TaskContext, task_tags: AbstractSet[str]) -> bool:
        return bool(self.detect(ctx, task_tags))

    def generate_report(
        self,
        contexts: dict[str | int, TaskContext],
        task_name: str,
        task_tags: AbstractSet[str],
    ) -> AnomalyReport:
        samples: dict[str, dict[str, dict[str, list[int]]]] = {}
        final_count = 0
        failed_count = 0
        anomaly_sample_count = 0
        anomaly_sample_details: dict[str, int] = defaultdict(int)
        anomaly_rollout_details: dict[str, int] = defaultdict(int)

        for sid, ctx in contexts.items():
            if ctx.stage == TaskStage.FINAL:
                final_count += 1
                anomalies = self.detect(ctx, task_tags)
                if anomalies:
                    samples[str(sid)] = {
                        str(ctx.iteration): {
                            rule: sorted(indices)
                            for rule, indices in sorted(anomalies.items())
                        }
                    }
                    anomaly_sample_count += 1
                    for rule, indices in anomalies.items():
                        anomaly_sample_details[rule] += 1
                        anomaly_rollout_details[rule] += len(indices)
            elif ctx.stage == TaskStage.FAILED:
                failed_count += 1

        return AnomalyReport(
            meta=AnomalyReportMeta(
                generated_at=datetime.now(UTC).isoformat(),
                task_name=task_name,
                rules_schema=get_rules_schema(),
                rules_hash=get_rules_hash(),
                note=(
                    "This is a reference anomaly detection result "
                    "generated by sieval. "
                    "You can use it directly or implement your own "
                    "detection logic. "
                    "See 'rules_schema' for details about each rule."
                ),
            ),
            samples=samples,
            summary=AnomalyReportSummary(
                total_samples=len(contexts),
                final_samples=final_count,
                failed_samples=failed_count,
                anomaly_samples=anomaly_sample_count,
                anomaly_sample_details=dict(anomaly_sample_details),
                anomaly_rollout_details=dict(anomaly_rollout_details),
            ),
        )

    async def load(self) -> AnomalyReport | None:
        if not await anyio.Path(self._report_path).exists():
            return None

        try:
            async with await anyio.open_file(self._report_path, "rb") as f:
                self._current_report = orjson.loads(await f.read())
            return self._current_report
        except Exception as e:
            logger.warning("Failed to load anomaly report: {}", e)
            return None

    async def save(self, report: AnomalyReport, backup_if_changed: bool = True) -> None:
        # Write to temp file then replace
        tmp_path = self._report_path.with_suffix(".tmp")
        try:
            # Backup old report if rules changed
            if backup_if_changed and await anyio.Path(self._report_path).exists():
                await self._backup_if_rules_changed(report)

            async with await anyio.open_file(tmp_path, "wb") as f:
                await f.write(orjson.dumps(report, option=orjson.OPT_NON_STR_KEYS))
            await anyio.Path(tmp_path).replace(self._report_path)

            self._current_report = report
            logger.info("Saved anomaly report to: {}", self._report_path)
        except Exception as e:
            with contextlib.suppress(OSError):
                await anyio.Path(tmp_path).unlink(missing_ok=True)
            logger.error("Failed to save anomaly report: {}", e)

    async def generate_and_save(
        self,
        contexts: dict[str | int, TaskContext],
        task_name: str,
        task_tags: AbstractSet[str],
        backup_if_changed: bool = True,
    ) -> AnomalyReport:
        report = self.generate_report(contexts, task_name, task_tags)
        await self.save(report, backup_if_changed=backup_if_changed)
        return report

    async def generate_and_save_from_results(
        self,
        anomaly_results: dict[str | int, dict[int, dict[str, list[int]]]],
        task_name: str,
        total_samples: int,
        final_count: int,
        failed_count: int,
        backup_if_changed: bool = True,
    ) -> AnomalyReport:
        """Generate and save a report from pre-computed anomaly results.

        Used by the runner to avoid retaining full context objects in memory
        across iterations. anomaly_results has the shape:
            {sample_id: {iteration: {rule: [indices]}}}
        """
        samples: dict[str, dict[str, dict[str, list[int]]]] = {}
        anomaly_sample_count = 0
        anomaly_sample_details: dict[str, int] = defaultdict(int)
        anomaly_rollout_details: dict[str, int] = defaultdict(int)

        for sid, iter_map in anomaly_results.items():
            if iter_map:
                samples[str(sid)] = {str(it): rules for it, rules in iter_map.items()}
                anomaly_sample_count += 1
                for rules in iter_map.values():
                    for rule, indices in rules.items():
                        anomaly_sample_details[rule] += 1
                        anomaly_rollout_details[rule] += len(indices)

        report = AnomalyReport(
            meta=AnomalyReportMeta(
                generated_at=datetime.now(UTC).isoformat(),
                task_name=task_name,
                rules_schema=get_rules_schema(),
                rules_hash=get_rules_hash(),
                note=(
                    "This is a reference anomaly detection result "
                    "generated by sieval. "
                    "You can use it directly or implement your own "
                    "detection logic. "
                    "See 'rules_schema' for details about each rule."
                ),
            ),
            samples=samples,
            summary=AnomalyReportSummary(
                total_samples=total_samples,
                final_samples=final_count,
                failed_samples=failed_count,
                anomaly_samples=anomaly_sample_count,
                anomaly_sample_details=dict(anomaly_sample_details),
                anomaly_rollout_details=dict(anomaly_rollout_details),
            ),
        )
        await self.save(report, backup_if_changed=backup_if_changed)
        return report

    def needs_regeneration(self) -> bool:
        if not self._current_report:
            return True

        old_hash = self._current_report["meta"]["rules_hash"]
        return old_hash != get_rules_hash()

    async def _backup_if_rules_changed(self, new_report: AnomalyReport) -> None:
        try:
            # Load old report to check hash
            async with await anyio.open_file(self._report_path, "rb") as f:
                old_report = orjson.loads(await f.read())

            old_hash = old_report.get("meta", {}).get("rules_hash")
            new_hash = new_report["meta"]["rules_hash"]

            # Only backup if hashes differ
            if old_hash and old_hash != new_hash:
                # Generate timestamp from old report's generated_at
                generated_at = old_report.get("meta", {}).get("generated_at", "")
                if generated_at:
                    dt = datetime.fromisoformat(generated_at)
                    timestamp = dt.strftime("%Y%m%d%H%M%S")
                    backup_path = self._root_dir / f"anomalies.{timestamp}.json"
                    await anyio.Path(self._report_path).replace(backup_path)
                    logger.info("Backed up old report to: {}", backup_path)
        except Exception as e:
            logger.warning("Failed to backup old anomaly report: {}", e)


# Built-in Detection Rules
def _unwrap_result(result: Any) -> Any:
    if isinstance(result, TaskStageOutput):
        return result.value
    return result


@sieval_detection_rule(
    description="Inference result is empty for generation tasks (texts=[])",
    category="output_quality",
    rationale=(
        "Empty inference results usually indicate API failures, "
        "rate limiting, or configuration issues."
    ),
    applies_to=["gen"],
    tags=["generation", "api_failure", "empty_output"],
)
def detect_empty_infer_gen(ctx: TaskContext) -> set[int]:
    if ctx.infer_result is None:
        return set()
    result = _unwrap_result(ctx.infer_result)
    if isinstance(result, ModelOutput):
        # texts=[] means all samples are missing — report index 0 as sentinel
        return {0} if not result.texts else set()
    return set()


@sieval_detection_rule(
    description="Inference result is empty for perplexity tasks (logprobs=[])",
    category="output_quality",
    rationale=("Empty logprobs indicate API failures or unsupported model features."),
    applies_to=["ppl"],
    tags=["perplexity", "api_failure", "empty_output"],
)
def detect_empty_infer_ppl(ctx: TaskContext) -> set[int]:
    if ctx.infer_result is None:
        return set()
    result = _unwrap_result(ctx.infer_result)
    if not isinstance(result, ModelOutput):
        return set()
    # For PPL tasks, we need both logprobs and logprobs_tokens
    has_logprobs = result.logprobs is not None
    has_logprobs_tokens = result.logprobs_tokens is not None
    if has_logprobs or has_logprobs_tokens:
        logprobs_empty = has_logprobs and not result.logprobs
        logprobs_tokens_empty = has_logprobs_tokens and not result.logprobs_tokens
        # Report index 0 as sentinel if any field is empty
        return {0} if (logprobs_empty or logprobs_tokens_empty) else set()
    return set()


@sieval_detection_rule(
    description="Output was truncated due to length limit",
    category="output_quality",
    rationale=(
        "Truncated outputs may indicate insufficient max_tokens setting "
        "or unexpectedly long responses. Scoped to gen: single-token logprob "
        "tasks (ppl/clp) infer with max_tokens=1 and always finish with "
        "'length', so the signal is only meaningful for generation."
    ),
    severity="info",
    applies_to=["gen"],
    tags=["truncation", "length_limit", "incomplete_output"],
)
def detect_truncated_output(ctx: TaskContext) -> set[int]:
    if ctx.infer_result is None:
        return set()
    result = _unwrap_result(ctx.infer_result)
    if not isinstance(result, ModelOutput) or not result.finish_reasons:
        return set()
    return {
        i
        for i, reason in enumerate(result.finish_reasons)
        if reason in ("length", "max_tokens", "content_filter")
    }


@sieval_detection_rule(
    description="Postprocess result is empty or None",
    category="correctness",
    rationale=(
        "Empty postprocess results indicate parsing failures or "
        "inability to extract answers from model output."
    ),
    tags=["parsing", "extraction", "correctness"],
)
def detect_empty_postprocess(ctx: TaskContext) -> set[int]:
    if ctx.postprocess_result is None:
        return {0}
    post_result = _unwrap_result(ctx.postprocess_result)
    is_empty_collection = (
        isinstance(post_result, (list, dict, set, tuple)) and len(post_result) == 0
    )
    is_empty_string = isinstance(post_result, str) and not post_result.strip()
    if post_result is None or is_empty_collection or is_empty_string:
        return {0}
    return set()
