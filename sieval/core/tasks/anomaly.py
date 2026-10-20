"""Anomaly detection framework: decorator-based rule registration and reporting."""

import json
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NotRequired, Protocol, TypedDict

import orjson
import xxhash
from loguru import logger

from sieval.core.models import ModelOutput
from sieval.core.tasks.context import TaskContext, TaskStage, TaskStageOutput
from sieval.core.tasks.records import is_prediction_record
from sieval.core.utils.atomic import aatomic_write
from sieval.core.utils.meta import is_truncated_finish_reason


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


class AnomalyReport(TypedDict):
    meta: AnomalyReportMeta
    samples: dict[str, dict[str, dict[str, list[int]]]]


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

    def build_report(
        self,
        results: Mapping[str | int, Mapping[int, Mapping[str, list[int]]]],
        task_name: str,
    ) -> AnomalyReport:
        """``anomalies.json`` from ``{sid: {iteration: {rule: indices}}}``.

        Counts live in ``report.json``'s ``sieval`` block, not here.
        """
        samples = {
            str(sid): {
                str(it): {rule: sorted(ix) for rule, ix in sorted(rules.items())}
                for it, rules in sorted(iterations.items())
            }
            for sid, iterations in results.items()
            if iterations
        }
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
        )

    async def write(
        self,
        results: Mapping[str | int, Mapping[int, Mapping[str, list[int]]]],
        task_name: str,
    ) -> str | None:
        """Write ``anomalies.json``; return its rules hash, ``None`` if it failed.

        No decision reads it, so a failed write is only logged.
        """
        report = self.build_report(results, task_name)
        try:
            await aatomic_write(
                self._report_path,
                orjson.dumps(report, option=orjson.OPT_NON_STR_KEYS),
            )
        except OSError as e:
            logger.error("Failed to save anomaly report: {}", e)
            return None
        logger.info("Saved anomaly report to: {}", self._report_path)
        return report["meta"]["rules_hash"]


# Built-in Detection Rules
def _unwrap_result(result: Any) -> Any:
    if isinstance(result, TaskStageOutput):
        return result.value
    return result


def _model_outputs(result: Any) -> tuple[ModelOutput, ...]:
    """The ``ModelOutput``s a stage produced, whether it returned one or a list.

    ``infer`` may return one ``ModelOutput`` or a list of them -- one call per turn
    for a multi-turn task, one per candidate for ``ppl``. The runner already reads
    that list as *one sample, several calls* when it sums token usage
    (``_build_auto_meta``), and rules read it the same way: the **rollout** index is
    the position inside each output's ``texts``, the list dimension is extra calls.
    So ``len(indices)`` keeps meaning "how many rollouts", which is what
    ``anomaly_rollout_details`` counts.

    Without this, a rule guarding on ``isinstance(result, ModelOutput)`` returns an
    empty set for every multi-call sample -- indistinguishable from "nothing wrong".
    """
    if isinstance(result, ModelOutput):
        return (result,)
    # An empty list means no calls to judge, not a call that came back empty --
    # `detect_empty_postprocess` reports that, from the record.
    if isinstance(result, list) and all(
        isinstance(item, ModelOutput) for item in result
    ):
        return tuple(result)
    return ()


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
    outputs = _model_outputs(_unwrap_result(ctx.infer_result))
    # texts=[] means all samples are missing — report index 0 as sentinel.
    # For a multi-call stage (one call per conversation turn) a single call
    # coming back empty is the same signal, so any of them is enough.
    return {0} if any(not output.texts for output in outputs) else set()


@sieval_detection_rule(
    description=(
        "Inference result is empty for perplexity tasks (any candidate's logprobs=[])"
    ),
    category="output_quality",
    rationale=("Empty logprobs indicate API failures or unsupported model features."),
    applies_to=["ppl"],
    tags=["perplexity", "api_failure", "empty_output"],
)
def detect_empty_infer_ppl(ctx: TaskContext) -> set[int]:
    if ctx.infer_result is None:
        return set()
    # One call per candidate is what `ppl` *is*, so every ppl task returns a list;
    # reading only a lone `ModelOutput` left this rule inert for all of them. Any
    # candidate is enough: `postprocess` takes an argmax across them, and an empty
    # one sums to 0.0 where a real one is negative, so it wins.
    for output in _model_outputs(_unwrap_result(ctx.infer_result)):
        # For PPL tasks, we need both logprobs and logprobs_tokens
        has_logprobs = output.logprobs is not None
        has_logprobs_tokens = output.logprobs_tokens is not None
        if not (has_logprobs or has_logprobs_tokens):
            continue
        logprobs_empty = has_logprobs and not output.logprobs
        logprobs_tokens_empty = has_logprobs_tokens and not output.logprobs_tokens
        # Report index 0 as sentinel if any field is empty
        if logprobs_empty or logprobs_tokens_empty:
            return {0}
    return set()


@sieval_detection_rule(
    description="Output was interrupted before completion",
    category="output_quality",
    rationale=(
        "A token limit, output filter, or refusal cut the output short; only "
        "a token limit is fixed by raising max_tokens. Scoped to gen: "
        "single-token logprob "
        "tasks (ppl/clp) infer with max_tokens=1 and always finish with "
        "'length', so the signal is only meaningful for generation."
    ),
    severity="info",
    applies_to=["gen"],
    tags=["truncation", "length_limit", "output_filter", "incomplete_output"],
)
def detect_truncated_output(ctx: TaskContext) -> set[int]:
    if ctx.infer_result is None:
        return set()
    # Unioned across calls: whichever turn was cut short, the rollout was. The
    # predicate is shared with the `n_truncated` report key so the two cannot drift.
    truncated: set[int] = set()
    for output in _model_outputs(_unwrap_result(ctx.infer_result)):
        sources = output.finish_reason_sources or ()
        for index, reason in enumerate(output.finish_reasons or ()):
            source = sources[index] if index < len(sources) else None
            if is_truncated_finish_reason(reason, source):
                truncated.add(index)
    return truncated


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
    if is_prediction_record(post_result):
        # A PredictionRecord is a non-empty dict whatever its contents, so the
        # emptiness heuristics below cannot see into it. detect_extraction_failure
        # reads it properly; staying silent here keeps the two from double-reporting.
        return set()
    is_empty_collection = (
        isinstance(post_result, (list, dict, set, tuple)) and len(post_result) == 0
    )
    is_empty_string = isinstance(post_result, str) and not post_result.strip()
    if post_result is None or is_empty_collection or is_empty_string:
        return {0}
    return set()


@sieval_detection_rule(
    description=(
        "A rollout yielded no usable answer -- extraction found none, or, for an "
        "open-ended task, the generation itself was blank"
    ),
    category="correctness",
    rationale=(
        "Either way the rollout scores as wrong without the model necessarily "
        "being wrong: for a task that extracts, this points at the prompt or the "
        "extraction rule rather than at capability; for an open-ended task, at an "
        "empty generation. Reported per rollout, so an occasional miss under n>1 "
        "is distinguishable from a total failure."
    ),
    tags=["parsing", "extraction", "empty_output", "correctness"],
)
def detect_extraction_failure(ctx: TaskContext) -> set[int]:
    if ctx.postprocess_result is None:
        return set()  # detect_empty_postprocess owns the missing-result case
    post_result = _unwrap_result(ctx.postprocess_result)
    if not is_prediction_record(post_result):
        return set()
    rollouts = post_result.get("rollouts") or []
    if not rollouts:
        # A prediction record covering no rollouts is itself the anomaly; index 0
        # is a sentinel here, since there is no rollout to point at.
        return {0}
    return {
        rollout.get("index", position)
        for position, rollout in enumerate(rollouts)
        if not rollout.get("extracted")
    }
