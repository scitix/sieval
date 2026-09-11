"""Shared stages for the BFCL v3 single-turn task family.

Four leaves come out of two independent choices, and this module holds what
every one of them shares.

**Group** (``non_live`` / ``live``) selects the dataset and the aggregation
rule. It is the generic base, because it is what binds the concrete sample type
and what picks the rollup.

**Protocol** (``Prompt`` / ``FC``) is upstream's own pair of published columns,
and it is the axis the two mixins here carry. They differ in exactly two
places: how the function schemas reach the model -- rendered into a system
turn, or sent as native tools -- and how calls are read back -- parsed out of
the reply text, or read off structured tool calls. Nothing else about the two
runs differs, which is why the protocol is a stateless mixin rather than a
second class hierarchy.

**One divergence follows from the protocol, and it is upstream's.** A native
tool name cannot carry a dot in the OpenAI dialects, so upstream rewrites
``geometry.triangle_area`` to ``geometry_triangle_area`` on the way out and
reverses the rewrite before comparing against gold. That reversal is the
``underscore_to_dot`` flag: true for FC, false for Prompt.

**The flag travels as a call argument, never as a module global.** Grading runs
in a worker process, so a global set in the parent never arrives: an FC run
would then be graded under the Prompt convention, and the symptom is a
plausible-looking score rather than an error. That is the whole reason grading
goes through one module-level function instead of a method.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import ClassVar

from sieval.community.bfcl_v3 import (
    CALL_EXPECTED,
    DEFAULT_SYSTEM_PROMPT,
    GOLDLESS_CATEGORIES,
    LIVE_COUNTS,
    NON_LIVE_COUNTS,
    ast_checker,
    calculate_unweighted_accuracy,
    calculate_weighted_accuracy,
    convert_to_tool,
    is_empty_output,
    is_function_calling_format_output,
    set_underscore_to_dot,
)
from sieval.community.bfcl_v3.tool_convert import ModelStyle
from sieval.community.bfcl_v3.type_mappings import GORILLA_TO_OPENAPI
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    InputKind,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    Task,
    TaskRequirements,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    interval_metrics,
    merge_metrics,
    metric_interval,
)
from sieval.core.utils.offload import GRADE_TIMEOUT, run_cpu_bound

# `CALL_EXPECTED` is upstream's opposite polarity: `live_relevance` is correct
# when a call IS produced, the two irrelevance sets when one is NOT.


def grade_single_turn(
    functions_json: str,
    decoded: list | None,
    gold_json: str | None,
    language: str,
    category: str,
    underscore_to_dot: bool,
) -> bool:
    """Grade one decoded reply. Module-level so `run_cpu_bound` can pickle it.

    `underscore_to_dot` is an ARGUMENT, not a module global set by the caller:
    this runs in a worker process, so a global set in the parent never arrives.
    Getting that wrong grades an FC run under the Prompt convention and produces
    a plausible score rather than an error.

    `decoded` is None when the reply could not be decoded into calls at all.
    """
    set_underscore_to_dot(underscore_to_dot)

    produced_call = (
        decoded is not None
        and is_function_calling_format_output(decoded)
        and not is_empty_output(decoded)
    )

    if category in GOLDLESS_CATEGORIES:
        return produced_call if category in CALL_EXPECTED else not produced_call

    if not produced_call:
        # `ast_decoder:decoder_failed` / `:decoder_wrong_output_format` -- a
        # scored outcome, not an error.
        return False

    if gold_json is None:
        raise ValueError(
            f"BFCL v3 category {category!r} is not goldless but row gold is None; "
            "a value-reference task must not grade without its reference."
        )

    result = ast_checker(
        json.loads(functions_json),
        decoded,
        json.loads(gold_json),
        language,
        category,
        "sieval",
    )
    return bool(result["valid"])


#: Upstream's harness at the pinned revision, cited by every leaf's
#: `reference_impl`.
BFCL_V3_HARNESS_URL = (
    "https://github.com/ShishirPatil/gorilla/tree/"
    "ea13468e4423454d0c213704fb87cf7cb3990433/berkeley-function-call-leaderboard"
)

#: What every `reference_impl.notes` for this benchmark has to say, whichever
#: group and protocol it is. The specific half is prepended by each leaf.
BFCL_V3_SHARED_NOTES = (
    "AST checker, source parsers, type converters, prompt and aggregation "
    "vendored verbatim from gorilla v1.3 (Apache-2.0); only `bfcl_eval.*` "
    "imports were rewritten. "
    "exec_*/rest/sql/chatable are excluded -- upstream has them commented out of "
    "its own TEST_FILE_MAPPING and they need live API keys."
)


class BfclV3Task[TSample](
    Task[
        TSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `str`: the report carries `score_key` and `denominator_policy`, which
        # name things rather than measure them. `list[float]` is an interval and
        # `dict[str, str]` the map saying which population each is clustered on.
        dict[str, float | int | str | list[float] | dict[str, str]],
    ]
):
    """Every stage but the protocol hooks and the group-level rollup."""

    #: Set by the protocol mixin.
    UNDERSCORE_TO_DOT: ClassVar[bool]
    #: Set by the group base. Maps every category of this group to its declared
    #: row count, which is also its scoring DENOMINATOR.
    CATEGORY_COUNTS: ClassVar[dict[str, int]]
    #: Set by the group base. `"non_live"` / `"live"` -- names the headline key
    #: (`f"{GROUP_KEY}_overall_acc"`) and so also `score_key`.
    GROUP_KEY: ClassVar[str]

    def _build_messages(self, raw) -> tuple[list, list[dict] | None]:
        """Return (messages, tools). Implemented by the protocol mixin.

        The messages list is bare: a record's `prompt` is `JSONValue`, and a
        `list[dict]` is not assignable to it.
        """
        raise NotImplementedError

    def _decode(self, output: ModelOutput, language: str) -> list | None:
        """Return decoded calls, or None when the reply yields none.

        `language` is the row's language -- "Python", "Java" or "JavaScript".
        The Prompt protocol passes it straight to `ast_parse`, which dispatches
        on it. The FC protocol ignores it: it reads structured tool calls and
        never parses source text.
        """
        raise NotImplementedError

    def _aggregate(self, cell: Callable[[str], dict]) -> dict:
        """Group-level rollup. Implemented by the group base."""
        raise NotImplementedError

    async def preprocess(self, raw, ctx) -> PromptRecord:
        messages, tools = self._build_messages(raw)
        return build_prompt_record(
            messages,
            reference=raw["ground_truth"],
            extra={
                "id": raw["id"],
                "category": raw["category"],
                "language": raw["language"],
                "function": raw["function"],
                "tools": tools,
            },
        )

    async def infer(self, pre, ctx) -> ModelOutput:
        tools = pre["extra"]["tools"]
        kwargs = {"tools": tools} if tools else {}
        return await self.model.agenerate(pre["prompt"], **kwargs)

    async def postprocess(self, inf, ctx) -> PredictionRecord:
        # The row's language selects the PARSER, so it has to reach `_decode`.
        # It is carried on the preprocess record rather than re-derived here.
        language = ctx.preprocess_result["extra"]["language"]
        try:
            decoded = self._decode(inf, language)
        except Exception as exc:  # noqa: BLE001 -- a decode failure is a SCORE
            # Upstream's `ast_decoder:decoder_failed`: the model produced
            # something undecodable, which is the model's outcome, not a fault.
            return build_prediction_record([None], extras=[{"decode_error": str(exc)}])
        return build_prediction_record([decoded])

    async def feedback(self, post, ctx) -> tuple[bool, JudgementRecord]:
        pre_extra = ctx.preprocess_result["extra"]
        category = pre_extra["category"]
        # `.get()`, not `[]`: a None prediction is dropped by serialization, so
        # on a resume the key is ABSENT for exactly the rows that failed to
        # decode -- the ones this task most needs to score.
        decoded = post["rollouts"][0].get("prediction")
        gold = ctx.preprocess_result.get("reference")

        try:
            correct = await run_cpu_bound(
                grade_single_turn,
                pre_extra["function"],
                decoded,
                gold,
                pre_extra["language"],
                category,
                self.UNDERSCORE_TO_DOT,
                timeout=GRADE_TIMEOUT,
            )
        except TimeoutError:
            # A grade that could not be computed in time is a wrong answer --
            # the prediction is a shape the grader cannot bound. Every OTHER
            # exception propagates: a broken grader must not read as a wrong
            # model.
            correct = False

        # A goldless category's reference is the expected OUTCOME, not None:
        # `reference_kind="value"` is inferred from whether the call sites pass
        # a literal None, and these rows do have something to compare against.
        reference = (
            gold
            if gold is not None
            else ("call" if category in CALL_EXPECTED else "no_call")
        )
        return True, build_judgement_record(
            reference,
            [build_rollout_judgement(0, correct)],
            extra={"category": category, "id": pre_extra["id"]},
        )

    async def report(self, finals: list, fails: list) -> dict:
        # One 0/1 per sample that came back, NOT one percentage point: this list
        # is both what the cells average and what the interval estimators are
        # handed, and those read `sum(values) / denominator` as a probability.
        # Hand them percent and any rate above a single point reads as a
        # saturated set, which still publishes a whole triple -- an interval
        # bracketing 100 beside a rate that is nothing of the kind. The rate is
        # scaled to percentage points once, in `cell`, so the two agree.
        correct_by_category: dict[str, list[float]] = defaultdict(list)
        for ctx in finals:
            fb = ctx.feedback_result
            correct_by_category[fb["extra"]["category"]].append(
                1.0 if fb["rollouts"][0]["correct"] else 0.0
            )

        def cell(category: str) -> dict:
            """Upstream's accuracy-dict shape, over the DECLARED denominator.

            `total_count` is the category's row count, not the number that came
            back: `DENOMINATOR_REQUESTED` charges a missing sample as wrong, and
            a sample that failed before `preprocess` has no category to be
            attributed to anyway.

            All three of upstream's keys, because both aggregation helpers read
            `display_accuracy` unconditionally and a two-key cell raises
            `KeyError`. It is never "N/A" here: that marks a category upstream
            did not evaluate, and this task always scores the full declared set.
            Their return value carries the same three keys, which is what lets
            `simple_ast` nest straight back in as a cell.
            """
            values = correct_by_category.get(category, [])
            denominator = self.CATEGORY_COUNTS[category]
            # Percentage points -- the units every rate here is published in,
            # and the units the estimators return their bounds in.
            accuracy = 100.0 * sum(values) / denominator
            return {
                "accuracy": accuracy,
                "total_count": denominator,
                "display_accuracy": accuracy,
            }

        rollup = self._aggregate(cell)

        result: dict = {
            "score": rollup[f"{self.GROUP_KEY}_overall_acc"],
            "fails": len(fails),
            SCORE_KEY_FIELD: f"{self.GROUP_KEY}_overall_acc",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        result |= rollup
        for category in self.CATEGORY_COUNTS:
            result[f"n_{category}"] = self.CATEGORY_COUNTS[category]

        # Per-category intervals: each IS `sum(values) / denominator`, so each
        # is a candidate. No `group_keys`: one row is one problem is one
        # rollout, so there are no repeated copies to collapse.
        fragments = [
            metric_interval(
                category,
                correct_by_category.get(category, []),
                denominator=self.CATEGORY_COUNTS[category],
                unit=f"n_{category}",
            )
            for category in self.CATEGORY_COUNTS
        ]
        fragments.extend(self._group_intervals(correct_by_category))
        return result | merge_metrics(*fragments)

    def _group_intervals(
        self, correct_by_category: Mapping[str, list[float]]
    ) -> list[dict]:
        """Intervals for the group-level rollups, where any are defensible."""
        del correct_by_category
        return []


class BfclV3NonLiveTask[TSample](BfclV3Task[TSample]):
    """Non-live: seven categories, rolled up by UNWEIGHTED means.

    `simple_ast` nests three categories inside the five-way overall, so the
    headline is a mean of means -- not the pooled rate over 1390 rows, and not
    convergent to it: `javascript` (50 rows) enters with the same weight as
    `simple` (400). That is why it publishes no interval.
    """

    GROUP_KEY: ClassVar[str] = "non_live"
    CATEGORY_COUNTS: ClassVar[dict[str, int]] = dict(NON_LIVE_COUNTS)

    def _aggregate(self, cell: Callable[[str], dict]) -> dict:
        simple_ast = calculate_unweighted_accuracy(
            [cell("simple"), cell("java"), cell("javascript")]
        )
        ast_cells = [
            simple_ast,
            cell("multiple"),
            cell("parallel"),
            cell("parallel_multiple"),
        ]
        ast_summary = calculate_unweighted_accuracy(ast_cells)
        overall = calculate_unweighted_accuracy([*ast_cells, cell("irrelevance")])
        return {
            "non_live_overall_acc": overall["accuracy"],
            "ast_summary": ast_summary["accuracy"],
            "simple_ast": simple_ast["accuracy"],
            **{
                category: cell(category)["accuracy"]
                for category in self.CATEGORY_COUNTS
            },
        }


class BfclV3LiveTask[TSample](BfclV3Task[TSample]):
    """Live: six categories, rolled up by SAMPLE-COUNT-WEIGHTED means.

    A weighted mean over these cells is algebraically the pooled rate over the
    union of their rows, so both rollups are genuine per-sample rates and both
    carry intervals.
    """

    GROUP_KEY: ClassVar[str] = "live"
    CATEGORY_COUNTS: ClassVar[dict[str, int]] = dict(LIVE_COUNTS)

    #: The four AST categories, in upstream's column order. `ast_summary` is
    #: over these only -- irrelevance and relevance join at the overall.
    AST_CATEGORIES: ClassVar[tuple[str, ...]] = (
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
    )

    def _aggregate(self, cell: Callable[[str], dict]) -> dict:
        ast_summary = calculate_weighted_accuracy(
            [cell(category) for category in self.AST_CATEGORIES]
        )
        overall = calculate_weighted_accuracy(
            [cell(category) for category in self.CATEGORY_COUNTS]
        )
        return {
            "live_overall_acc": overall["accuracy"],
            "ast_summary": ast_summary["accuracy"],
            **{
                category: cell(category)["accuracy"]
                for category in self.CATEGORY_COUNTS
            },
        }

    def _group_intervals(
        self, correct_by_category: Mapping[str, list[float]]
    ) -> list[dict]:
        ast_values = [
            value
            for category in self.AST_CATEGORIES
            for value in correct_by_category.get(category, [])
        ]
        all_values = [
            value
            for category in self.CATEGORY_COUNTS
            for value in correct_by_category.get(category, [])
        ]
        return [
            # The headline and `live_overall_acc` are ONE number under two key
            # names, so the alias rides along on the same call -- never a second
            # call with the same arguments. This emits `n_problems` itself.
            interval_metrics(
                all_values,
                denominator=sum(self.CATEGORY_COUNTS.values()),
                aliases=("live_overall_acc",),
            ),
            # Emits `n_ast` itself, alongside the interval it is the unit for.
            metric_interval(
                "ast_summary",
                ast_values,
                denominator=sum(
                    self.CATEGORY_COUNTS[category] for category in self.AST_CATEGORIES
                ),
                unit="n_ast",
            ),
        ]


def _call_arguments(call) -> dict:
    """Read one tool call's arguments as a dict.

    `FunctionToolCall.arguments` is typed `JSONValue`, not `str`. The OpenAI
    dialects put the provider's wire JSON string there and substitute `""` when
    the provider omits it entirely, so a bare `json.loads` raises on a
    no-argument call -- a real shape for a zero-parameter tool.
    """
    arguments = call.arguments
    if isinstance(arguments, str):
        return json.loads(arguments) if arguments.strip() else {}
    return dict(arguments) if arguments else {}


class BfclV3PromptMixin:
    """Upstream's prompting protocol: schemas in the system turn, calls in text."""

    UNDERSCORE_TO_DOT: ClassVar[bool] = False

    def _build_messages(self, raw) -> tuple[list, list[dict] | None]:
        system = DEFAULT_SYSTEM_PROMPT.format(functions=raw["function"])
        return [{"role": "system", "content": system}, *raw["question"]], None

    def _decode(self, output: ModelOutput, language: str) -> list | None:
        # Deferred on purpose: `ast_parse` is the one symbol in the vendored
        # package that reaches tree-sitter, which lives in the optional
        # `bfcl-v3` extra. Importing a task module registers it, and
        # registration is paid by every task listing and by every eval that
        # fails before grading -- so at module scope this would make the extra
        # a hard requirement and break importing the task registry on a base
        # install. It is genuinely needed only here -- but here it is needed
        # for EVERY language, not just Java and JavaScript: `ast_parse` imports
        # both source parsers at module scope, so the first call pulls
        # tree-sitter in whichever grammar it ends up dispatching to.
        from sieval.community.bfcl_v3 import ast_parse

        text = output.texts[0] if output.texts else ""
        return ast_parse(text, language)


class BfclV3FCMixin:
    """Upstream's native-function-calling protocol: schemas as tools."""

    UNDERSCORE_TO_DOT: ClassVar[bool] = True

    requires = TaskRequirements(input=InputKind.CHAT, function_tools=True)

    def _build_messages(self, raw) -> tuple[list, list[dict] | None]:
        tools = convert_to_tool(
            json.loads(raw["function"]),
            GORILLA_TO_OPENAPI,
            ModelStyle.OpenAI_Completions,
        )
        return list(raw["question"]), tools

    def _decode(self, output: ModelOutput, language: str) -> list | None:
        del language  # Structured calls arrive typed; no source text is parsed.
        calls = output.tool_calls
        if not calls:
            return []
        return [{call.name: _call_arguments(call)} for call in calls]
