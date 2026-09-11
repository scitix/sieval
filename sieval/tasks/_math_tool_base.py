"""Shared machinery for math tasks that may run Python while solving.

The loop, the protocol adapters and the sandbox client live here; a leaf binds a
benchmark's dataset and reuses that benchmark's own prompt, extractor and grader
unchanged. Keeping the grader identical to the no-tool sibling is what makes the
pair a measurable difference rather than two unrelated numbers.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import asyncio
import os
import re
from abc import abstractmethod
from dataclasses import dataclass, field

import httpx
from loguru import logger

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    Task,
    TaskStageOutput,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    health_metrics,
    sampling_report,
    ungated_intervals,
)
from sieval.core.utils.meta import build_stage_meta
from sieval.core.utils.offload import GRADE_TIMEOUT, run_cpu_bound
from sieval.core.utils.serialization import sieval_record

from ._math_verify import normalize_vote, verify_answer

#: Mirrors the service's own cap. Kept here as well so a client pointed at an
#: older service still bounds what reaches a shard record and the model's context.
MAX_STREAM_CHARS = 8192

DEFAULT_CODE_RUN_API = "http://localhost:11451/code-runs"


def _get[T](data: dict, key: str, default: T) -> T:
    """Read *key* from *data*, filling in for both "absent" and "explicit null".

    ``dict.get(key, default)`` substitutes *default* only when the key is
    missing; a service that sends the key with an explicit ``null`` -- valid
    JSON regardless of what this route's schema currently promises -- passes
    ``None`` through unchanged, and a caller that expected *default*'s type
    breaks on it instead. Every optional-but-typed field pulled out of a
    response goes through this one function, so a field added later inherits
    the guard rather than needing its own copy of this reasoning.
    """
    value = data.get(key, default)
    return default if value is None else value


@sieval_record
@dataclass
class ToolCall:
    """One code execution: what was sent, and everything that came back."""

    index: int
    code: str
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    truncated: bool
    wall_s: float
    #: Reserved for a future stateful backend; always None while execution is
    #: one fresh process per call. Absent rather than null once serialized.
    session_id: str | None = None
    #: Text the model generated after its closing fence, which was discarded
    #: rather than sent. Empty in the ordinary case. Recorded because a model
    #: that writes its own fake tool output is scoring on a hallucination, and
    #: the rate has to be measurable rather than assumed.
    discarded_tail: str = ""


@sieval_record
@dataclass
class RolloutTrajectory:
    """One rollout's whole solve: every model call and every execution."""

    index: int
    outputs: list[ModelOutput] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: "answered" | "budget_exhausted" | "no_tool_use" | "request_failed"
    stop_reason: str = "answered"


@sieval_record
@dataclass
class MathToolTrajectory:
    """Every rollout of one sample, plus what produced them."""

    rollouts: list[RolloutTrajectory]
    protocol: str
    sandbox: dict


class SandboxClient:
    """One code-evaluator `/code-runs` endpoint, as a source of `ToolCall`s."""

    def __init__(
        self,
        *,
        api: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float,
        memory_limit: int = 1024,
        max_connections: int = 8,
    ) -> None:
        self._api = (
            api if api else os.getenv("SIEVAL_CODE_RUN_API", DEFAULT_CODE_RUN_API)
        )
        self._client = http_client or httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections)
        )
        self._timeout = timeout
        self._memory_limit = memory_limit
        self.service_version: str | None = None

    async def run(self, code: str) -> ToolCall:
        """Execute *code*; never raise.

        A sandbox that is down, slow, malformed, or has drifted from this
        client's expected response shape produces a ToolCall carrying the
        reason, which the loop feeds back to the model and the record keeps.
        Raising instead would fail the sample -- and under DENOMINATOR_REQUESTED
        a failed sample is charged as wrong, so one flaky request or one
        unexpected null would read as a model that could not do arithmetic.
        Mapping the response into a ToolCall is inside the same guarded region
        as sending the request, rather than a narrower guard that trusts the
        body once it parses as JSON: a response shaped unexpectedly -- not a
        dict at all, or a field null where today's contract has it required --
        is exactly as unpredictable as a response that never arrives, and both
        need the same one fallback.
        """
        try:
            resp = await self._client.post(
                self._api,
                json={
                    "uuid": f"{id(self):x}-{len(code)}",
                    "lang": "python",
                    "code": code,
                    "timeout": self._timeout,
                    "memory_limit": self._memory_limit,
                },
                # Above the service's own wall, so a server-side timeout comes
                # back as a RESULT (timed_out=True) rather than as a client
                # abort that cannot tell a slow snippet from a dead service.
                timeout=self._timeout + 15.0,
            )
            resp.raise_for_status()
            body = resp.json()
            data = body.get("data") or {}
            self.service_version = _get(data, "service_version", self.service_version)
            return ToolCall(
                index=0,
                code=code,
                stdout=_get(data, "stdout", "")[:MAX_STREAM_CHARS],
                stderr=(_get(data, "stderr", "") or _get(body, "msg", ""))[
                    :MAX_STREAM_CHARS
                ],
                exit_code=data.get("exit_code"),
                timed_out=bool(_get(data, "timed_out", False)),
                truncated=bool(_get(data, "truncated", False)),
                wall_s=float(_get(data, "wall_s", 0.0)),
                session_id=data.get("session_id"),
            )
        except Exception as exc:
            return ToolCall(
                index=0,
                code=code,
                stdout="",
                stderr=f"sandbox unreachable: [{type(exc).__name__}] {exc}",
                exit_code=None,
                timed_out=False,
                truncated=False,
                wall_s=0.0,
            )


#: The protocol, stated to the model. PINNED: this string is part of what the
#: task measures, so rewording it moves every score and makes a stored delta
#: incomparable to a fresh one. A test asserts its exact bytes.
TOOL_SYSTEM_PROMPT = (
    "You may run Python to help you compute. To do so, write a single fenced "
    "block:\n"
    "```python\n"
    "# your code; print() what you need to see\n"
    "```\n"
    "Stop immediately after the closing fence. The program's output will be "
    "given to you in the next message, and you may then run more code or give "
    "your final answer. Code you do not print produces no output. Each block "
    "runs in a fresh interpreter, so repeat any definitions you still need."
)

#: Stop sequence. Ending generation at the closing fence is what keeps the model
#: from inventing its own tool output; the adapter still truncates, because not
#: every backend honours `stop`.
FENCE_STOP = "```\n"

_FENCE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


class TextToolAdapter:
    """The fenced-code-block protocol: any chat model can speak it."""

    protocol = "text"

    def system_prompt(self) -> str:
        return TOOL_SYSTEM_PROMPT

    def extract_call(self, output: ModelOutput) -> tuple[str, str] | None:
        """``(code, discarded_tail)``, or None when the model did not call.

        None covers both "answered" and "produced nothing usable" -- an
        unterminated fence included, because executing a program the model was
        cut off mid-way through would grade a syntax error it never wrote.
        """
        if not output.texts:
            return None
        text = output.texts[0]
        match = _FENCE.search(text)
        if match is None:
            return None
        code = match.group(1)
        if not code.strip():
            return None
        return code, text[match.end() :]

    def render(self, call: ToolCall) -> list[dict]:
        """The assistant turn that made the call, and the result turn.

        The assistant turn is rebuilt from the CODE rather than replayed from the
        reply, so the invented tail is not smuggled back into context by the very
        message that reports the real output.
        """
        if call.timed_out:
            result = f"The code timed out and was stopped.\n{call.stderr}".strip()
        elif call.exit_code != 0:
            result = (call.stderr or "The code failed with no output.").strip()
        elif not call.stdout.strip():
            result = "The code ran and printed nothing."
        else:
            result = call.stdout.rstrip()
        if call.truncated:
            result += "\n[output truncated]"
        return [
            {"role": "assistant", "content": f"```python\n{call.code}```"},
            {"role": "user", "content": f"Output:\n```\n{result}\n```"},
        ]


async def run_tool_loop(
    *,
    model,
    sandbox,
    adapter,
    messages: list[dict],
    index: int,
    max_tool_calls: int,
) -> RolloutTrajectory:
    """One rollout: alternate model turns and executions until it answers.

    `n=1` throughout. A rollout is one branch of the solve, and `agenerate(n=k)`
    would return k texts from a single call with no way to continue each of them
    against its own tool output -- so the budget is spent as k separate loops by
    the caller, not as one wide request here.
    """
    trajectory = RolloutTrajectory(index=index)
    turn = list(messages)
    for attempt in range(max_tool_calls + 1):
        try:
            # `stop` ends generation AT the closing fence, so a well-behaved
            # backend never gets the chance to invent its own tool output. The
            # adapter still truncates: not every backend honours `stop`, and the
            # one that ignores it is exactly the one this guards against.
            output = await model.agenerate(turn, n=1, stop=[FENCE_STOP])
        except Exception as exc:
            if not trajectory.outputs:
                # Nothing was answered, so there is nothing to salvage: fail the
                # sample the way a single-turn task's would.
                raise
            # Broad on purpose: anything reaching here has already exhausted
            # `max_retries`. Never silent -- the rollout says why it stopped.
            logger.warning(
                "tool loop rollout {} lost its request after {} turn(s): {}",
                index,
                len(trajectory.outputs),
                exc,
            )
            trajectory.stop_reason = "request_failed"
            return trajectory
        trajectory.outputs.append(output)

        extracted = adapter.extract_call(output)
        if extracted is None:
            trajectory.stop_reason = (
                "answered" if trajectory.tool_calls else "no_tool_use"
            )
            return trajectory
        if attempt == max_tool_calls:
            # The budget is spent. The reply above is the last word, whether or
            # not it holds an answer -- reported so a score a small budget
            # explains is not mistaken for one capability explains.
            trajectory.stop_reason = "budget_exhausted"
            return trajectory

        code, tail = extracted
        call = await sandbox.run(code)
        call.index = len(trajectory.tool_calls)
        call.discarded_tail = tail
        trajectory.tool_calls.append(call)
        turn.extend(adapter.render(call))
    # Unreachable: the `attempt == max_tool_calls` branch returns on the last
    # pass. Kept so the function has one exit type rather than an implicit None.
    trajectory.stop_reason = "budget_exhausted"
    return trajectory


class MathToolTask[TRawSample](
    Task[
        TRawSample,
        PromptRecord,
        TaskStageOutput[MathToolTrajectory],
        PredictionRecord,
        JudgementRecord,
        dict[str, float | str | list[float] | dict[str, str]],
    ]
):
    """A math benchmark whose model may run Python while solving it.

    Everything scored is the sibling task's: the same prompt template, the same
    answer pattern, the same verifier, the same denominator. Only the affordance
    differs, which is what makes the two numbers a difference rather than two
    measurements.
    """

    #: Set by the leaf: the benchmark's own query template and answer pattern.
    query_template: str
    answer_pattern: str

    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        k: int = 1,
        n: int = 1,
        *,
        max_tool_calls: int,
        tool_timeout: float = 10.0,
        tool_memory_limit: int = 1024,
        code_run_api: str | None = None,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        if k > n:
            raise ValueError(
                f"pass@{k} needs at least {k} sample(s) per problem, got n={n}."
            )
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")
        self._k = k
        self._n = n
        self._max_tool_calls = max_tool_calls
        self._adapter = TextToolAdapter()
        self._sandbox = SandboxClient(
            api=code_run_api,
            timeout=tool_timeout,
            memory_limit=tool_memory_limit,
            max_connections=max(8, n),
        )

    @abstractmethod
    def _question(self, raw) -> str:
        """The problem text this benchmark puts in the prompt."""

    @abstractmethod
    def _reference(self, raw) -> str:
        """This benchmark's gold answer."""

    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [
                {"role": "system", "content": self._adapter.system_prompt()},
                {
                    "role": "user",
                    "content": self.query_template.format(problem=self._question(raw)),
                },
            ],
            reference=self._reference(raw),
        )

    async def infer(self, pre, ctx):
        rollouts = await asyncio.gather(
            *(
                run_tool_loop(
                    model=self.model,
                    sandbox=self._sandbox,
                    adapter=self._adapter,
                    messages=list(pre["prompt"]),
                    index=index,
                    max_tool_calls=self._max_tool_calls,
                )
                for index in range(self._n)
            )
        )
        trajectory = MathToolTrajectory(
            rollouts=list(rollouts),
            protocol=self._adapter.protocol,
            sandbox={"service_version": self._sandbox.service_version},
        )
        flat = [output for rollout in rollouts for output in rollout.outputs]
        # Boxed with explicit meta: the runner derives `model_calls` only from a
        # ModelOutput or a list of them, and a composite value would silently
        # contribute none -- the stage's whole token spend missing from the
        # profile, with nothing on disk to say so.
        return TaskStageOutput(value=trajectory, meta=build_stage_meta(*flat))

    async def postprocess(self, inf, ctx):
        predictions: list[str | None] = []
        extras: list[dict | None] = []
        for rollout in inf.value.rollouts:
            text = (
                rollout.outputs[-1].texts[0]
                if (rollout.outputs and rollout.outputs[-1].texts)
                else ""
            )
            match = re.search(self.answer_pattern, text)
            predictions.append(match.group(1).strip() if match else None)
            extras.append(
                {
                    "n_tool_calls": len(rollout.tool_calls),
                    "stop_reason": rollout.stop_reason,
                    "n_execution_errors": sum(
                        1
                        for call in rollout.tool_calls
                        if call.exit_code not in (0, None) and not call.timed_out
                    ),
                    "n_tool_timeouts": sum(
                        1 for call in rollout.tool_calls if call.timed_out
                    ),
                    "n_discarded_tails": sum(
                        1 for call in rollout.tool_calls if call.discarded_tail.strip()
                    ),
                }
            )
        return build_prediction_record(predictions, extras=extras)

    async def feedback(self, post, ctx):
        rollouts = []
        ground_truth = self._reference(ctx.raw_sample)
        for rollout in post["rollouts"]:
            pred = rollout.get("prediction")
            if pred is None:
                rollouts.append(build_rollout_judgement(rollout["index"], False))
                continue
            try:
                correct = await run_cpu_bound(
                    verify_answer,
                    f"${ground_truth}$",
                    f"${pred}$",
                    timeout=GRADE_TIMEOUT,
                )
            except TimeoutError:
                # A grade that could not be COMPUTED IN TIME is a wrong answer:
                # the prediction is a shape the grader cannot bound. Every other
                # exception propagates, so a broken grader is distinguishable
                # from a model that answered wrongly.
                logger.warning(
                    "Grading sample {} exceeded {}s and was scored wrong.",
                    ctx.sample_id,
                    GRADE_TIMEOUT,
                )
                correct = False
            rollouts.append(build_rollout_judgement(rollout["index"], correct))
        return True, build_judgement_record(ground_truth, rollouts)

    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        rolled = sampling_report(
            finals,
            n=self._n,
            k=self._k,
            denominator=total,
            normalize=normalize_vote,
            score_key="pass@1",
            grouping=self.problem_groups(finals),
        )
        pass_at_1 = rolled["pass@1"]
        metrics: dict[str, float | str | list[float] | dict[str, str]] = {
            "score": pass_at_1,
            "fails": len(fails),
            "pass@1": pass_at_1,
            SCORE_KEY_FIELD: "pass@1",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        metrics |= ungated_intervals(rolled, metrics=("score", "pass@1"))
        if self._n > 1:
            metrics.update(rolled)
        return metrics | health_metrics(finals) | self._tool_metrics(finals)

    def _tool_metrics(self, finals) -> dict[str, float]:
        """Counts, not rates: each is a deterministic function of the trajectory.

        No interval on any of them -- an interval belongs to a measurement of the
        population the headline is clustered on, and these are tallies over it.
        """
        totals = {
            "n_tool_calls": 0.0,
            "n_rollouts_using_tool": 0.0,
            "n_execution_errors": 0.0,
            "n_tool_timeouts": 0.0,
            "n_budget_exhausted": 0.0,
            "n_discarded_tails": 0.0,
        }
        for final in finals:
            record = final.postprocess_result or {}
            for rollout in record.get("rollouts", ()):
                extra = rollout.get("extra") or {}
                calls = extra.get("n_tool_calls", 0)
                totals["n_tool_calls"] += calls
                totals["n_rollouts_using_tool"] += 1.0 if calls else 0.0
                totals["n_execution_errors"] += extra.get("n_execution_errors", 0)
                totals["n_tool_timeouts"] += extra.get("n_tool_timeouts", 0)
                totals["n_discarded_tails"] += extra.get("n_discarded_tails", 0)
                if extra.get("stop_reason") == "budget_exhausted":
                    totals["n_budget_exhausted"] += 1.0
        return totals
