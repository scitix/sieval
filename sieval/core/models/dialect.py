"""Dialect contracts, request auditing, and response guarantees.

The keyed capability/reconcile plane is the primary setup mechanism.  This
module deliberately keeps the per-call gate small: it accounts for concrete
request leaves, rejects unavailable semantics, and validates root response
channels without a condition language or solver.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Protocol, runtime_checkable

from sieval.core.types import JSONValue

from .ir import (
    ChatInput,
    CompletionInput,
    FunctionToolCall,
    ImagePart,
    InputScoringResult,
    ReasoningOutput,
    Request,
    Response,
    StructuredOutput,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    TopKEntry,
    response_field_contract,
)


class DialectError(ValueError):
    """Base class for deterministic dialect contract failures."""


class RequestAuditError(DialectError):
    """A request leaf was rejected, omitted, or accounted for incorrectly."""


class OutputContractError(DialectError):
    """A successful wire reply violated its declared output contract."""


class Guarantee(StrEnum):
    PRESENT_OR_ERROR = "present_or_error"
    BEST_EFFORT = "best_effort"
    NEVER = "never"


@dataclass(frozen=True)
class Consumed:
    path: str


@dataclass(frozen=True)
class Rejected:
    path: str
    reason: str


@dataclass(frozen=True)
class NoOp:
    path: str
    reason: str


@dataclass(frozen=True)
class Passthrough:
    path: str
    destination: str


type AuditDecision = Consumed | Rejected | NoOp | Passthrough


@dataclass(frozen=True)
class PassthroughObservation:
    destination: str
    value: object


@dataclass(frozen=True)
class PreparedRequest:
    """Dialect-local wire preparation plus explicit audit evidence."""

    operation: str
    body: Mapping[str, JSONValue]
    consumed_paths: frozenset[str]
    passthrough: Mapping[str, PassthroughObservation]
    context: object | None = None


def _input_leaves(req: Request) -> dict[str, object]:
    if isinstance(req.input, CompletionInput):
        result: dict[str, object] = {"input.completion": req.input.text}
        if req.input.suffix is not None:
            result["input.completion.suffix"] = req.input.suffix
        return result

    if not isinstance(req.input, ChatInput):
        raise TypeError(
            f"Request.input has unsupported type {type(req.input).__name__}"
        )

    result = {"input.chat": req.input}
    part_paths: dict[type, str] = {
        TextPart: "input.modality.text",
        ImagePart: "input.modality.image",
        ToolCallPart: "input.modality.tool_call",
        ToolResultPart: "input.modality.tool_result",
    }
    # A record-level leaf covers fields that every admitted branch preserves.
    # Branch-sensitive optional fields need their own leaf so a dialect must
    # explicitly consume or reject them instead of hiding a partial drop.
    for message in req.input.messages:
        for part in message.content:
            path = part_paths.get(type(part))
            if path is None:
                raise TypeError(
                    f"ChatMessage contains unclassified part {type(part).__name__}"
                )
            result.setdefault(path, True)
            if isinstance(part, ImagePart) and part.media_type is not None:
                result.setdefault("input.modality.image.media_type", part.media_type)
            if isinstance(part, ToolResultPart) and part.is_error:
                result.setdefault("input.modality.tool_result.is_error", True)
    return result


def active_request_leaves(req: Request) -> dict[str, object]:
    """Return exact active semantic leaf paths for one concrete request."""

    active = _input_leaves(req)
    grouped = {
        "sampling": req.sampling,
        "scoring": req.scoring,
        "reasoning": req.reasoning,
        "tools": req.tools,
        "structured_output": req.structured_output,
        "session": req.session,
        "scheduling": req.scheduling,
    }
    for group_name, group in grouped.items():
        default = type(group)()
        for item in fields(group):
            value = getattr(group, item.name)
            if value != getattr(default, item.name):
                active[f"{group_name}.{item.name}"] = value

    if req.dialect_options is not None:
        if not req.dialect_options.values:
            active["dialect_options"] = req.dialect_options.dialect_id
        for key, value in req.dialect_options.values.items():
            active[f"dialect_options.{key}"] = value
    return active


def nondefault_leaf_paths(req: Request) -> frozenset[str]:
    return frozenset(active_request_leaves(req))


class RequestAudit:
    """Require exactly one observable decision for each active request leaf."""

    def __init__(self, active: Mapping[str, object]):
        self._active = dict(active)
        self._decisions: dict[str, AuditDecision] = {}

    @property
    def active(self) -> Mapping[str, object]:
        return dict(self._active)

    @property
    def decisions(self) -> Mapping[str, AuditDecision]:
        return dict(self._decisions)

    def _record(self, path: str, decision: AuditDecision) -> None:
        if path not in self._active:
            raise RequestAuditError(f"decision references inactive leaf {path!r}")
        if path in self._decisions:
            raise RequestAuditError(f"request leaf {path!r} was accounted for twice")
        self._decisions[path] = decision

    def consumed(self, path: str) -> None:
        self._record(path, Consumed(path))

    def rejected(self, path: str, reason: str) -> None:
        self._record(path, Rejected(path, reason))

    def noop(self, path: str, reason: str) -> None:
        if not reason.strip():
            raise RequestAuditError("NoOp requires documented semantic equivalence")
        self._record(path, NoOp(path, reason))

    def passthrough(self, path: str, destination: str) -> None:
        if not path.startswith("dialect_options."):
            raise RequestAuditError("only dialect options may use Passthrough")
        self._record(path, Passthrough(path, destination))

    def raise_rejections(self) -> None:
        rejected = [d for d in self._decisions.values() if isinstance(d, Rejected)]
        if rejected:
            detail = "; ".join(f"{d.path}: {d.reason}" for d in rejected)
            raise RequestAuditError(detail)

    def finish(self, prepared: PreparedRequest) -> None:
        missing = set(self._active) - set(self._decisions)
        if missing:
            raise RequestAuditError(
                "unaccounted request leaves: " + ", ".join(sorted(missing))
            )

        consumed = {
            path
            for path, decision in self._decisions.items()
            if isinstance(decision, Consumed)
        }
        if consumed != set(prepared.consumed_paths):
            missing_observation = consumed - set(prepared.consumed_paths)
            unclaimed = set(prepared.consumed_paths) - consumed
            detail: list[str] = []
            if missing_observation:
                detail.append("unobserved=" + ",".join(sorted(missing_observation)))
            if unclaimed:
                detail.append("unclaimed=" + ",".join(sorted(unclaimed)))
            raise RequestAuditError(
                "prepared consumption mismatch: " + "; ".join(detail)
            )

        expected_passthrough = {
            path: decision
            for path, decision in self._decisions.items()
            if isinstance(decision, Passthrough)
        }
        if set(expected_passthrough) != set(prepared.passthrough):
            raise RequestAuditError("prepared passthrough paths do not match decisions")
        for path, decision in expected_passthrough.items():
            observed = prepared.passthrough[path]
            if observed.destination != decision.destination:
                raise RequestAuditError(f"passthrough destination changed for {path!r}")
            if observed.value != self._active[path]:
                raise RequestAuditError(f"passthrough value changed for {path!r}")


def request_capability(path: str) -> str | None:
    """Map a concrete request leaf to its sole owning semantic capability."""

    exact = {
        "input.completion.suffix": "fim",
        "input.modality.image": "multimodal_input",
        "input.modality.image.media_type": "multimodal_input",
        "scoring.input_scoring": "input_scoring",
        "scoring.sampled_logprobs": "sampled_logprobs",
        "scoring.top_logprobs": "top_logprobs",
        "tools.hosted": "hosted_tools",
        "session.previous_response_id": "stateful_session",
        "session.opaque_continuation": "opaque_continuation",
    }
    if path in exact:
        return exact[path]
    prefixes = {
        "reasoning.": "reasoning",
        "tools.functions": "function_tools",
        "tools.choice": "function_tools",
        "tools.parallel": "function_tools",
        "structured_output.": "structured_output",
    }
    for prefix, capability in prefixes.items():
        if path.startswith(prefix):
            return capability
    return None


@runtime_checkable
class RuntimePlanView(Protocol):
    @property
    def dialect_id(self) -> str: ...

    @property
    def available_capabilities(self) -> frozenset[str]: ...

    @property
    def capability_minimums(self) -> Mapping[str, Mapping[str, JSONValue]]: ...

    @property
    def required_output_channels(self) -> frozenset[str]: ...


def validate_runtime_binding_plan(plan: RuntimePlanView, req: Request) -> None:
    if (
        req.dialect_options is not None
        and req.dialect_options.dialect_id != plan.dialect_id
    ):
        raise DialectError(
            f"dialect options target {req.dialect_options.dialect_id!r}, "
            f"but model is bound to {plan.dialect_id!r}"
        )
    continuation = req.session.opaque_continuation
    if continuation is not None and continuation.dialect_id != plan.dialect_id:
        raise DialectError(
            f"opaque continuation originated from {continuation.dialect_id!r}, "
            f"but model is bound to {plan.dialect_id!r}"
        )
    for path in nondefault_leaf_paths(req):
        capability = request_capability(path)
        if capability is not None and capability not in plan.available_capabilities:
            raise DialectError(
                f"request leaf {path!r} activates unavailable capability {capability!r}"
            )

    minimum = plan.capability_minimums.get("top_logprobs", {}).get("minimum")
    if (
        minimum is not None
        and req.scoring.top_logprobs > 0
        and req.scoring.top_logprobs < int(minimum)
    ):
        raise DialectError(
            f"top_logprobs={req.scoring.top_logprobs} weakens required "
            f"minimum {minimum}"
        )


def active_request_capabilities(req: Request) -> frozenset[str]:
    """Return semantic capabilities activated by this concrete request."""

    active: set[str] = set()
    for path in nondefault_leaf_paths(req):
        capability = request_capability(path)
        if capability is not None:
            active.add(capability)
    return frozenset(active)


def active_response_channels(req: Request) -> frozenset[str]:
    channels: set[str] = set()
    if any(
        value is not None
        for value in (
            req.reasoning.effort,
            req.reasoning.budget_tokens,
            req.reasoning.summary,
        )
    ):
        channels.add("reasoning")
    if req.scoring.sampled_logprobs:
        channels.add("logprobs")
    if req.scoring.top_logprobs > 0:
        channels.add("top_logprobs")
    if req.scoring.input_scoring:
        channels.add("input_scoring")
    if req.tools.functions:
        channels.add("tool_calls")
    if req.tools.hosted:
        channels.add("server_tool_uses")
    if req.structured_output.format is not None:
        channels.add("structured_output")
    if req.session.previous_response_id is not None:
        channels.add("session_id")
    if req.session.opaque_continuation is not None:
        channels.add("reasoning")
    return frozenset(channels)


def required_response_channels(req: Request) -> frozenset[str]:
    required = set(active_response_channels(req))
    # Supplying function/hosted tools does not require the model to invoke one.
    required.discard("tool_calls")
    required.discard("server_tool_uses")
    if (
        req.reasoning.summary in (None, "none")
        and req.session.opaque_continuation is None
    ):
        required.discard("reasoning")
    return frozenset(required)


def validate_request_invariants(req: Request) -> None:
    sampling = req.sampling
    for name in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
        value = getattr(sampling, name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int | float)
        ):
            raise TypeError(f"sampling.{name} must be a number")
    for name in ("top_k", "max_tokens", "seed"):
        value = getattr(sampling, name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise TypeError(f"sampling.{name} must be an int")
    if sampling.stop is not None and (
        not isinstance(sampling.stop, tuple)
        or not all(isinstance(item, str) for item in sampling.stop)
    ):
        raise TypeError("sampling.stop must be a tuple of strings")
    if isinstance(req.sampling.n, bool) or not isinstance(req.sampling.n, int):
        raise TypeError("sampling.n must be an int")
    if req.sampling.n < 1:
        raise ValueError("sampling.n must be >= 1")
    for name in ("input_scoring", "sampled_logprobs"):
        if not isinstance(getattr(req.scoring, name), bool):
            raise TypeError(f"scoring.{name} must be a bool")
    if isinstance(req.scoring.top_logprobs, bool) or not isinstance(
        req.scoring.top_logprobs, int
    ):
        raise TypeError("scoring.top_logprobs must be an int")
    if req.scoring.top_logprobs < 0:
        raise ValueError("scoring.top_logprobs must be >= 0")
    if req.scoring.top_logprobs > 0 and not req.scoring.sampled_logprobs:
        raise ValueError("top_logprobs requires sampled_logprobs")
    if req.reasoning.effort is not None and not isinstance(req.reasoning.effort, str):
        raise TypeError("reasoning.effort must be a string")
    if req.reasoning.budget_tokens is not None and (
        isinstance(req.reasoning.budget_tokens, bool)
        or not isinstance(req.reasoning.budget_tokens, int)
    ):
        raise TypeError("reasoning.budget_tokens must be an int")
    if req.reasoning.summary not in {None, "none", "auto", "concise", "detailed"}:
        raise ValueError("reasoning.summary is invalid")
    if req.tools.parallel is not None and not isinstance(req.tools.parallel, bool):
        raise TypeError("tools.parallel must be a bool")
    if not isinstance(req.scheduling.stream, bool):
        raise TypeError("scheduling.stream must be a bool")

    if req.sampling.n > 1:
        contract = response_field_contract()
        singular = sorted(
            channel
            for channel in active_response_channels(req)
            if not contract[channel][1]
        )
        if singular:
            raise ValueError(
                f"sampling.n={req.sampling.n} is incompatible with singular "
                "response channels: " + ", ".join(singular)
            )


type OutputValidator = Callable[[object], None]


def _accept(_: object) -> None:
    return


@dataclass(frozen=True)
class OutputRule:
    guarantee: Guarantee
    validator: OutputValidator = _accept


@dataclass(frozen=True)
class OutputContract:
    rules: Mapping[str, OutputRule]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", dict(self.rules))
        channels = {
            name
            for name, (role, _) in response_field_contract().items()
            if role == "channel"
        }
        missing = channels - set(self.rules)
        extra = set(self.rules) - channels
        if missing or extra:
            raise ValueError(
                f"output contract incomplete: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

    def validate(self, plan: RuntimePlanView, req: Request, response: Response) -> None:
        # Binding-level requirements are setup/evidence about calls this model
        # must be able to serve.  Only the projected request decides which
        # response channels this particular call must return.
        required = set(required_response_channels(req))
        for name, rule in self.rules.items():
            value = getattr(response, name)
            if rule.guarantee is Guarantee.NEVER and value is not None:
                raise OutputContractError(f"dialect promised {name!r} would be absent")
            if name in required:
                if rule.guarantee is not Guarantee.PRESENT_OR_ERROR:
                    raise OutputContractError(
                        f"{name!r} is required but has {rule.guarantee.value} guarantee"
                    )
                if value is None:
                    raise OutputContractError(
                        f"required response channel {name!r} is absent"
                    )
            if value is not None:
                rule.validator(value)

        if req.session.opaque_continuation is not None:
            reasoning = response.reasoning
            if reasoning is None or any(
                item is None
                or not isinstance(item, ReasoningOutput)
                or not isinstance(item.opaque_roundtrip, str)
                or not item.opaque_roundtrip
                for item in reasoning
            ):
                raise OutputContractError(
                    "opaque continuation requires a non-empty opaque round-trip "
                    "payload for every choice"
                )

        if len(response.texts) != req.sampling.n:
            raise OutputContractError(
                f"expected {req.sampling.n} choices, received {len(response.texts)}"
            )


def validate_reasoning(value: object) -> None:
    if not isinstance(value, tuple) or not all(
        item is None or isinstance(item, ReasoningOutput) for item in value
    ):
        raise OutputContractError("reasoning channel has invalid shape")


def validate_tool_calls(value: object) -> None:
    if not isinstance(value, tuple) or not all(
        isinstance(item, FunctionToolCall) for item in value
    ):
        raise OutputContractError("tool_calls channel has invalid shape")


def validate_top_logprobs(value: object) -> None:
    if not isinstance(value, tuple) or not all(
        isinstance(position, tuple)
        and all(isinstance(item, TopKEntry) for item in position)
        for position in value
    ):
        raise OutputContractError("top_logprobs channel has invalid shape")


def validate_input_scoring(value: object) -> None:
    if not isinstance(value, InputScoringResult):
        raise OutputContractError("input_scoring channel has invalid shape")


def validate_structured_output(value: object) -> None:
    if not isinstance(value, StructuredOutput):
        raise OutputContractError("structured_output channel has invalid shape")


@runtime_checkable
class Dialect(Protocol):
    dialect_id: str
    connection_family: str
    output_contract: OutputContract

    def validate_request(
        self, req: Request, audit: RequestAudit, plan: RuntimePlanView
    ) -> None: ...

    def prepare(self, req: Request, audit: RequestAudit) -> PreparedRequest: ...

    async def execute(self, prepared: PreparedRequest) -> Response: ...
