"""OpenAI ``/v1/completions`` dialect for the provider-neutral model IR.

The completions wire is the sole dialect that implements input scoring through
``echo=True``.  Echoed prompt tokens are separated from sampled tokens only at
the usage-reported prompt-token boundary; an absent or contradictory boundary
is a protocol error rather than an invitation to guess.

The dialect borrows its OpenAI client.  Connection admission and lifecycle stay
with the connection pool that constructed the bound model.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import math
from collections.abc import AsyncIterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    Capability,
    CapabilityKey,
    DialectCapabilityBinding,
    DialectCapabilityDecision,
    Supported,
    Unsupported,
)
from sieval.core.models.deployment import BINDING_RESOURCE_KEYS
from sieval.core.models.dialect import (
    DialectError,
    Guarantee,
    OutputContract,
    OutputContractError,
    OutputRule,
    PassthroughObservation,
    PreparedRequest,
    RequestAudit,
    RequestAuditError,
    RuntimePlanView,
    active_request_leaves,
    validate_input_scoring,
    validate_request_invariants,
    validate_runtime_binding_plan,
    validate_top_logprobs,
)
from sieval.core.models.ir import (
    CompletionInput,
    InputScoringResult,
    Request,
    Response,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)
from sieval.core.types import JSONValue

_PREFILL_UNSUPPORTED_REASON = (
    "assistant prefill is a chat input operation, not a completion operation"
)


def _decisions() -> Mapping[CapabilityKey, DialectCapabilityDecision]:
    decisions: dict[CapabilityKey, DialectCapabilityDecision] = {
        "input_scoring": Supported(
            DialectCapabilityBinding(
                "input_scoring",
                request_leaves=("scoring.input_scoring",),
                response_channels=("input_scoring",),
            )
        ),
        "sampled_logprobs": Supported(
            DialectCapabilityBinding(
                "sampled_logprobs",
                request_leaves=("scoring.sampled_logprobs",),
                response_channels=("logprobs",),
            )
        ),
        "top_logprobs": Supported(
            DialectCapabilityBinding(
                "top_logprobs",
                request_leaves=("scoring.top_logprobs",),
                response_channels=("top_logprobs",),
            )
        ),
        "reasoning": Unsupported(
            "the completions endpoint has no reasoning-control protocol"
        ),
        "function_tools": Unsupported(
            "the completions endpoint has no function-tool protocol"
        ),
        "hosted_tools": Unsupported(
            "the completions endpoint has no hosted-tool protocol"
        ),
        "structured_output": Unsupported(
            "the completions endpoint has no structured-output protocol"
        ),
        "stateful_session": Unsupported(
            "the completions endpoint has no response-id continuation"
        ),
        "opaque_continuation": Unsupported(
            "the completions endpoint has no opaque continuation channel"
        ),
        "multimodal_input": Unsupported(
            "the completions endpoint accepts text prompts only"
        ),
        "prefill": Unsupported(_PREFILL_UNSUPPORTED_REASON),
        "fim": Supported(
            DialectCapabilityBinding("fim", request_leaves=("input.completion.suffix",))
        ),
    }
    if set(decisions) != set(CAPABILITY_KEYS):
        raise AssertionError("OpenAI completions capability row is incomplete")
    return MappingProxyType(decisions)


def _validate_token_logprobs(value: object) -> None:
    if not isinstance(value, tuple) or not all(
        isinstance(item, TokenLogprob) for item in value
    ):
        raise OutputContractError("logprobs channel has invalid shape")


CAPABILITY_DECISIONS = _decisions()


OUTPUT_CONTRACT = OutputContract(
    {
        "reasoning": OutputRule(Guarantee.NEVER),
        "tool_calls": OutputRule(Guarantee.NEVER),
        "server_tool_uses": OutputRule(Guarantee.NEVER),
        "structured_output": OutputRule(Guarantee.NEVER),
        "logprobs": OutputRule(Guarantee.PRESENT_OR_ERROR, _validate_token_logprobs),
        "top_logprobs": OutputRule(Guarantee.PRESENT_OR_ERROR, validate_top_logprobs),
        "input_scoring": OutputRule(Guarantee.PRESENT_OR_ERROR, validate_input_scoring),
        "citations": OutputRule(Guarantee.NEVER),
        "grounding": OutputRule(Guarantee.NEVER),
        "session_id": OutputRule(Guarantee.NEVER),
        "usage": OutputRule(Guarantee.BEST_EFFORT),
    }
)


_CONSUMED_PATHS = frozenset(
    {
        "input.completion",
        "input.completion.suffix",
        "sampling.temperature",
        "sampling.top_p",
        "sampling.top_k",
        "sampling.max_tokens",
        "sampling.stop",
        "sampling.seed",
        "sampling.frequency_penalty",
        "sampling.presence_penalty",
        "sampling.n",
        "scoring.input_scoring",
        "scoring.sampled_logprobs",
        "scoring.top_logprobs",
        "scheduling.stream",
    }
)

_IR_OWNED_BODY_KEYS = BINDING_RESOURCE_KEYS | frozenset(
    {
        "model",
        "prompt",
        "suffix",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "n",
        "echo",
        "score_input",
        "return_logprobs",
        "logprobs",
        "prompt_logprobs",
        "top_logprobs",
        "reasoning",
        "reasoning_effort",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "server_tools",
        "response_format",
        "previous_response_id",
        "session_id",
        "opaque_continuation",
        "stream",
        "stream_options",
        "extra_body",
    }
)


@dataclass(frozen=True)
class _CompletionContext:
    n: int
    input_scoring: bool
    stream: bool
    request_params: Mapping[str, JSONValue]


@dataclass(frozen=True)
class _LegacyPlan:
    """Minimal plan used only by the deprecated direct-transport call path."""

    dialect_id: str = "openai_completions"
    available_capabilities: frozenset[str] = frozenset(
        key
        for key, decision in CAPABILITY_DECISIONS.items()
        if isinstance(decision, Supported)
    )
    capability_minimums: Mapping[str, Mapping[str, JSONValue]] = field(
        default_factory=dict
    )
    required_output_channels: frozenset[str] = frozenset()


def _ensure_matching_audit(req: Request, audit: RequestAudit) -> None:
    active = active_request_leaves(req)
    if audit.active != active:
        missing = sorted(set(active) - set(audit.active))
        extra = sorted(set(audit.active) - set(active))
        raise RequestAuditError(
            f"audit does not match request: missing={missing}, extra={extra}"
        )


def _completion_top_logprobs(raw: object) -> list[dict[str, float]] | None:
    """Validate completions-style per-position alternative-token mappings."""

    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise OutputContractError("completion top_logprobs must be a sequence")

    top_logprobs: list[dict[str, float]] = []
    for position, item in enumerate(raw):
        if item is None:
            top_logprobs.append({})
            continue
        if not isinstance(item, Mapping):
            raise OutputContractError(
                f"completion top_logprobs position {position} must be a mapping or None"
            )
        parsed: dict[str, float] = {}
        for token, logprob in item.items():
            if not isinstance(token, str):
                raise OutputContractError(
                    "completion top_logprobs position "
                    f"{position} has a non-string token"
                )
            if isinstance(logprob, bool) or not isinstance(logprob, int | float):
                raise OutputContractError(
                    f"completion top_logprobs[{position}][{token!r}] must be numeric"
                )
            normalized = float(logprob)
            if not math.isfinite(normalized):
                raise OutputContractError(
                    f"completion top_logprobs[{position}][{token!r}] must be finite"
                )
            parsed[token] = normalized
        top_logprobs.append(parsed)
    return top_logprobs


def _optional_response_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise OutputContractError(f"completions {path} must be a string or None")
    return value


def _completions_usage_stats(raw: object) -> UsageStats | None:
    if raw is None:
        return None

    names = ("prompt_tokens", "completion_tokens", "total_tokens")
    values: list[int] = []
    for name in names:
        value = getattr(raw, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OutputContractError(
                f"completions usage.{name} must be a non-negative integer"
            )
        values.append(value)
    return UsageStats(
        input_tokens=values[0],
        output_tokens=values[1],
        total_tokens=values[2],
    )


def _logprob_sequence(raw: object, path: str) -> Sequence[object]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise OutputContractError(f"{path} must be a sequence")
    return raw


def _sampled_logprob(raw: object, path: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise OutputContractError(f"{path} must be numeric or None")
    normalized = float(raw)
    if not math.isfinite(normalized):
        raise OutputContractError(f"{path} must be finite")
    return normalized


def _choice_index(choice: object, n: int) -> int:
    index = getattr(choice, "index", None)
    if isinstance(index, bool) or not isinstance(index, int):
        raise OutputContractError("completion choice index must be an integer")
    if not 0 <= index < n:
        raise OutputContractError(
            f"completion choice index {index} is outside the requested range [0, {n})"
        )
    return index


def _choice_sequence(raw: object) -> Sequence[object]:
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise OutputContractError("completion response choices must be a sequence")
    return raw


class OpenAICompletionsDialect:
    """Executable dialect for OpenAI-compatible completions clients."""

    dialect_id = "openai_completions"
    connection_family = "openai_sdk"
    capability_decisions = CAPABILITY_DECISIONS
    output_contract = OUTPUT_CONTRACT
    CAPABILITIES = frozenset(
        {
            Capability.Completion,
            Capability.FIM,
            Capability.InputScoring,
            Capability.SampledLogprobs,
            Capability.TopKLogprobs,
        }
    )

    def __init__(self, client: Any, model: str) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self._client = client
        self._model = model

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Deprecated enum view retained with the transport import alias."""

        return self.CAPABILITIES

    def validate_request(
        self, req: Request, audit: RequestAudit, plan: RuntimePlanView
    ) -> None:
        """Reject non-completion semantics before any client operation."""

        _ensure_matching_audit(req, audit)
        for path in audit.active:
            if path in _CONSUMED_PATHS or path == "dialect_options":
                continue
            if path.startswith("dialect_options."):
                key = path.removeprefix("dialect_options.")
                if key in {"prefill", "prefix"}:
                    audit.rejected(path, _PREFILL_UNSUPPORTED_REASON)
                elif key in _IR_OWNED_BODY_KEYS:
                    audit.rejected(
                        path,
                        f"{key!r} is owned by a first-class request field",
                    )
                continue
            audit.rejected(
                path,
                "the OpenAI completions dialect does not support this request leaf",
            )

        del plan

    def prepare(self, req: Request, audit: RequestAudit) -> PreparedRequest:
        """Lower one validated request and attach exact audit observations."""

        _ensure_matching_audit(req, audit)
        audit.raise_rejections()
        if not isinstance(req.input, CompletionInput):
            raise DialectError("OpenAI completions requires CompletionInput")

        body: dict[str, Any] = {
            "model": self._model,
            "prompt": req.input.text,
            "stream": req.scheduling.stream,
        }
        consumed: set[str] = set()
        passthrough: dict[str, PassthroughObservation] = {}

        def consume(path: str) -> None:
            audit.consumed(path)
            consumed.add(path)

        consume("input.completion")
        if req.input.suffix is not None:
            body["suffix"] = req.input.suffix
            consume("input.completion.suffix")

        sampling = req.sampling
        simple_sampling = {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "seed": sampling.seed,
            "frequency_penalty": sampling.frequency_penalty,
            "presence_penalty": sampling.presence_penalty,
        }
        for name, value in simple_sampling.items():
            if value is not None:
                body[name] = value
                consume(f"sampling.{name}")
        if sampling.stop is not None:
            body["stop"] = list(sampling.stop)
            consume("sampling.stop")
        if sampling.n != 1:
            body["n"] = sampling.n
            consume("sampling.n")

        extra_body: dict[str, JSONValue] = {}
        if sampling.top_k is not None:
            extra_body["top_k"] = sampling.top_k
            consume("sampling.top_k")

        scoring = req.scoring
        if scoring.input_scoring:
            body["echo"] = True
            consume("scoring.input_scoring")
        if scoring.sampled_logprobs:
            consume("scoring.sampled_logprobs")
        if scoring.top_logprobs > 0:
            consume("scoring.top_logprobs")
        if scoring.sampled_logprobs or scoring.input_scoring:
            body["logprobs"] = scoring.top_logprobs

        if req.scheduling.stream:
            consume("scheduling.stream")

        options = req.dialect_options
        if options is not None:
            if not options.values:
                audit.noop(
                    "dialect_options",
                    "an empty options mapping has no wire semantics",
                )
            for key, option_value in options.values.items():
                path = f"dialect_options.{key}"
                extra_body[key] = option_value
                audit.passthrough(path, "extra_body")
                passthrough[path] = PassthroughObservation("extra_body", option_value)

        if req.scheduling.stream and "stream_options" not in extra_body:
            body["stream_options"] = {"include_usage": True}
        if extra_body:
            body["extra_body"] = extra_body

        request_params = {
            key: value for key, value in body.items() if key not in {"model", "prompt"}
        }
        prepared = PreparedRequest(
            operation="completions.create",
            body=body,
            consumed_paths=frozenset(consumed),
            passthrough=passthrough,
            context=_CompletionContext(
                n=sampling.n,
                input_scoring=scoring.input_scoring,
                stream=req.scheduling.stream,
                request_params=request_params,
            ),
        )
        return prepared

    async def execute(self, prepared: PreparedRequest) -> Response:
        """Execute a prepared request using the borrowed OpenAI client."""

        if prepared.operation != "completions.create":
            raise DialectError(
                f"unexpected completions operation {prepared.operation!r}"
            )
        context = prepared.context
        if not isinstance(context, _CompletionContext):
            raise DialectError("prepared completions request has invalid context")

        raw = await self._client.completions.create(**dict(prepared.body))
        if context.stream:
            return await self._lift_stream(raw, context)
        return self._lift(raw, context)

    async def arun(self, req: Request) -> Response:
        """One-cycle direct compatibility path; Model uses the split contract."""

        plan = _LegacyPlan()
        validate_request_invariants(req)
        validate_runtime_binding_plan(plan, req)
        audit = RequestAudit(active_request_leaves(req))
        self.validate_request(req, audit, plan)
        audit.raise_rejections()
        prepared = self.prepare(req, audit)
        audit.finish(prepared)
        response = await self.execute(prepared)
        self.output_contract.validate(plan, req, response)
        return response

    def _lift(self, raw: object, context: _CompletionContext) -> Response:
        accumulator = _Accumulator(context, streaming=False)
        accumulator.capture_metadata(raw)
        accumulator.capture_choices(getattr(raw, "choices", ()))
        usage = _completions_usage_stats(getattr(raw, "usage", None))
        return accumulator.response(usage)

    async def _lift_stream(
        self, stream: object, context: _CompletionContext
    ) -> Response:
        if not isinstance(stream, AsyncIterable):
            raise OutputContractError(
                "streaming completions response is not asynchronously iterable"
            )
        accumulator = _Accumulator(context, streaming=True)
        usage: UsageStats | None = None
        async for chunk in stream:
            accumulator.capture_metadata(chunk)
            accumulator.capture_choices(getattr(chunk, "choices", ()))
            raw_usage = getattr(chunk, "usage", None)
            if raw_usage is not None:
                usage = _completions_usage_stats(raw_usage)
        return accumulator.response(usage)


class _Accumulator:
    def __init__(self, context: _CompletionContext, *, streaming: bool) -> None:
        self._context = context
        self._streaming = streaming
        self._texts = [""] * context.n
        self._finish_reasons = [""] * context.n
        self._seen_indices: set[int] = set()
        self._tokens: list[str] = []
        self._token_logprobs: list[float | None] = []
        self._top_logprobs: list[dict[str, float]] = []
        self._saw_top_logprobs = False
        self._saw_logprobs = False
        self._response_model: str | None = None
        self._system_fingerprint: str | None = None

    def capture_metadata(self, raw: object) -> None:
        response_model = _optional_response_string(getattr(raw, "model", None), "model")
        system_fingerprint = _optional_response_string(
            getattr(raw, "system_fingerprint", None), "system_fingerprint"
        )
        if self._response_model is None:
            self._response_model = response_model
        if self._system_fingerprint is None:
            self._system_fingerprint = system_fingerprint

    def capture_choices(self, raw_choices: object) -> None:
        for choice in _choice_sequence(raw_choices):
            index = _choice_index(choice, self._context.n)
            if not self._streaming and index in self._seen_indices:
                raise OutputContractError(
                    f"completion response duplicated choice index {index}"
                )
            self._seen_indices.add(index)

            text = getattr(choice, "text", None)
            if isinstance(text, str):
                self._texts[index] += text
            finish_reason = getattr(choice, "finish_reason", None)
            if isinstance(finish_reason, str):
                self._finish_reasons[index] = finish_reason

            if index != 0:
                continue
            logprobs = getattr(choice, "logprobs", None)
            if logprobs is None:
                continue
            self._saw_logprobs = True
            raw_tokens = _logprob_sequence(
                getattr(logprobs, "tokens", None),
                "completion logprobs.tokens",
            )
            raw_token_logprobs = _logprob_sequence(
                getattr(logprobs, "token_logprobs", None),
                "completion logprobs.token_logprobs",
            )
            if len(raw_tokens) != len(raw_token_logprobs):
                raise OutputContractError(
                    "completion token and token-logprob counts are inconsistent"
                )
            parsed_tokens: list[str] = []
            parsed_logprobs: list[float | None] = []
            for position, token in enumerate(raw_tokens):
                if not isinstance(token, str):
                    raise OutputContractError(
                        f"completion logprobs.tokens[{position}] must be a string"
                    )
                parsed_tokens.append(token)
            for position, logprob in enumerate(raw_token_logprobs):
                parsed_logprobs.append(
                    _sampled_logprob(
                        logprob,
                        f"completion logprobs.token_logprobs[{position}]",
                    )
                )
            parsed_top_logprobs = _completion_top_logprobs(
                getattr(logprobs, "top_logprobs", None)
            )
            if parsed_top_logprobs is not None:
                if len(parsed_top_logprobs) != len(parsed_tokens):
                    raise OutputContractError(
                        "completion top-logprob positions are inconsistent"
                    )
                self._saw_top_logprobs = True
                self._top_logprobs.extend(parsed_top_logprobs)
            self._tokens.extend(parsed_tokens)
            self._token_logprobs.extend(parsed_logprobs)

    def response(self, usage: UsageStats | None) -> Response:
        missing = sorted(set(range(self._context.n)) - self._seen_indices)
        if missing:
            mode = "stream" if self._streaming else "response"
            raise OutputContractError(
                f"completion {mode} omitted choice indexes {missing}"
            )

        input_scoring: InputScoringResult | None = None
        sampled: tuple[TokenLogprob, ...] | None = None
        top: tuple[tuple[TopKEntry, ...], ...] | None = None

        if self._context.input_scoring:
            if usage is None:
                raise OutputContractError(
                    "input scoring requires a usage prompt-token boundary"
                )
            if not self._saw_logprobs:
                raise OutputContractError(
                    "input scoring response omitted echoed token logprobs"
                )
            self._validate_input_scoring_boundary(usage)

        if self._saw_logprobs:
            if len(self._token_logprobs) != len(self._tokens):
                raise OutputContractError(
                    "completion token and token-logprob counts are inconsistent"
                )
            if self._saw_top_logprobs and len(self._top_logprobs) != len(self._tokens):
                raise OutputContractError(
                    "completion top-logprob positions are inconsistent"
                )
            all_tokens = tuple(
                TokenLogprob(token=token, logprob=logprob)
                for token, logprob in zip(
                    self._tokens, self._token_logprobs, strict=True
                )
            )
            all_top = (
                tuple(
                    tuple(
                        TopKEntry(token=token, logprob=logprob)
                        for token, logprob in position.items()
                    )
                    for position in self._top_logprobs
                )
                if self._saw_top_logprobs
                else None
            )
            if self._context.input_scoring:
                assert usage is not None
                boundary = usage.input_tokens
                input_scoring = InputScoringResult(all_tokens[:boundary])
                sampled = all_tokens[boundary:]
                top = all_top[boundary:] if all_top is not None else None
            else:
                sampled = all_tokens
                top = all_top

        return Response(
            texts=tuple(self._texts),
            finish_reasons=tuple(self._finish_reasons),
            logprobs=sampled,
            top_logprobs=top,
            input_scoring=input_scoring,
            usage=usage,
            request_params=self._context.request_params,
            response_model=self._response_model,
            system_fingerprint=self._system_fingerprint,
        )

    def _validate_input_scoring_boundary(self, usage: UsageStats) -> None:
        token_count = len(self._tokens)
        if len(self._token_logprobs) != token_count:
            raise OutputContractError(
                "input scoring token and token-logprob counts are inconsistent"
            )
        if usage.input_tokens <= 0:
            raise OutputContractError(
                "input scoring requires a positive prompt-token boundary"
            )
        if usage.input_tokens > token_count:
            raise OutputContractError(
                "usage prompt-token boundary exceeds echoed logprob positions"
            )
        observed_output = token_count - usage.input_tokens
        if usage.output_tokens != observed_output:
            raise OutputContractError(
                "usage completion-token count contradicts echoed logprob positions"
            )
        if usage.total_tokens != usage.input_tokens + usage.output_tokens:
            raise OutputContractError(
                "usage total-token count contradicts prompt and completion counts"
            )
        if self._top_logprobs and len(self._top_logprobs) != token_count:
            raise OutputContractError(
                "input scoring top-logprob positions are inconsistent"
            )
