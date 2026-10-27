"""SGLang native ``POST /generate`` dialect for the provider-neutral model IR.

SGLang's ``/v1/completions`` rejects ``echo=True`` with ``logprobs``, so input
scoring needs ``/generate``; this dialect serves generation and scoring there.
``input_token_logprobs`` lifts to ``Response.input_scoring`` and
``output_token_logprobs`` to ``Response.logprobs``; triples keep their token id.

The wire body is the retired ``sglang_legacy`` bypass's, byte for byte, except
that options it sent or dropped are now refused before I/O: ``prefill``
(SGLang's ``SamplingParams`` has no such argument), a request seed (``/generate``
has none) and unknown options.  No model id is reported: ``/generate`` returns
none.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import httpx

from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    Capability,
    CapabilityKey,
    DialectCapabilityBinding,
    DialectCapabilityDecision,
    Supported,
    Unsupported,
)
from sieval.core.models.connection_factory import AsyncHTTPJSONConnection
from sieval.core.models.dialect import (
    DialectError,
    Guarantee,
    OutputContract,
    OutputContractError,
    OutputRule,
    PreparedRequest,
    RequestAudit,
    RuntimePlanView,
    WireEvidence,
    WireObservation,
    active_request_leaves,
    validate_choice_logprobs,
    validate_input_scoring,
    validate_request_invariants,
    validate_runtime_binding_plan,
    validate_top_logprobs,
)
from sieval.core.models.ir import (
    ChoiceLogprobs,
    CompletionInput,
    InputScoringResult,
    Request,
    Response,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)
from sieval.core.types import JSONValue

from ._http_status import raise_for_provider_status

_OPERATION = "generate"

_PREFILL_UNSUPPORTED_REASON = (
    "SGLang /generate has no prefill: SamplingParams takes no prefill "
    "argument, so the server refuses it"
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
        "reasoning": Unsupported("/generate has no reasoning-control protocol"),
        "function_tools": Unsupported("/generate has no function-tool protocol"),
        "hosted_tools": Unsupported("/generate has no hosted-tool protocol"),
        "structured_output": Unsupported(
            "/generate structured decoding is not mapped to the IR"
        ),
        "stateful_session": Unsupported("/generate has no response-id continuation"),
        "opaque_continuation": Unsupported(
            "/generate has no opaque continuation channel"
        ),
        "multimodal_input": Unsupported("this dialect sends text prompts only"),
        "prefill": Unsupported(_PREFILL_UNSUPPORTED_REASON),
        "fim": Unsupported("/generate has no suffix input"),
    }
    if set(decisions) != set(CAPABILITY_KEYS):
        raise AssertionError("SGLang native capability row is incomplete")
    return MappingProxyType(decisions)


CAPABILITY_DECISIONS = _decisions()


def _validate_token_logprobs(value: object) -> None:
    if not isinstance(value, tuple) or not all(
        isinstance(item, TokenLogprob) for item in value
    ):
        raise OutputContractError("logprobs channel has invalid shape")


OUTPUT_CONTRACT = OutputContract(
    {
        "reasoning": OutputRule(Guarantee.NEVER),
        "tool_calls": OutputRule(Guarantee.NEVER),
        "server_tool_uses": OutputRule(Guarantee.NEVER),
        "structured_output": OutputRule(Guarantee.NEVER),
        "logprobs": OutputRule(Guarantee.PRESENT_OR_ERROR, _validate_token_logprobs),
        "top_logprobs": OutputRule(Guarantee.PRESENT_OR_ERROR, validate_top_logprobs),
        "choice_logprobs": OutputRule(
            Guarantee.PRESENT_OR_ERROR, validate_choice_logprobs
        ),
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
        "sampling.temperature",
        "sampling.top_p",
        "sampling.top_k",
        "sampling.max_tokens",
        "sampling.stop",
        "sampling.frequency_penalty",
        "sampling.presence_penalty",
        "sampling.n",
        "scoring.input_scoring",
        "scoring.sampled_logprobs",
        "scoring.top_logprobs",
    }
)

_NOOP_PATHS: Mapping[str, str] = MappingProxyType(
    {
        # The builder defaults stream=True; one POST returns the same content.
        "scheduling.stream": "/generate is sent as one non-streaming POST",
    }
)

_SEED_UNSUPPORTED_REASON = (
    "SGLang /generate has no per-request seed; pin it on the engine with "
    "--random-seed (a managed deterministic engine already does)"
)

#: SGLang sampling parameters with no first-class request leaf, forwarded
#: into ``sampling_params`` as dialect options.
_SAMPLING_OPTION_KEYS = frozenset({"min_p", "repetition_penalty"})

#: Option keys that spell a first-class request leaf, and that leaf.
_IR_OWNED_OPTION_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "max_tokens": "sampling.max_tokens",
        "max_new_tokens": "sampling.max_tokens",
        "temperature": "sampling.temperature",
        "top_p": "sampling.top_p",
        "top_k": "sampling.top_k",
        "stop": "sampling.stop",
        "frequency_penalty": "sampling.frequency_penalty",
        "presence_penalty": "sampling.presence_penalty",
        "n": "sampling.n",
        "stream": "scheduling.stream",
    }
)


@dataclass(frozen=True)
class _GenerateContext:
    n: int
    input_scoring: bool
    sampled_logprobs: bool
    top_logprobs: int


@dataclass(frozen=True)
class _MaxNewTokensVerifier:
    """Prove ``max_new_tokens`` is the request's value or its documented clamp."""

    expected: int

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        sampling = body.get("sampling_params")
        if not isinstance(sampling, Mapping):
            return "body.sampling_params is missing"
        actual = cast(Mapping[str, JSONValue], sampling).get("max_new_tokens")
        if isinstance(actual, bool) or actual != self.expected:
            return "body.sampling_params.max_new_tokens does not match the request"
        return None


@dataclass(frozen=True)
class _LogprobRequestVerifier:
    """Prove the logprob switches and start position match the scoring leaves."""

    input_scoring: bool
    top_logprobs: int

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        expected: dict[str, JSONValue] = {
            "return_logprob": True,
            # 0 scores every input position; -1 only the generated tokens.
            "logprob_start_len": 0 if self.input_scoring else -1,
            "top_logprobs_num": self.top_logprobs,
            "return_text_in_logprobs": True,
        }
        for key, value in expected.items():
            actual = body.get(key)
            if type(actual) is not type(value) or actual != value:
                return f"body.{key} does not match the request's scoring leaves"
        return None


@dataclass(frozen=True)
class _LegacyPlan:
    """Minimal plan used only by the direct ``arun`` call path."""

    dialect_id: str = "sglang_native"
    available_capabilities: frozenset[str] = frozenset(
        key
        for key, decision in CAPABILITY_DECISIONS.items()
        if isinstance(decision, Supported)
    )
    capability_minimums: Mapping[str, Mapping[str, JSONValue]] = field(
        default_factory=dict
    )
    required_output_channels: frozenset[str] = frozenset()


def generate_url(base_url: httpx.URL) -> httpx.URL:
    """The native ``/generate`` URL beside an OpenAI-style ``/v1`` base.

    A deployment's ``api_base`` is the OpenAI base (``http://host:30000/v1``);
    ``/generate`` sits at the server root, so a trailing ``/v1`` is dropped.
    """

    path = base_url.path.rstrip("/").removesuffix("/v1").rstrip("/")
    return base_url.copy_with(path=f"{path}/{_OPERATION}", query=None)


def _normalize_token_text(text: object) -> str:
    """Map GPT-2 byte-level BPE markers (``Ġ``, ``Ċ``) back to whitespace.

    Byte-level tokenizers (Qwen) surface them even with
    ``return_text_in_logprobs``; option scoring matches ``" A"``, so ``"ĠA"``
    would never match.  ``None`` text (``--skip-tokenizer-init``) fails loud.
    """

    if text is None:
        raise OutputContractError(
            "sglang returned a logprob entry with no token text; option/letter "
            "scoring needs detokenized text. Do not launch sglang with "
            "--skip-tokenizer-init (it ignores return_text_in_logprobs)."
        )
    if not isinstance(text, str):
        raise OutputContractError("sglang logprob token text must be a string")
    return text.replace("Ġ", " ").replace("Ċ", "\n")


def _triple(raw: object, path: str) -> tuple[float | None, int, str]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes) or len(raw) != 3:
        raise OutputContractError(
            f"{path} must be a [logprob, token_id, token_text] triple"
        )
    logprob, token_id, token_text = raw
    if logprob is not None:
        if isinstance(logprob, bool) or not isinstance(logprob, int | float):
            raise OutputContractError(f"{path} logprob must be numeric or None")
        logprob = float(logprob)
        if math.isnan(logprob) or logprob == math.inf:
            raise OutputContractError(f"{path} logprob must not be NaN or +inf")
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise OutputContractError(f"{path} token id must be an integer")
    return logprob, token_id, _normalize_token_text(token_text)


def _sequence(raw: object, path: str) -> Sequence[object]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise OutputContractError(f"{path} must be a list")
    return raw


def _tokens(raw: object, path: str) -> tuple[TokenLogprob, ...]:
    tokens: list[TokenLogprob] = []
    for position, item in enumerate(_sequence(raw, path)):
        logprob, token_id, text = _triple(item, f"{path}[{position}]")
        tokens.append(TokenLogprob(token=text, logprob=logprob, token_id=token_id))
    return tuple(tokens)


def _top_k(raw: object, path: str) -> tuple[tuple[TopKEntry, ...], ...]:
    """Lift per-position top-k triples; a ``None`` position (the first input
    token, which has no context) becomes ``()``."""

    positions: list[tuple[TopKEntry, ...]] = []
    for position, entries in enumerate(_sequence(raw, path)):
        if entries is None:
            positions.append(())
            continue
        alternatives: list[TopKEntry] = []
        for rank, item in enumerate(_sequence(entries, f"{path}[{position}]")):
            logprob, token_id, text = _triple(item, f"{path}[{position}][{rank}]")
            if logprob is None:
                raise OutputContractError(
                    f"{path}[{position}][{rank}] top-k logprob must be numeric"
                )
            alternatives.append(
                TopKEntry(token=text, logprob=logprob, token_id=token_id)
            )
        positions.append(tuple(alternatives))
    return tuple(positions)


def _count(meta: Mapping[str, object], key: str) -> int | None:
    value = meta.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OutputContractError(f"sglang meta_info.{key} must be a non-negative int")
    return value


def _usage(metas: Sequence[Mapping[str, object]]) -> UsageStats | None:
    """Usage from ``meta_info``: completions sum over the ``n`` samples, while
    ``prompt_tokens`` and ``cached_tokens`` describe the shared prompt and are
    read once.  A malformed count fails the sample (the cache guard reads it).
    """

    input_tokens = _count(metas[0], "prompt_tokens")
    if input_tokens is None:
        return None
    output_tokens = sum(_count(meta, "completion_tokens") or 0 for meta in metas)
    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_tokens=_count(metas[0], "cached_tokens"),
    )


def _finish_reason(meta: Mapping[str, object]) -> str:
    """A flat finish reason; SGLang sends ``{"type": "length", ...}``."""

    reason = meta.get("finish_reason")
    if isinstance(reason, Mapping):
        kind = cast(Mapping[str, object], reason).get("type")
        return kind if isinstance(kind, str) else ""
    return reason if isinstance(reason, str) else ""


def _guard_radix_cache(meta: Mapping[str, object], positions: int) -> None:
    """Refuse input scores a radix-cache hit truncated.

    On a hit SGLang silently returns only ``prompt_tokens - cached_tokens``
    input scores, so any cache touch, or a reply without ``prompt_tokens``, is
    refused.
    """

    prompt_tokens = _count(meta, "prompt_tokens")
    cached_tokens = _count(meta, "cached_tokens") or 0
    if prompt_tokens is None:
        raise OutputContractError(
            "sglang response omitted prompt_tokens, so echoed-input "
            "completeness cannot be verified; refusing to score silently. "
            "Launch sglang with --disable-radix-cache."
        )
    if cached_tokens or positions != prompt_tokens:
        raise OutputContractError(
            "sglang returned partial echoed-input logprobs "
            f"({positions} of {prompt_tokens} prompt tokens, "
            f"cached_tokens={cached_tokens}): its radix prefix cache does "
            "not recompute logprobs for cached positions, so echo-based "
            "scoring would be silently wrong. Launch sglang with "
            "--disable-radix-cache."
        )


def _sample(raw: object, index: int) -> tuple[str, Mapping[str, object]]:
    if not isinstance(raw, Mapping):
        raise OutputContractError(f"sglang /generate sample {index} must be an object")
    sample = cast(Mapping[str, object], raw)
    text = sample.get("text")
    meta = sample.get("meta_info")
    if not isinstance(text, str):
        raise OutputContractError(f"sglang /generate sample {index} has no text")
    if not isinstance(meta, Mapping):
        raise OutputContractError(f"sglang /generate sample {index} has no meta_info")
    return text, cast(Mapping[str, object], meta)


def _choice_logprobs(meta: Mapping[str, object], top_k: int) -> ChoiceLogprobs:
    raw_tokens = meta.get("output_token_logprobs")
    if raw_tokens is None:
        raise OutputContractError(
            "sglang /generate returned no output_token_logprobs for a sample"
        )
    tokens = _tokens(raw_tokens, "output_token_logprobs")
    if top_k == 0:
        return ChoiceLogprobs(tokens=tokens)
    top = _top_k(meta.get("output_top_logprobs") or (), "output_top_logprobs")
    if len(top) != len(tokens):
        raise OutputContractError(
            f"sglang /generate returned top logprobs for {len(top)} of "
            f"{len(tokens)} sampled tokens"
        )
    return ChoiceLogprobs(tokens=tokens, top_logprobs=top)


def _input_scoring(meta: Mapping[str, object], top_k: int) -> InputScoringResult:
    raw_tokens = meta.get("input_token_logprobs")
    if raw_tokens is None:
        raise OutputContractError("sglang /generate returned no input_token_logprobs")
    tokens = _tokens(raw_tokens, "input_token_logprobs")
    if not tokens:
        raise OutputContractError("sglang /generate scored no input tokens")
    _guard_radix_cache(meta, len(tokens))
    if top_k == 0:
        return InputScoringResult(token_logprobs=tokens)
    top = _top_k(meta.get("input_top_logprobs") or (), "input_top_logprobs")
    if len(top) != len(tokens):
        raise OutputContractError(
            f"sglang returned input top-k logprobs for {len(top)} of "
            f"{len(tokens)} echoed input tokens; refusing to misalign them"
        )
    return InputScoringResult(token_logprobs=tokens, top_logprobs=top)


class SglangNativeDialect:
    """Executable dialect for SGLang's native ``/generate`` endpoint."""

    dialect_id = "sglang_native"
    connection_family = "async_http_json"
    capability_decisions = CAPABILITY_DECISIONS
    output_contract = OUTPUT_CONTRACT
    CAPABILITIES = frozenset(
        {
            Capability.Completion,
            Capability.InputScoring,
            Capability.SampledLogprobs,
            Capability.SampledLogprobsWithTokenIds,
            Capability.TopKLogprobs,
        }
    )

    def __init__(self, connection: object, requested_model_id: str) -> None:
        if not isinstance(connection, AsyncHTTPJSONConnection):
            raise TypeError("sglang_native requires AsyncHTTPJSONConnection")
        if not requested_model_id:
            raise ValueError("requested_model_id must not be empty")
        self._connection = connection
        self._requested_model_id = requested_model_id
        self._url = generate_url(connection.client.base_url)

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Deprecated enum view of the keyed capability row."""

        return self.CAPABILITIES

    def validate_request(
        self, req: Request, audit: RequestAudit, plan: RuntimePlanView
    ) -> None:
        """Account for every active leaf before any client operation."""

        audit.require_matches_request(req)
        del plan
        for path in audit.active_paths:
            if path in _CONSUMED_PATHS or path in _NOOP_PATHS:
                continue
            if path == "dialect_options":
                continue
            if path == "sampling.seed":
                audit.rejected(path, _SEED_UNSUPPORTED_REASON)
                continue
            if not path.startswith("dialect_options."):
                audit.rejected(
                    path, "the SGLang native dialect does not support this request leaf"
                )
                continue
            key = path.removeprefix("dialect_options.")
            options = req.dialect_options
            assert options is not None
            if key in {"prefill", "prefix"}:
                audit.rejected(path, _PREFILL_UNSUPPORTED_REASON)
            elif key in _IR_OWNED_OPTION_KEYS:
                audit.rejected(
                    path, f"use the request leaf {_IR_OWNED_OPTION_KEYS[key]}"
                )
            elif key not in _SAMPLING_OPTION_KEYS:
                audit.rejected(
                    path,
                    "unknown SGLang option; forwarded sampling options are "
                    + ", ".join(sorted(_SAMPLING_OPTION_KEYS)),
                )
            elif options.values[key] is None:
                audit.rejected(path, "a null option has no wire value")

    def prepare(self, req: Request, audit: RequestAudit) -> PreparedRequest:
        """Lower one validated request and register exact audit evidence."""

        audit.require_matches_request(req)
        audit.raise_rejections()
        if not isinstance(req.input, CompletionInput):
            raise DialectError("sglang_native requires CompletionInput")

        def consume(path: str, *wire: WireEvidence) -> None:
            audit.consumed(path, *wire)

        sampling: dict[str, JSONValue] = {}
        body: dict[str, JSONValue] = {
            "text": req.input.text,
            "sampling_params": sampling,
        }
        consume("input.completion", WireObservation(("text",)))

        params = req.sampling
        scoring = req.scoring
        scores = scoring.input_scoring or scoring.sampled_logprobs
        max_tokens = params.max_tokens
        # A scoring request decodes at least one token (and input scoring one
        # rather than the default 128).  Set first to keep the retired
        # transport's key order, and so the recorded request_params bytes.
        if max_tokens is None and scoring.input_scoring:
            sampling["max_new_tokens"] = 1
        elif max_tokens is not None:
            sampling["max_new_tokens"] = 1 if scores and max_tokens == 0 else max_tokens
        if max_tokens is not None:
            consume(
                "sampling.max_tokens",
                _MaxNewTokensVerifier(cast(int, sampling["max_new_tokens"])),
            )

        simple: dict[str, JSONValue] = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "stop": None if params.stop is None else list(params.stop),
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
        }
        for name, value in simple.items():
            if value is not None:
                sampling[name] = value
                consume(f"sampling.{name}", WireObservation(("sampling_params", name)))
        if params.n != 1:
            sampling["n"] = params.n
            consume("sampling.n", WireObservation(("sampling_params", "n")))

        if scores:
            body["return_logprob"] = True
            body["logprob_start_len"] = 0 if scoring.input_scoring else -1
            body["top_logprobs_num"] = scoring.top_logprobs
            body["return_text_in_logprobs"] = True
            verifier = _LogprobRequestVerifier(
                scoring.input_scoring, scoring.top_logprobs
            )
            if scoring.input_scoring:
                consume("scoring.input_scoring", verifier)
            if scoring.sampled_logprobs:
                consume("scoring.sampled_logprobs", verifier)
            if scoring.top_logprobs > 0:
                consume("scoring.top_logprobs", WireObservation(("top_logprobs_num",)))

        for path, reason in _NOOP_PATHS.items():
            if path in audit.active_paths:
                audit.noop(path, reason)

        options = req.dialect_options
        if options is not None:
            if not options.values:
                audit.noop(
                    "dialect_options", "an empty options mapping has no wire semantics"
                )
            for key, value in options.values.items():
                sampling[key] = value
                audit.passthrough(f"dialect_options.{key}", "sampling_params")

        return PreparedRequest(
            operation=_OPERATION,
            body=body,
            context=_GenerateContext(
                n=params.n,
                input_scoring=scoring.input_scoring,
                sampled_logprobs=scoring.sampled_logprobs,
                top_logprobs=scoring.top_logprobs,
            ),
        )

    async def execute(self, prepared: PreparedRequest) -> Response:
        """POST the prepared body to ``/generate`` and lift the reply."""

        if prepared.operation != _OPERATION:
            raise DialectError(f"unexpected SGLang operation {prepared.operation!r}")
        context = prepared.context
        if not isinstance(context, _GenerateContext):
            raise DialectError("prepared SGLang request has invalid context")
        body = prepared.thaw_body()
        headers = {"Content-Type": "application/json"}
        credential = self._connection.credential
        if credential is not None:
            headers["Authorization"] = f"Bearer {credential}"
        response = await self._connection.client.post(
            self._url, headers=headers, json=body
        )
        raise_for_provider_status(response, "SGLang")
        try:
            payload = json.loads(response.text)
        except ValueError as exc:
            raise OutputContractError("sglang /generate returned invalid JSON") from exc
        request_params = {key: value for key, value in body.items() if key != "text"}
        return self._lift(payload, context, request_params)

    def _lift(
        self,
        payload: object,
        context: _GenerateContext,
        request_params: Mapping[str, JSONValue],
    ) -> Response:
        # n > 1 replies with one object per sample; n == 1 with a bare object.
        raw_samples = payload if isinstance(payload, list) else [payload]
        if len(raw_samples) != context.n:
            raise OutputContractError(
                f"sglang /generate returned {len(raw_samples)} samples for "
                f"n={context.n}"
            )
        samples = [_sample(raw, index) for index, raw in enumerate(raw_samples)]
        metas = [meta for _, meta in samples]

        input_scoring: InputScoringResult | None = None
        logprobs: tuple[TokenLogprob, ...] | None = None
        top_logprobs: tuple[tuple[TopKEntry, ...], ...] | None = None
        choice_logprobs: tuple[ChoiceLogprobs, ...] | None = None
        if context.input_scoring:
            input_scoring = _input_scoring(metas[0], context.top_logprobs)
        if context.sampled_logprobs and context.n > 1:
            choice_logprobs = tuple(
                _choice_logprobs(meta, context.top_logprobs) for meta in metas
            )
        elif context.sampled_logprobs:
            sampled = _choice_logprobs(metas[0], context.top_logprobs)
            logprobs = sampled.tokens
            top_logprobs = sampled.top_logprobs
        elif context.input_scoring:
            # Generated-token scores are not requested; kept when sent.
            raw_output = metas[0].get("output_token_logprobs")
            logprobs = (
                ()
                if raw_output is None
                else _tokens(raw_output, "output_token_logprobs")
            )

        return Response(
            texts=tuple(text for text, _ in samples),
            finish_reasons=tuple(_finish_reason(meta) for meta in metas),
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            choice_logprobs=choice_logprobs,
            input_scoring=input_scoring,
            usage=_usage(metas),
            request_params=request_params,
            response_model=None,
        )

    async def arun(self, req: Request) -> Response:
        """Direct compatibility path; ``Model`` uses the split contract."""

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
