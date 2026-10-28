"""vLLM native completions dialect for the provider-neutral model IR.

vLLM's native surface is the fields its ``/v1/completions`` accepts beyond
OpenAI's, posted over raw HTTP (the OpenAI SDK would drop them from the typed
reply):

- ``prompt_logprobs`` scores the input as its own channel, without ``echo``;
- ``return_token_ids`` gives input and sampled tokens their ids;
- ``structured_outputs`` constrains decoding to a JSON schema or any JSON.

vLLM ignores an unknown field silently (measured on 0.26.0), so options
outside a probed set are refused rather than forwarded.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import httpx

from sieval.core.models._shared import thaw_json_mapping
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
    validate_structured_output,
    validate_top_logprobs,
)
from sieval.core.models.ir import (
    ChoiceLogprobs,
    CompletionInput,
    InputScoringResult,
    Request,
    Response,
    StructuredOutput,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)
from sieval.core.types import JSONValue

from ._http_status import raise_for_provider_status

_OPERATION = "completions"

_PARSER_UNSUPPORTED = (
    "/v1/completions takes a raw prompt; vLLM's reasoning and tool parsers "
    "act on chat requests only"
)

_PREFILL_UNSUPPORTED = (
    "assistant prefill is a chat input operation, not a completion one"
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
        "structured_output": Supported(
            DialectCapabilityBinding(
                "structured_output",
                request_leaves=(
                    "structured_output.format",
                    "structured_output.schema",
                    "structured_output.name",
                    "structured_output.strict",
                ),
                response_channels=("structured_output",),
            )
        ),
        "reasoning": Unsupported(_PARSER_UNSUPPORTED),
        "function_tools": Unsupported(_PARSER_UNSUPPORTED),
        "hosted_tools": Unsupported("vLLM hosts no server-side tools"),
        "stateful_session": Unsupported("/v1/completions has no response-id state"),
        "opaque_continuation": Unsupported(
            "/v1/completions has no opaque continuation channel"
        ),
        "multimodal_input": Unsupported("this dialect sends text prompts only"),
        "prefill": Unsupported(_PREFILL_UNSUPPORTED),
        "fim": Unsupported("vLLM refuses suffix: 'suffix is not currently supported'"),
    }
    if set(decisions) != set(CAPABILITY_KEYS):
        raise AssertionError("vLLM native capability row is incomplete")
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
        "structured_output": OutputRule(
            Guarantee.PRESENT_OR_ERROR, validate_structured_output
        ),
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
        "sampling.seed",
        "sampling.frequency_penalty",
        "sampling.presence_penalty",
        "sampling.n",
        "scoring.input_scoring",
        "scoring.sampled_logprobs",
        "scoring.top_logprobs",
        "structured_output.format",
        "structured_output.schema",
        "structured_output.name",
        "structured_output.strict",
    }
)

_NOOP_PATHS: Mapping[str, str] = MappingProxyType(
    {
        # Streamed choices carry no prompt_logprobs (measured on 0.26.0).
        "scheduling.stream": "the request is sent as one non-streaming POST",
    }
)

_SCHEMA_NAME_NOOP = (
    "structured_outputs takes no schema name; the name only labels the schema"
)
_STRICT_NOOP = "vLLM's grammar-constrained decoding is always strict"

#: vLLM sampling parameters with no request leaf, forwarded as top-level
#: fields; each was probed live.
_SAMPLING_OPTION_KEYS = frozenset({"min_p", "repetition_penalty", "stop_token_ids"})

#: Option keys that spell a first-class request leaf, or a field this dialect
#: derives from one, and that leaf.
_IR_OWNED_OPTION_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "prompt": "input.completion",
        "suffix": "input.completion.suffix",
        "temperature": "sampling.temperature",
        "top_p": "sampling.top_p",
        "top_k": "sampling.top_k",
        "max_tokens": "sampling.max_tokens",
        "stop": "sampling.stop",
        "seed": "sampling.seed",
        "frequency_penalty": "sampling.frequency_penalty",
        "presence_penalty": "sampling.presence_penalty",
        "n": "sampling.n",
        "echo": "scoring.input_scoring",
        "prompt_logprobs": "scoring.input_scoring",
        "logprobs": "scoring.sampled_logprobs",
        "return_token_ids": "scoring.sampled_logprobs",
        "structured_outputs": "structured_output.format",
        "response_format": "structured_output.format",
        "stream": "scheduling.stream",
        "stream_options": "scheduling.stream",
    }
)


@dataclass(frozen=True)
class _CompletionContext:
    prompt: str
    n: int
    input_scoring: bool
    sampled_logprobs: bool
    top_logprobs: int
    structured_output: bool


@dataclass(frozen=True)
class _StructuredOutputsVerifier:
    """Prove ``structured_outputs`` spells the request's format and schema."""

    expected: Mapping[str, JSONValue]

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        actual = body.get("structured_outputs")
        if not isinstance(actual, Mapping) or thaw_json_mapping(
            cast(Mapping[str, JSONValue], actual), "body.structured_outputs"
        ) != thaw_json_mapping(self.expected, "structured_outputs"):
            return "body.structured_outputs does not match the request's format"
        return None


@dataclass(frozen=True)
class _ScoringVerifier:
    """Prove a scoring field carries the request's breadth, with token ids.

    The leaf is a flag; its wire field (``prompt_logprobs`` or ``logprobs``)
    holds the top-k breadth, so a ``WireObservation`` cannot compare them.
    """

    wire_field: str
    breadth: int

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        actual = body.get(self.wire_field)
        if type(actual) is not int or actual != self.breadth:
            return f"body.{self.wire_field} does not match the request's top-k breadth"
        if body.get("return_token_ids") is not True:
            return "body.return_token_ids must be true for a scoring request"
        return None


@dataclass(frozen=True)
class _LegacyPlan:
    """Minimal plan used only by the direct ``arun`` call path."""

    dialect_id: str = "vllm_native"
    available_capabilities: frozenset[str] = frozenset(
        key
        for key, decision in CAPABILITY_DECISIONS.items()
        if isinstance(decision, Supported)
    )
    capability_minimums: Mapping[str, Mapping[str, JSONValue]] = field(
        default_factory=dict
    )
    required_output_channels: frozenset[str] = frozenset()


def completions_url(base_url: httpx.URL) -> httpx.URL:
    """``/completions`` under the OpenAI-style base (``http://host:8000/v1``)."""

    path = base_url.path.rstrip("/")
    return base_url.copy_with(path=f"{path}/{_OPERATION}", query=None)


def _structured_outputs(req: Request) -> dict[str, JSONValue] | None:
    params = req.structured_output
    if params.format is None:
        return None
    if params.format == "json_object":
        return {"json_object": True}
    assert params.schema is not None  # the IR requires it for json_schema
    return {"json": thaw_json_mapping(params.schema, "structured_output.schema")}


def _sequence(raw: object, path: str) -> Sequence[object]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise OutputContractError(f"{path} must be a list")
    return raw


def _mapping(raw: object, path: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise OutputContractError(f"{path} must be an object")
    return cast(Mapping[str, object], raw)


def _logprob(raw: object, path: str) -> float:
    """A finite or ``-inf`` logprob; vLLM sends ``-inf`` for a masked token."""

    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise OutputContractError(f"{path} must be numeric")
    value = float(raw)
    if math.isnan(value) or value == math.inf:
        raise OutputContractError(f"{path} must not be NaN or +inf")
    return value


def _token_id(raw: object, path: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise OutputContractError(f"{path} must be a non-negative integer")
    return raw


def _text(raw: object, path: str) -> str:
    if not isinstance(raw, str):
        raise OutputContractError(
            f"{path} must be a string; scoring needs detokenized text, so do "
            "not launch vLLM with --skip-tokenizer-init"
        )
    return raw


def _token_ids(raw: object, path: str) -> tuple[int, ...]:
    if raw is None:
        raise OutputContractError(
            f"vLLM returned no {path}; the scoring lift needs return_token_ids, "
            "which this server ignored (it predates the field)"
        )
    items = _sequence(raw, path)
    return tuple(
        _token_id(item, f"{path}[{index}]") for index, item in enumerate(items)
    )


def _prompt_position(
    raw: object, token_id: int, top_k: int, path: str
) -> tuple[TokenLogprob, tuple[TopKEntry, ...]]:
    """Lift one ``{token_id: {logprob, rank, decoded_token}}`` position.

    The scored token is the entry keyed by the prompt's own id; ``top_k + 1``
    entries list it beside the top-k, even when a tie gives it a top-k rank.
    """

    entries = _mapping(raw, path)
    if len(entries) not in {top_k, top_k + 1}:
        raise OutputContractError(
            f"{path} lists {len(entries)} entries for a top-{top_k} position"
        )
    beside = len(entries) == top_k + 1
    scored: TokenLogprob | None = None
    ranked: list[tuple[int, TopKEntry]] = []
    for key, value in entries.items():
        entry_path = f"{path}[{key!r}]"
        try:
            entry_id = int(key)
        except ValueError:
            raise OutputContractError(f"{entry_path} key must be a token id") from None
        entry = _mapping(value, entry_path)
        logprob = _logprob(entry.get("logprob"), f"{entry_path}.logprob")
        text = _text(entry.get("decoded_token"), f"{entry_path}.decoded_token")
        rank = entry.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise OutputContractError(f"{entry_path}.rank must be a positive integer")
        if entry_id == token_id:
            scored = TokenLogprob(token=text, logprob=logprob, token_id=entry_id)
        if not (beside and entry_id == token_id):
            ranked.append(
                (rank, TopKEntry(token=text, logprob=logprob, token_id=entry_id))
            )
    if scored is None:
        raise OutputContractError(
            f"{path} does not score the prompt's own token id {token_id}"
        )
    ranked.sort(key=lambda item: item[0])
    return scored, tuple(entry for _, entry in ranked)


def _first_token_text(prompt: str, scored: Sequence[TokenLogprob]) -> str:
    """The unscored first position's text: the prompt prefix the rest leaves.

    vLLM reports ``None`` for position 0, text included; the rest decode
    incrementally to the prompt's suffix (measured on CJK, byte-fallback
    emoji, leading spaces and newlines).  Empty for a BOS outside the prompt,
    or when the decoded tokens do not spell it.
    """

    rest = "".join(item.token for item in scored)
    if not prompt.endswith(rest):
        return ""
    return prompt[: len(prompt) - len(rest)]


def _input_scoring(
    choice: Mapping[str, object],
    prompt: str,
    prompt_tokens: int | None,
    top_k: int,
) -> InputScoringResult:
    raw_positions = choice.get("prompt_logprobs")
    if raw_positions is None:
        raise OutputContractError("vLLM returned no prompt_logprobs")
    positions = _sequence(raw_positions, "prompt_logprobs")
    token_ids = _token_ids(choice.get("prompt_token_ids"), "prompt_token_ids")
    if not positions:
        raise OutputContractError("vLLM scored no prompt tokens")
    # One position per prompt token, by both the ids and the usage count.
    if len(positions) != len(token_ids) or (
        prompt_tokens is not None and len(positions) != prompt_tokens
    ):
        raise OutputContractError(
            f"vLLM returned {len(positions)} prompt_logprobs positions for "
            f"{len(token_ids)} prompt token ids and usage.prompt_tokens="
            f"{prompt_tokens}; refusing to score a partial prompt"
        )
    if positions[0] is not None:
        raise OutputContractError("vLLM scored the first prompt position")
    scored: list[TokenLogprob] = []
    top: list[tuple[TopKEntry, ...]] = [()]
    for index in range(1, len(positions)):
        token, alternatives = _prompt_position(
            positions[index], token_ids[index], top_k, f"prompt_logprobs[{index}]"
        )
        scored.append(token)
        top.append(alternatives)
    first = TokenLogprob(
        token=_first_token_text(prompt, scored), logprob=None, token_id=token_ids[0]
    )
    tokens = (first, *scored)
    if top_k == 0:
        return InputScoringResult(token_logprobs=tokens)
    return InputScoringResult(token_logprobs=tokens, top_logprobs=tuple(top))


def _sampled_top(
    raw: object, sampled: Sequence[TokenLogprob], top_k: int
) -> tuple[tuple[TopKEntry, ...], ...]:
    """Per-position ``{text: logprob}`` maps, trimmed to the top ``top_k``.

    The map is keyed by decoded text: ``top_k + 1`` entries list the sampled
    token beside the top-k, and it is dropped.  Tokens that decode alike share
    a key; a map that is then short, or holds another value for the sampled
    token, is refused.  A collision that still leaves ``top_k`` entries is
    indistinguishable, since the map carries no rank.
    """

    positions = _sequence(raw, "logprobs.top_logprobs")
    if len(positions) != len(sampled):
        raise OutputContractError(
            f"vLLM returned top logprobs for {len(positions)} of {len(sampled)} "
            "sampled tokens"
        )
    lifted: list[tuple[TopKEntry, ...]] = []
    for index, (position, token) in enumerate(zip(positions, sampled, strict=True)):
        path = f"logprobs.top_logprobs[{index}]"
        entries = {
            text: _logprob(value, f"{path}[{text!r}]")
            for text, value in _mapping(position, path).items()
        }
        if (
            len(entries) not in {top_k, top_k + 1}
            or entries.get(token.token) != token.logprob
        ):
            raise OutputContractError(
                f"{path} cannot be split into the sampled token and its top-{top_k}: "
                "two tokens decode to the same text"
            )
        if len(entries) == top_k + 1:
            del entries[token.token]
        ranked = sorted(entries.items(), key=lambda item: -item[1])
        lifted.append(tuple(TopKEntry(token=text, logprob=lp) for text, lp in ranked))
    return tuple(lifted)


def _choice_logprobs(choice: Mapping[str, object], top_k: int) -> ChoiceLogprobs:
    raw = choice.get("logprobs")
    if raw is None:
        raise OutputContractError("vLLM returned no logprobs for a sampled choice")
    logprobs = _mapping(raw, "logprobs")
    texts = _sequence(logprobs.get("tokens"), "logprobs.tokens")
    values = _sequence(logprobs.get("token_logprobs"), "logprobs.token_logprobs")
    ids = _token_ids(choice.get("token_ids"), "token_ids")
    if not len(texts) == len(values) == len(ids):
        raise OutputContractError(
            f"vLLM returned {len(texts)} sampled tokens, {len(values)} logprobs "
            f"and {len(ids)} token ids"
        )
    tokens = tuple(
        TokenLogprob(
            token=_text(text, f"logprobs.tokens[{index}]"),
            logprob=_logprob(value, f"logprobs.token_logprobs[{index}]"),
            token_id=token_id,
        )
        for index, (text, value, token_id) in enumerate(
            zip(texts, values, ids, strict=True)
        )
    )
    if top_k == 0:
        return ChoiceLogprobs(tokens=tokens)
    return ChoiceLogprobs(
        tokens=tokens,
        top_logprobs=_sampled_top(logprobs.get("top_logprobs"), tokens, top_k),
    )


def _count(usage: Mapping[str, object], key: str) -> int:
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OutputContractError(f"vLLM usage.{key} must be a non-negative integer")
    return value


def _usage(raw: object) -> UsageStats | None:
    """Usage from the two counts; ``cached_tokens`` only when vLLM reports it
    (``--enable-prompt-tokens-details``)."""

    if raw is None:
        return None
    usage = _mapping(raw, "usage")
    input_tokens = _count(usage, "prompt_tokens")
    output_tokens = _count(usage, "completion_tokens")
    cached: int | None = None
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        value = cast(Mapping[str, object], details).get("cached_tokens")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            cached = value
    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_tokens=cached,
    )


def _choices(payload: Mapping[str, object], n: int) -> list[Mapping[str, object]]:
    by_index: dict[int, Mapping[str, object]] = {}
    for position, raw in enumerate(_sequence(payload.get("choices"), "choices")):
        choice = _mapping(raw, f"choices[{position}]")
        index = choice.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < n:
            raise OutputContractError(
                f"vLLM choice index {index!r} is outside the requested range [0, {n})"
            )
        if index in by_index:
            raise OutputContractError(f"vLLM duplicated choice index {index}")
        if not isinstance(choice.get("text"), str):
            raise OutputContractError(f"vLLM choice {index} has no text")
        by_index[index] = choice
    missing = sorted(set(range(n)) - set(by_index))
    if missing:
        raise OutputContractError(f"vLLM omitted choice indexes {missing}")
    return [by_index[index] for index in range(n)]


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise OutputContractError(f"vLLM {key} must be a string")
    return value


class VllmNativeDialect:
    """Executable dialect for vLLM's ``/v1/completions`` with its extensions."""

    dialect_id = "vllm_native"
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
            Capability.StructuredOutput,
        }
    )

    def __init__(self, connection: object, requested_model_id: str) -> None:
        if not isinstance(connection, AsyncHTTPJSONConnection):
            raise TypeError("vllm_native requires AsyncHTTPJSONConnection")
        if not requested_model_id:
            raise ValueError("requested_model_id must not be empty")
        self._connection = connection
        self._requested_model_id = requested_model_id
        self._url = completions_url(connection.client.base_url)

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
        structured = req.structured_output
        for path in audit.active_paths:
            if path == "sampling.max_tokens" and req.sampling.max_tokens == 0:
                audit.rejected(path, "vLLM requires max_tokens >= 1")
            elif (
                path
                in {
                    "structured_output.schema",
                    "structured_output.name",
                    "structured_output.strict",
                }
                and structured.format != "json_schema"
            ):
                audit.rejected(
                    path, "schema, name, and strict require format='json_schema'"
                )
            elif path == "structured_output.strict" and structured.strict is False:
                audit.rejected(
                    path, "vLLM cannot relax its grammar-constrained decoding"
                )
            elif (
                path in _CONSUMED_PATHS
                or path in _NOOP_PATHS
                or path == "dialect_options"
            ):
                continue
            elif not path.startswith("dialect_options."):
                audit.rejected(
                    path, "the vLLM native dialect does not support this request leaf"
                )
            else:
                self._validate_option(req, audit, path)

    @staticmethod
    def _validate_option(req: Request, audit: RequestAudit, path: str) -> None:
        key = path.removeprefix("dialect_options.")
        options = req.dialect_options
        assert options is not None
        if key in {"prefill", "prefix"}:
            audit.rejected(path, _PREFILL_UNSUPPORTED)
        elif key in _IR_OWNED_OPTION_KEYS:
            audit.rejected(path, f"use the request leaf {_IR_OWNED_OPTION_KEYS[key]}")
        elif key not in _SAMPLING_OPTION_KEYS:
            audit.rejected(
                path,
                "unknown vLLM option (vLLM ignores unknown fields silently); "
                "forwarded options are " + ", ".join(sorted(_SAMPLING_OPTION_KEYS)),
            )
        elif options.values[key] is None:
            audit.rejected(path, "a null option has no wire value")

    def prepare(self, req: Request, audit: RequestAudit) -> PreparedRequest:
        """Lower one validated request and register exact audit evidence."""

        audit.require_matches_request(req)
        audit.raise_rejections()
        if not isinstance(req.input, CompletionInput):
            raise DialectError("vllm_native requires CompletionInput")

        def consume(path: str, *wire: WireEvidence) -> None:
            audit.consumed(path, *wire)

        body: dict[str, JSONValue] = {
            "model": self._requested_model_id,
            "prompt": req.input.text,
            "stream": False,
        }
        consume("input.completion", WireObservation(("prompt",)))

        params = req.sampling
        simple: dict[str, JSONValue] = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "max_tokens": params.max_tokens,
            "stop": None if params.stop is None else list(params.stop),
            "seed": params.seed,
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
        }
        for name, value in simple.items():
            if value is not None:
                body[name] = value
                consume(f"sampling.{name}", WireObservation((name,)))
        if params.n != 1:
            body["n"] = params.n
            consume("sampling.n", WireObservation(("n",)))

        scoring = req.scoring
        if scoring.input_scoring or scoring.sampled_logprobs:
            body["return_token_ids"] = True
        if scoring.input_scoring:
            body["prompt_logprobs"] = scoring.top_logprobs
            consume(
                "scoring.input_scoring",
                _ScoringVerifier("prompt_logprobs", scoring.top_logprobs),
            )
        if scoring.sampled_logprobs:
            body["logprobs"] = scoring.top_logprobs
            consume(
                "scoring.sampled_logprobs",
                _ScoringVerifier("logprobs", scoring.top_logprobs),
            )
        if scoring.top_logprobs > 0:
            # One leaf, two breadth fields: claim the sampled one when sent.
            breadth = "logprobs" if scoring.sampled_logprobs else "prompt_logprobs"
            consume("scoring.top_logprobs", WireObservation((breadth,)))

        structured = _structured_outputs(req)
        if structured is not None:
            body["structured_outputs"] = structured
            consume("structured_output.format", _StructuredOutputsVerifier(structured))
            if req.structured_output.format == "json_schema":
                consume(
                    "structured_output.schema",
                    WireObservation(("structured_outputs", "json")),
                )
                if "structured_output.name" in audit.active_paths:
                    audit.noop("structured_output.name", _SCHEMA_NAME_NOOP)
                if "structured_output.strict" in audit.active_paths:
                    audit.noop("structured_output.strict", _STRICT_NOOP)

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
                # Top-level fields, each consumed against its own body key.
                body[key] = value
                consume(f"dialect_options.{key}", WireObservation((key,)))

        return PreparedRequest(
            operation=_OPERATION,
            body=body,
            context=_CompletionContext(
                prompt=req.input.text,
                n=params.n,
                input_scoring=scoring.input_scoring,
                sampled_logprobs=scoring.sampled_logprobs,
                top_logprobs=scoring.top_logprobs,
                structured_output=structured is not None,
            ),
        )

    async def execute(self, prepared: PreparedRequest) -> Response:
        """POST the prepared body to ``/completions`` and lift the reply."""

        if prepared.operation != _OPERATION:
            raise DialectError(f"unexpected vLLM operation {prepared.operation!r}")
        context = prepared.context
        if not isinstance(context, _CompletionContext):
            raise DialectError("prepared vLLM request has invalid context")
        body = prepared.thaw_body()
        headers = {"Content-Type": "application/json"}
        credential = self._connection.credential
        if credential is not None:
            headers["Authorization"] = f"Bearer {credential}"
        response = await self._connection.client.post(
            self._url, headers=headers, json=body
        )
        raise_for_provider_status(response, "vLLM")
        try:
            payload = json.loads(response.text)
        except ValueError as exc:
            raise OutputContractError("vLLM returned invalid JSON") from exc
        request_params = {
            key: value for key, value in body.items() if key not in {"model", "prompt"}
        }
        return self._lift(payload, context, request_params)

    def _lift(
        self,
        raw: object,
        context: _CompletionContext,
        request_params: Mapping[str, JSONValue],
    ) -> Response:
        payload = _mapping(raw, "vLLM completions reply")
        choices = _choices(payload, context.n)
        usage = _usage(payload.get("usage"))
        texts = tuple(cast(str, choice["text"]) for choice in choices)

        input_scoring: InputScoringResult | None = None
        if context.input_scoring:
            input_scoring = _input_scoring(
                choices[0],
                context.prompt,
                None if usage is None else usage.input_tokens,
                context.top_logprobs,
            )

        logprobs: tuple[TokenLogprob, ...] | None = None
        top_logprobs: tuple[tuple[TopKEntry, ...], ...] | None = None
        choice_logprobs: tuple[ChoiceLogprobs, ...] | None = None
        if context.sampled_logprobs and context.n > 1:
            choice_logprobs = tuple(
                _choice_logprobs(choice, context.top_logprobs) for choice in choices
            )
        elif context.sampled_logprobs:
            sampled = _choice_logprobs(choices[0], context.top_logprobs)
            logprobs = sampled.tokens
            top_logprobs = sampled.top_logprobs

        structured_output: StructuredOutput | None = None
        if context.structured_output:
            try:
                structured_output = StructuredOutput(json.loads(texts[0]))
            except json.JSONDecodeError as exc:
                raise OutputContractError(
                    "structured output is not valid JSON"
                ) from exc

        return Response(
            texts=texts,
            finish_reasons=tuple(
                reason if isinstance(reason := choice.get("finish_reason"), str) else ""
                for choice in choices
            ),
            structured_output=structured_output,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            choice_logprobs=choice_logprobs,
            input_scoring=input_scoring,
            usage=usage,
            request_params=request_params,
            response_model=_optional_string(payload, "model"),
            system_fingerprint=_optional_string(payload, "system_fingerprint"),
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
