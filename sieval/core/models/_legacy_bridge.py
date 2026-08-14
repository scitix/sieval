"""One-cycle compatibility bridge between task kwargs and the canonical IR.

Tasks still call ``agenerate``/``alogprobs`` with loose keyword arguments and
read a ``ModelOutput`` back; the canonical plane speaks ``Request``/``Response``
only. Separated from ``model`` by lifetime, so the move to ``arun`` deletes this
file as a unit — true only while it stays free-function shaped and free of
primitives the canonical plane also needs. ``Model.meta()`` is the one thread to
cut by hand: it is public, has a caller outside the legacy path, and returns the
``ModelMeta`` defined below.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import NotRequired, TypedDict, cast

from sieval.core.types import JSONValue
from sieval.core.utils.serialization import sieval_record

from ._shared import named_json_value
from .deployment import BINDING_RESOURCE_KEYS
from .ir import (
    CompletionInput,
    DialectOptions,
    ModelInput,
    ModelProvenance,
    OpaqueContinuation,
    ReasoningParams,
    Request,
    Response,
    SamplingParams,
    SchedulingParams,
    ScoringParams,
    SessionParams,
    StructuredOutputParams,
    TokenLogprob,
    ToolParams,
    UsageStats,
)


class ModelUsage(TypedDict):
    """Token usage statistics from a single model API call.

    The optional keys are breakdowns of the first two counts, not addends, and
    are written only where the server reported them -- so an absent key means
    "not reported", never zero. Read them with ``.get()``.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: NotRequired[int]
    cached_tokens: NotRequired[int]
    accepted_prediction_tokens: NotRequired[int]
    rejected_prediction_tokens: NotRequired[int]
    reported_total_tokens: NotRequired[int]


class ModelMeta(TypedDict):
    """Persisted identity and defaults for one model call."""

    model: str
    api_base: str | None
    default_params: dict[str, JSONValue]
    extra: NotRequired[dict[str, JSONValue]]
    provenance: NotRequired[ModelProvenance]


class ModelCallMeta(TypedDict):
    """Per-API-call metadata: model info, usage, params, finish reasons."""

    model: ModelMeta
    usage: NotRequired[ModelUsage]
    request_params: NotRequired[dict[str, JSONValue]]
    finish_reasons: NotRequired[list[str]]
    response_model: NotRequired[str]
    system_fingerprint: NotRequired[str | None]


@sieval_record
@dataclass
class ModelOutput:
    """Legacy return type preserved while tasks migrate to ``Response``."""

    model: ModelMeta
    texts: list[str]
    finish_reasons: list[str] | None = None
    reasoning_texts: list[str] | None = None
    logprobs_tokens: list[str] | None = None
    logprobs: list[float | None] | None = None
    top_logprobs: list[dict[str, float]] | None = None
    usage: ModelUsage | None = None
    request_params: dict[str, JSONValue] | None = None
    response_model: str | None = None
    system_fingerprint: str | None = None


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _pop_compatible_alias(
    values: dict[str, object],
    canonical: str,
    legacy: str,
    *,
    default: object,
) -> object:
    """Pop one semantic value without silently resolving two conflicting owners."""

    missing = object()
    canonical_value = values.pop(canonical, missing)
    legacy_value = values.pop(legacy, missing)
    if (
        canonical_value is not missing
        and legacy_value is not missing
        and canonical_value is not None
        and legacy_value is not None
        and canonical_value != legacy_value
    ):
        raise ValueError(
            f"{canonical} conflicts with its legacy alias {legacy}: "
            f"{canonical_value!r} != {legacy_value!r}"
        )
    if canonical_value is not missing and canonical_value is not None:
        return canonical_value
    if legacy_value is not missing:
        return legacy_value
    if canonical_value is not missing:
        return canonical_value
    return default


def _coerce_structured_output(value: object) -> StructuredOutputParams:
    if value is None:
        return StructuredOutputParams()
    if isinstance(value, StructuredOutputParams):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("response_format must be a mapping")
    value_mapping = cast(Mapping[str, object], value)
    format_ = value_mapping.get("type")
    if format_ == "json_object":
        return StructuredOutputParams(format="json_object")
    if format_ != "json_schema":
        raise ValueError(f"unsupported response_format type: {format_!r}")
    raw_schema = value_mapping.get("json_schema")
    if not isinstance(raw_schema, Mapping):
        raise TypeError("json_schema response_format requires a mapping")
    raw_schema_mapping = cast(Mapping[str, object], raw_schema)
    schema = raw_schema_mapping.get("schema")
    if not isinstance(schema, Mapping):
        raise TypeError("json_schema response_format requires `schema`")
    name = raw_schema_mapping.get("name")
    strict = raw_schema_mapping.get("strict")
    if name is not None and not isinstance(name, str):
        raise TypeError("json_schema name must be a string")
    if strict is not None and not isinstance(strict, bool):
        raise TypeError("json_schema strict must be a bool")
    return StructuredOutputParams(
        format="json_schema",
        schema=cast(Mapping[str, JSONValue], named_json_value(schema, "schema")),
        name=name,
        strict=strict,
    )


def validate_n(kwargs: Mapping[str, object]) -> int:
    n = kwargs.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__}: {n!r}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return n


def kwargs_to_request(
    dialect_id: str,
    input_: ModelInput,
    final_kwargs: Mapping[str, object],
) -> Request:
    kw = dict(final_kwargs)
    reserved = sorted(BINDING_RESOURCE_KEYS & set(kw))
    if reserved:
        raise ValueError(
            "request arguments cannot change binding resources: " + ", ".join(reserved)
        )
    n = validate_n(kw)
    kw.pop("n", None)

    stream = kw.pop("stream", True)
    if not isinstance(stream, bool):
        raise TypeError("stream must be a bool")

    max_tokens = _pop_compatible_alias(
        kw,
        "max_tokens",
        "max_completion_tokens",
        default=None,
    )
    temperature_value = kw.pop("temperature", None)
    top_p_value = kw.pop("top_p", None)
    top_k_value = kw.pop("top_k", None)
    seed_value = kw.pop("seed", None)
    frequency_penalty_value = kw.pop("frequency_penalty", None)
    presence_penalty_value = kw.pop("presence_penalty", None)
    stop = kw.pop("stop", None)
    stop_value: tuple[str, ...] | None = None
    if stop is not None:
        if isinstance(stop, str):
            stop_value = (stop,)
        elif isinstance(stop, list | tuple):
            values = tuple(stop)
            if not all(isinstance(item, str) for item in values):
                raise TypeError("stop must contain only strings")
            stop_value = cast(tuple[str, ...], values)
        else:
            # ``list``/``tuple`` only, matching ``named_json_value``: this
            # value is echoed into the persisted ``request_params``, and a
            # ``set`` would land there in hash order.
            raise TypeError("stop must be a string, list, or tuple of strings")
    sampling = SamplingParams(
        temperature=_optional_float(temperature_value, "temperature"),
        top_p=_optional_float(top_p_value, "top_p"),
        top_k=_optional_int(top_k_value, "top_k"),
        max_tokens=_optional_int(max_tokens, "max_tokens"),
        stop=stop_value,
        seed=_optional_int(seed_value, "seed"),
        frequency_penalty=_optional_float(frequency_penalty_value, "frequency_penalty"),
        presence_penalty=_optional_float(presence_penalty_value, "presence_penalty"),
        n=n,
    )

    return_logprobs = kw.pop("return_logprobs", False)
    if not isinstance(return_logprobs, bool):
        raise TypeError("return_logprobs must be a bool")
    top_logprobs = kw.pop("top_logprobs", None)
    legacy_logprobs = kw.pop("logprobs", None)
    sampled = return_logprobs
    breadth = 0
    if isinstance(legacy_logprobs, bool):
        sampled = sampled or legacy_logprobs
    elif legacy_logprobs is not None:
        if isinstance(legacy_logprobs, bool) or not isinstance(legacy_logprobs, int):
            raise TypeError("logprobs must be a bool or integer")
        sampled = True
        breadth = legacy_logprobs
    if top_logprobs is not None:
        if isinstance(top_logprobs, bool) or not isinstance(top_logprobs, int):
            raise TypeError("top_logprobs must be an integer")
        breadth = top_logprobs
        sampled = sampled or top_logprobs > 0
    score_input = _pop_compatible_alias(
        kw,
        "score_input",
        "echo",
        default=False,
    )
    if not isinstance(score_input, bool):
        raise TypeError("echo/score_input must be a bool")
    scoring = ScoringParams(
        input_scoring=score_input,
        sampled_logprobs=sampled,
        top_logprobs=breadth,
    )

    reasoning = kw.pop("reasoning", None)
    effort = kw.pop("reasoning_effort", None)
    if effort is not None and not isinstance(effort, str):
        raise TypeError("reasoning_effort must be a string")
    if reasoning is None:
        reasoning_params = ReasoningParams(effort=effort)
    elif isinstance(reasoning, ReasoningParams):
        if effort is not None:
            raise ValueError("reasoning and reasoning_effort cannot both be set")
        reasoning_params = reasoning
    else:
        raise TypeError("reasoning must be ReasoningParams")

    raw_tools = kw.pop("tools", ())
    if raw_tools is None:
        raw_tools = ()
    if not isinstance(raw_tools, Iterable) or isinstance(
        raw_tools, Mapping | str | bytes
    ):
        raise TypeError("tools must be an iterable of mappings")
    functions: list[Mapping[str, JSONValue]] = []
    for index, tool in enumerate(raw_tools):
        if not isinstance(tool, Mapping):
            raise TypeError("tools must contain mappings")
        functions.append(
            cast(
                Mapping[str, JSONValue],
                named_json_value(tool, f"tools[{index}]"),
            )
        )
    choice = named_json_value(kw.pop("tool_choice", None), "tool_choice")
    parallel = kw.pop("parallel_tool_calls", None)
    if parallel is not None and not isinstance(parallel, bool):
        raise TypeError("parallel_tool_calls must be a bool")
    tools = ToolParams(
        functions=tuple(functions),
        choice=choice,
        parallel=parallel,
    )

    structured_output = _coerce_structured_output(kw.pop("response_format", None))
    previous_response_id = _pop_compatible_alias(
        kw,
        "previous_response_id",
        "session_id",
        default=None,
    )
    if previous_response_id is not None and not isinstance(previous_response_id, str):
        raise TypeError("previous_response_id must be a string")
    continuation = kw.pop("opaque_continuation", None)
    if continuation is not None and not isinstance(continuation, OpaqueContinuation):
        raise TypeError("opaque_continuation must be OpaqueContinuation")

    suffix = kw.pop("suffix", None)
    if suffix is not None:
        if not isinstance(suffix, str):
            raise TypeError("suffix must be a string")
        if not isinstance(input_, CompletionInput):
            raise TypeError("suffix requires CompletionInput")
        if input_.suffix is not None and input_.suffix != suffix:
            raise ValueError("suffix conflicts with CompletionInput.suffix")
        input_ = replace(input_, suffix=suffix)

    options: dict[str, JSONValue] = {}
    explicit_options = kw.pop("dialect_options", None)
    if explicit_options is not None:
        if not isinstance(explicit_options, DialectOptions):
            raise TypeError("dialect_options must be DialectOptions")
        if explicit_options.dialect_id != dialect_id:
            raise ValueError("dialect_options target another dialect")
        reserved = sorted(BINDING_RESOURCE_KEYS & set(explicit_options.values))
        if reserved:
            raise ValueError(
                "dialect_options cannot contain binding resources: "
                + ", ".join(reserved)
            )
        options.update(explicit_options.values)
    for container_name in ("extra_body", "extra_wire_params"):
        raw = kw.pop(container_name, None)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise TypeError(f"{container_name} must be a mapping")
        for key, value in raw.items():
            if not isinstance(key, str):
                raise TypeError(f"{container_name} keys must be strings")
            if key in BINDING_RESOURCE_KEYS:
                raise ValueError(
                    f"{container_name} cannot contain binding resource {key!r}"
                )
            if key in options:
                raise ValueError(f"duplicate dialect option {key!r}")
            options[key] = named_json_value(value, f"{container_name}.{key}")
    for key, value in kw.items():
        if key in options:
            raise ValueError(f"duplicate dialect option {key!r}")
        options[key] = named_json_value(value, key)

    return Request(
        input=input_,
        sampling=sampling,
        scoring=scoring,
        reasoning=reasoning_params,
        tools=tools,
        structured_output=structured_output,
        session=SessionParams(
            previous_response_id=previous_response_id,
            opaque_continuation=continuation,
        ),
        scheduling=SchedulingParams(stream=stream),
        dialect_options=(DialectOptions(dialect_id, options) if options else None),
    )


def _model_usage(counts: UsageStats) -> ModelUsage:
    """Project usage onto the record shape, omitting whatever went unreported.

    The three totals are always written. Each breakdown key is written only
    where the server actually reported it -- storing a zero instead would put a
    measurement on disk that was never taken, and any later average over a
    fleet of mixed servers would silently fold those in.
    """
    usage: ModelUsage = {
        "input_tokens": counts.input_tokens,
        "output_tokens": counts.output_tokens,
        "total_tokens": counts.total_tokens,
    }
    if counts.reasoning_tokens is not None:
        usage["reasoning_tokens"] = counts.reasoning_tokens
    if counts.cached_tokens is not None:
        usage["cached_tokens"] = counts.cached_tokens
    if counts.accepted_prediction_tokens is not None:
        usage["accepted_prediction_tokens"] = counts.accepted_prediction_tokens
    if counts.rejected_prediction_tokens is not None:
        usage["rejected_prediction_tokens"] = counts.rejected_prediction_tokens
    if counts.reported_total_tokens is not None:
        usage["reported_total_tokens"] = counts.reported_total_tokens
    return usage


def response_to_model_output(model_meta: ModelMeta, response: Response) -> ModelOutput:
    """Shape a canonical ``Response`` into the legacy ``ModelOutput``."""

    # The caller may keep this mapping, and provenance is persisted.
    model_meta = model_meta.copy()

    segments: list[TokenLogprob] = []
    if response.input_scoring is not None:
        segments.extend(response.input_scoring.token_logprobs)
    if response.logprobs is not None:
        segments.extend(response.logprobs)
    logprobs_present = (
        response.input_scoring is not None or response.logprobs is not None
    )
    logprobs_tokens = [item.token for item in segments] if logprobs_present else None
    logprobs = [item.logprob for item in segments] if logprobs_present else None

    top_logprobs: list[dict[str, float]] | None = None
    if response.top_logprobs is not None:
        top_logprobs = []
        for position in response.top_logprobs:
            merged: dict[str, float] = {}
            for item in position:
                previous = merged.get(item.token)
                if previous is None or item.logprob > previous:
                    merged[item.token] = item.logprob
            top_logprobs.append(merged)
        top_logprobs = top_logprobs or None

    usage = _model_usage(response.usage) if response.usage is not None else None
    reasoning_texts: list[str] | None = None
    if response.reasoning is not None:
        reasoning_texts = [
            item.text or "" if item is not None else "" for item in response.reasoning
        ]
        if not any(reasoning_texts):
            reasoning_texts = None

    if response.provenance is not None:
        model_meta["provenance"] = response.provenance
    return ModelOutput(
        model=model_meta,
        texts=list(response.texts),
        finish_reasons=(
            list(response.finish_reasons)
            if response.finish_reasons is not None
            else None
        ),
        reasoning_texts=reasoning_texts,
        logprobs_tokens=logprobs_tokens,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        usage=usage,
        request_params=(
            dict(response.request_params)
            if response.request_params is not None
            else None
        ),
        response_model=response.response_model,
        system_fingerprint=response.system_fingerprint,
    )
