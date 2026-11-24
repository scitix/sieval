"""Shared decision client contract and recorded HTTP execution.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol
from urllib.parse import urlsplit

import httpx

from sieval.core.models import (
    EvalSubject,
    RecordingAsyncClient,
    ServingFingerprint,
    SubjectMeta,
    subject_call,
    subject_call_key,
)
from sieval.core.types import JSONValue
from sieval.core.utils.serialization import sieval_record

from ._validation import json_copy, nonempty_string, number, object_value
from .content import ContextInput, JsonPart
from .decisions import DecisionParams, DecisionResult, validate_decision_result


@sieval_record
@dataclass(frozen=True)
class DecisionRequest:
    context: ContextInput
    decisions: DecisionParams

    def __post_init__(self) -> None:
        if not isinstance(self.context, ContextInput):
            raise TypeError("context must be ContextInput")
        if not isinstance(self.decisions, DecisionParams):
            raise TypeError("decisions must be DecisionParams")


@sieval_record
@dataclass(frozen=True)
class DecisionResponse:
    """Typed answers and the original response, without a generation wrapper."""

    result: DecisionResult
    model: str
    usage: JsonPart | None = None
    raw: JsonPart | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, DecisionResult):
            raise TypeError("result must be DecisionResult")
        nonempty_string(self.model, "response model")
        if self.usage is not None:
            if not isinstance(self.usage, JsonPart):
                raise TypeError("usage must be JsonPart")
            object_value(self.usage.value, "usage")
        if self.raw is not None:
            if not isinstance(self.raw, JsonPart):
                raise TypeError("raw must be JsonPart")
            object_value(self.raw.value, "raw response")


class DecisionsClient(EvalSubject, Protocol):
    async def decide(self, request: DecisionRequest) -> DecisionResponse: ...

    async def aclose(self) -> None: ...


class DecisionProtocolError(ValueError):
    """A successful HTTP response violated the decision protocol."""


class UnsupportedDecisionInput(ValueError):
    """The provider cannot represent the requested decision input."""


class HTTPDecisionsClient(ABC):
    """Transport shared by provider adapters; retries belong to SiEval's client."""

    provider: ClassVar[str]
    protocol_version: ClassVar[str] = "1"
    default_base_url: ClassVar[str]
    default_api_key_env: ClassVar[str]
    allowed_options: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str | None = None,
        api_key_env: str | None = None,
        revision: str | None = None,
        options: Mapping[str, JSONValue] | None = None,
        timeout: float = 60,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model_name = nonempty_string(model_name, "model_name")
        if revision is not None:
            nonempty_string(revision, "revision")
        self._revision = revision
        self._base_url = (base_url or self.default_base_url).rstrip("/")
        parsed = urlsplit(self._base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query or fragment")
        self._api_key_env = api_key_env or self.default_api_key_env
        nonempty_string(self._api_key_env, "api_key_env")
        self._options = JsonPart.from_value(options or {})
        unknown = set(options or {}) - self.allowed_options
        if unknown:
            raise ValueError(f"unsupported {self.provider} options: {sorted(unknown)}")
        if number(timeout, "timeout") <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("max_retries must be an integer")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._timeout = timeout
        self._max_retries = max_retries
        self._transport = transport
        self._http: RecordingAsyncClient | None = None

    def subject_meta(self) -> SubjectMeta:
        metadata: dict[str, JSONValue] = {
            "model": self.model_name,
            "provider": self.provider,
        }
        if self._revision is not None:
            metadata["revision"] = self._revision
        return metadata

    def call_key(self) -> str:
        return subject_call_key(self.subject_meta())

    def request_args(self) -> JSONValue:
        return {
            "protocol": f"sieval-decisions/{self.provider}/{self.protocol_version}",
            "base_url": self._base_url,
            "options": self._options.value,
        }

    def serving(self) -> ServingFingerprint | None:
        return None

    @abstractmethod
    def endpoint(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_request(self, request: DecisionRequest) -> dict[str, JSONValue]:
        raise NotImplementedError

    @abstractmethod
    def parse_response(
        self, data: Mapping[str, object], request: DecisionRequest
    ) -> DecisionResponse:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self._api_key_env)
        if not key:
            raise ValueError(f"set {self._api_key_env} before making requests")
        return {"Authorization": f"Bearer {key}"}

    def _request_options(self) -> dict[str, JSONValue]:
        value = self._options.value
        return {
            key: json_copy(item) for key, item in object_value(value, "options").items()
        }

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        if not isinstance(request, DecisionRequest):
            raise TypeError("request must be DecisionRequest")
        body = self.build_request(request)
        headers = self._headers()
        if self._http is None:
            self._http = RecordingAsyncClient(
                timeout=self._timeout,
                max_retries=self._max_retries,
                transport=self._transport,
                follow_redirects=False,
            )
        with subject_call(self) as call:
            response = await self._http.post(
                self.endpoint(), json=body, headers=headers
            )
            response.raise_for_status()
            try:
                data = object_value(response.json(), "response")
                parsed = self.parse_response(data, request)
                validate_decision_result(request.decisions, parsed.result)
            except (TypeError, ValueError, KeyError) as exc:
                raise DecisionProtocolError(
                    f"invalid {self.provider} response: {exc}"
                ) from exc
            usage = (
                None
                if parsed.usage is None
                else {
                    key: json_copy(value)
                    for key, value in object_value(parsed.usage.value, "usage").items()
                }
            )
            call.record(
                request_params={"body": body},
                usage=usage,
                response_model=parsed.model,
            )
        return parsed

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


def response_record(
    data: Mapping[str, object],
    result: DecisionResult,
    model_name: object,
    usage: Mapping[str, JSONValue] | None,
) -> DecisionResponse:
    return DecisionResponse(
        result=result,
        model=nonempty_string(model_name, "response model"),
        usage=None if usage is None else JsonPart.from_value(usage),
        raw=JsonPart.from_value(json_copy(data)),
    )
