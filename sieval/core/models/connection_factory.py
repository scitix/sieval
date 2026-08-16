"""Family-keyed runtime connection construction for model dialects.

Factories own client construction and retry-policy encoding. Credentials are
runtime-only inputs: they may be retained by a connection for its dialect, but
never participate in a connection identity, registry descriptor, or repr.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import httpx
from openai import AsyncOpenAI

#: The per-request timeout every sieval client that talks to a model declares.
#:
#: Declared rather than inherited: the OpenAI SDK defaults ``read`` to 600 and
#: ``httpx`` defaults every phase to 5, so a dialect's connection family silently
#: decided how long a generation could take, and either library can change its
#: default in a patch release. The value is what ``openai_sdk`` already had in
#: effect, so it moves only ``async_http_json`` -- whose dialects have no
#: implemented transport yet, making its 5s a trap rather than a live bug.
#:
#: Scope is clients whose call carries no bound of its own. The code-evaluator and
#: health-check clients each pass a per-request ``timeout`` derived from a budget
#: they already enforce server-side, which a client-level default would duplicate
#: or contradict.
#:
#: ``connect`` is short so an unroutable endpoint fails fast. ``read`` is long
#: because a premature read timeout is indistinguishable, in a result directory,
#: from the model having produced nothing. Note it bounds two different
#: quantities: the gap between chunks when streamed (a stall tolerance), the whole
#: response when not -- every one of a request's ``n`` rollouts together, since
#: sieval asks for all of them in one call. That second reading is the one a long
#: high-``n`` run can exhaust. ``SchedulingParams.stream`` is ``False``; the legacy
#: ``ChatModel``/``GenModel`` wrappers send ``True``.
DEFAULT_REQUEST_TIMEOUT = httpx.Timeout(600.0, connect=5.0)


class UnknownConnectionFamily(ValueError):
    """No registered factory owns the requested connection family."""


@dataclass(frozen=True)
class ConnectionRequest:
    """Runtime-only inputs for constructing one owned connection."""

    endpoint: str
    credential: str | None = field(repr=False)
    max_retries: int

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str) or not self.endpoint:
            raise ValueError("ConnectionRequest.endpoint must not be empty")
        if self.credential is not None and not isinstance(self.credential, str):
            raise TypeError("ConnectionRequest.credential must be a string or None")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise TypeError("ConnectionRequest.max_retries must be an integer")
        if self.max_retries < 0:
            raise ValueError("ConnectionRequest.max_retries must be non-negative")


class AsyncHTTPJSONConnection:
    """Provider-neutral async JSON client plus an unclassified credential.

    The connection deliberately does not translate ``credential`` into an
    HTTP header. Anthropic, Google, and native serving dialects choose their
    own authentication scheme when they are implemented.
    """

    __slots__ = ("_client", "_credential")

    def __init__(
        self,
        client: httpx.AsyncClient,
        credential: str | None,
    ) -> None:
        self._client = client
        self._credential = credential

    @property
    def client(self) -> httpx.AsyncClient:
        """Return the owned provider-neutral HTTP client."""

        return self._client

    @property
    def credential(self) -> str | None:
        """Return the runtime credential for dialect-specific classification."""

        return self._credential

    async def aclose(self) -> None:
        """Close the owned HTTP client."""

        await self._client.aclose()


type ConnectionBuilder = Callable[[ConnectionRequest], object]


@dataclass(frozen=True)
class ConnectionFactorySpec:
    """One connection family, its builder, and stable retry-policy namespace."""

    connection_family: str
    retry_policy_prefix: str
    builder: ConnectionBuilder = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.connection_family, str) or not self.connection_family:
            raise ValueError("connection_family must not be empty")
        if (
            not isinstance(self.retry_policy_prefix, str)
            or not self.retry_policy_prefix
            or not self.retry_policy_prefix.endswith("=")
        ):
            raise ValueError("retry_policy_prefix must be non-empty and end with '='")
        if not callable(self.builder):
            raise TypeError("connection builder must be callable")

    def retry_policy(self, max_retries: int) -> str:
        """Encode a family-specific, secret-free retry policy."""

        _validate_max_retries(max_retries)
        return f"{self.retry_policy_prefix}{max_retries}"

    def parse_retry_policy(self, retry_policy: str) -> int:
        """Validate this family's policy and return its retry count."""

        if not retry_policy.startswith(self.retry_policy_prefix):
            raise ValueError(
                f"retry policy {retry_policy!r} does not belong to connection "
                f"family {self.connection_family!r}"
            )
        suffix = retry_policy.removeprefix(self.retry_policy_prefix)
        if not suffix.isascii() or not suffix.isdigit():
            raise ValueError(f"invalid retry policy {retry_policy!r}")
        max_retries = int(suffix)
        _validate_max_retries(max_retries)
        return max_retries


class ConnectionFactoryRegistry:
    """Immutable family registry used by setup and allocation paths."""

    def __init__(self, specs: Iterable[ConnectionFactorySpec]) -> None:
        by_family: dict[str, ConnectionFactorySpec] = {}
        for spec in specs:
            if not isinstance(spec, ConnectionFactorySpec):
                raise TypeError("connection registry entries must be factory specs")
            if spec.connection_family in by_family:
                raise ValueError(
                    f"duplicate connection family {spec.connection_family!r}"
                )
            by_family[spec.connection_family] = spec
        self._specs: Mapping[str, ConnectionFactorySpec] = MappingProxyType(by_family)

    @property
    def families(self) -> frozenset[str]:
        return frozenset(self._specs)

    @property
    def specs(self) -> Mapping[str, ConnectionFactorySpec]:
        return self._specs

    def get(self, connection_family: str) -> ConnectionFactorySpec:
        """Resolve a family before any connection allocation can occur."""

        try:
            return self._specs[connection_family]
        except KeyError as exc:
            known = ", ".join(sorted(self._specs))
            raise UnknownConnectionFamily(
                f"unknown connection family {connection_family!r}; "
                f"registered families: {known}"
            ) from exc

    def retry_policy(self, connection_family: str, max_retries: int) -> str:
        return self.get(connection_family).retry_policy(max_retries)

    def validate_retry_policy(
        self,
        connection_family: str,
        retry_policy: str,
    ) -> int:
        return self.get(connection_family).parse_retry_policy(retry_policy)

    def create(
        self,
        connection_family: str,
        request: ConnectionRequest,
    ) -> object:
        """Construct a connection only after exact family lookup succeeds."""

        return self.get(connection_family).builder(request)

    def with_factory(self, spec: ConnectionFactorySpec) -> "ConnectionFactoryRegistry":
        """Return an extended registry without mutating process-global state."""

        return ConnectionFactoryRegistry((*self._specs.values(), spec))


def _validate_max_retries(max_retries: int) -> None:
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise TypeError("max_retries must be an integer")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")


def _openai_sdk_connection(request: ConnectionRequest) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=request.endpoint,
        api_key=request.credential,
        max_retries=request.max_retries,
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )


def _async_http_json_connection(
    request: ConnectionRequest,
) -> AsyncHTTPJSONConnection:
    transport = httpx.AsyncHTTPTransport(retries=request.max_retries)
    client = httpx.AsyncClient(
        base_url=request.endpoint,
        transport=transport,
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )
    return AsyncHTTPJSONConnection(client, request.credential)


OPENAI_SDK_FACTORY = ConnectionFactorySpec(
    connection_family="openai_sdk",
    retry_policy_prefix="openai-sdk:max-retries=",
    builder=_openai_sdk_connection,
)

ASYNC_HTTP_JSON_FACTORY = ConnectionFactorySpec(
    connection_family="async_http_json",
    retry_policy_prefix="httpx-transport:max-connect-retries=",
    builder=_async_http_json_connection,
)

CONNECTION_FACTORY_REGISTRY = ConnectionFactoryRegistry(
    (OPENAI_SDK_FACTORY, ASYNC_HTTP_JSON_FACTORY)
)
