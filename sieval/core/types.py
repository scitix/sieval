"""Common type aliases for the SiEval project."""

from collections.abc import Mapping

# Mapping (covariant in V) instead of dict (invariant) so that
# dict[str, float] satisfies Liskov against -> JSONValue returns.
# list stays concrete — Sequence would admit str (which is Sequence[str]).
type JSONValue = (
    str | int | float | bool | None | list["JSONValue"] | Mapping[str, "JSONValue"]
)
