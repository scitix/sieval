"""JSON-shape primitives for the modules that build persisted records.

`capabilities`, `requirements` and `reconcile` must agree on what counts as
representable JSON, or a record depends on which of them normalised it first.
`model` and `_legacy_bridge` need the same agreement, so the primitives live
here rather than in either caller — notably not in `_legacy_bridge`, which is
scheduled to be deleted whole. The two coercers differ only in how an error
names a rejected leaf; merging them would change existing messages.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Never, Self, cast

from sieval.core.types import JSONValue


class _FrozenJSONList(list[JSONValue]):
    """List-shaped JSON value that rejects mutation while preserving equality."""

    def _immutable(self, *args: object, **kwargs: object) -> Never:
        del self, args, kwargs
        raise TypeError("frozen JSON sequences do not support mutation")

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        del memo
        return self


# ``list`` exposes several mutation spellings, including in-place operators.
# Install the same guard for all of them so nested JSON sequences remain
# list-compatible without leaving a normal mutation path open.
for _method_name in (
    "append",
    "clear",
    "extend",
    "insert",
    "pop",
    "remove",
    "reverse",
    "sort",
    "__delitem__",
    "__iadd__",
    "__imul__",
    "__setitem__",
):
    setattr(_FrozenJSONList, _method_name, _FrozenJSONList._immutable)
del _method_name


def copy_json_value(value: object, path: str) -> JSONValue:
    """Deep-copy *value* into plain JSON, rejecting what it cannot represent.

    ``path`` is the dotted location errors name, so a rejection points at the
    offending leaf rather than the whole record.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain a non-finite float")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            copied[key] = copy_json_value(item, f"{path}.{key}")
        return copied
    if isinstance(value, (list, tuple)):
        return [
            copy_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def freeze_json_value(value: object, path: str) -> JSONValue:
    """Detach and recursively freeze one JSON-compatible value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        return freeze_json_mapping(cast(Mapping[str, JSONValue], value), path)
    if isinstance(value, (list, tuple)):
        return _FrozenJSONList(
            freeze_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def freeze_json_mapping(
    value: Mapping[str, JSONValue], path: str
) -> Mapping[str, JSONValue]:
    """Detach and recursively freeze a JSON mapping."""

    copied: dict[str, JSONValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{path} keys must be strings")
        copied[key] = freeze_json_value(item, f"{path}.{key}")
    return MappingProxyType(copied)


def thaw_json_mapping(
    value: Mapping[str, JSONValue], path: str
) -> dict[str, JSONValue]:
    """Return a detached, mutable plain-JSON copy of a frozen mapping."""

    thawed = copy_json_value(value, path)
    if not isinstance(thawed, dict):
        raise TypeError(f"{path} must be a mapping")
    return cast(dict[str, JSONValue], thawed)


def named_json_value(value: object, name: str) -> JSONValue:
    """Validate and detach a JSON value, naming the offending leaf on failure.

    Sequences are ``list``/``tuple`` only: the result is persisted, and a
    ``set`` would serialize in hash order while a generator would serialize
    as ``[]`` once consumed.
    """

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain a non-finite float")
        return value
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} keys must be strings")
            result[key] = named_json_value(item, f"{name}.{key}")
        return result
    if isinstance(value, list | tuple):
        return [named_json_value(item, name) for item in value]
    raise TypeError(f"{name} must be JSON-compatible, got {type(value).__name__}")


def validate_nonempty_string(value: object, name: str) -> None:
    """Reject a missing, non-string, or empty identifier."""

    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
