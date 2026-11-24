"""Validation shared by the plugin's records and wire adapters.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import math
from collections.abc import Mapping

from sieval.core.types import JSONValue


def nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def probability(value: object, name: str) -> float:
    result = number(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def optional_probability(value: object, name: str) -> float | None:
    return None if value is None else probability(value, name)


def integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def record_tuple(value: object, item_type: type | tuple[type, ...], name: str) -> None:
    if not isinstance(value, tuple) or not all(
        isinstance(item, item_type) for item in value
    ):
        raise TypeError(f"{name} must be a tuple of the declared record types")


def json_copy(value: object, path: str = "JSON") -> JSONValue:
    """Detach a JSON value, rejecting non-finite numbers and non-string keys."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return number(value, path)
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            result[key] = json_copy(item, f"{path}.{key}")
        return result
    if isinstance(value, list | tuple):
        return [json_copy(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def object_value(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise TypeError(f"{path} must be an object with string keys")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def array_value(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be an array")
    return list(value)
