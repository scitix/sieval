"""JSON-shape primitives for the modules that build reconciliation records.

`capabilities`, `requirements` and `reconcile` must agree on what counts as
representable JSON, or a record depends on which of them normalised it first.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import math
from collections.abc import Mapping

from sieval.core.types import JSONValue


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


def validate_nonempty_string(value: object, name: str) -> None:
    """Reject a missing, non-string, or empty identifier."""

    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
