"""Structured JSON content shared by evidence and decision guidance.

An encoded JSON record keeps explicit nulls intact under the existing record
serializer, which omits None-valued dictionary entries. Encoding is a storage
representation; a dialect receives structured data through ``value``.

AI-Generated Code - GPT-6 (OpenAI)
"""

import json
from dataclasses import dataclass
from typing import Self, cast

from sieval.core.types import JSONValue
from sieval.core.utils.serialization import sieval_record

from ._shared import copy_json_value


def _unique_object(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON content has duplicate key {key!r}")
        result[key] = value
    return result


@sieval_record
@dataclass(frozen=True)
class JsonPart:
    """Explicit structured content; use ``from_value`` for Python values.

    Objects, arrays and null are supported. Null is useful for an explicitly
    present question-guidance value; it is not standalone context evidence.
    ``value`` returns a fresh object, so mutation cannot change a request.
    """

    json: str

    def __post_init__(self) -> None:
        if not isinstance(self.json, str):
            raise TypeError("JsonPart.json must be a JSON string; use from_value()")
        value = json.loads(self.json, object_pairs_hook=_unique_object)
        copy_json_value(value, "JSON content")
        if value is not None and not isinstance(value, dict | list):
            raise TypeError("JSON content must be an object, array or null")

    @classmethod
    def from_value(cls, value: JSONValue) -> Self:
        copied = copy_json_value(value, "JSON content")
        return cls(
            json.dumps(
                copied, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        )

    @property
    def value(self) -> JSONValue:
        return cast(JSONValue, json.loads(self.json))
