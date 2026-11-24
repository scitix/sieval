"""Role-free decision evidence and null-preserving JSON records.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Self
from urllib.parse import urlsplit

from sieval.core.types import JSONValue
from sieval.core.utils.serialization import sieval_record

from ._validation import json_copy, nonempty_string, record_tuple


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
    """JSON encoded for persistence, with structured access through ``value``."""

    json: str

    def __post_init__(self) -> None:
        if not isinstance(self.json, str):
            raise TypeError("JsonPart.json must be a string; use from_value()")
        value = json.loads(self.json, object_pairs_hook=_unique_object)
        json_copy(value)
        if value is not None and not isinstance(value, dict | list):
            raise TypeError("JSON content must be an object, array or null")

    @classmethod
    def from_value(cls, value: JSONValue) -> Self:
        return cls(json.dumps(json_copy(value), ensure_ascii=False, allow_nan=False))

    @property
    def value(self) -> JSONValue:
        return json_copy(json.loads(self.json))


@sieval_record
@dataclass(frozen=True)
class TextEvidence:
    text: str

    def __post_init__(self) -> None:
        nonempty_string(self.text, "evidence text")


@sieval_record
@dataclass(frozen=True)
class ImageEvidence:
    """An HTTP(S) URL or base64 image data, without implicit downloads."""

    url: str | None = None
    data: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        if (self.url is None) == (self.data is None):
            raise ValueError("image evidence requires exactly one of url or data")
        if self.url is not None:
            parsed = urlsplit(self.url)
            if parsed.scheme not in {"https", "http"} or not parsed.hostname:
                raise ValueError("image URL must be an HTTP(S) URL")
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("image URL must not contain credentials")
        else:
            data = nonempty_string(self.data, "image data")
            if not isinstance(self.media_type, str) or not self.media_type.startswith(
                "image/"
            ):
                raise ValueError("inline image evidence requires an image media_type")
            try:
                base64.b64decode(data, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("image data must be base64") from exc


type EvidencePart = TextEvidence | JsonPart | ImageEvidence


@sieval_record
@dataclass(frozen=True)
class ContextInput:
    parts: tuple[EvidencePart, ...]

    def __post_init__(self) -> None:
        record_tuple(self.parts, (TextEvidence, JsonPart, ImageEvidence), "parts")
        if not self.parts:
            raise ValueError("context evidence must be non-empty")
        if any(
            isinstance(part, JsonPart) and part.value is None for part in self.parts
        ):
            raise ValueError("standalone null is not context evidence")
