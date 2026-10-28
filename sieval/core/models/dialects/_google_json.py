"""Strict JSON decoding shared by the Google dialects.

Rejects duplicate object keys and the non-finite constants ``NaN`` and
``Infinity`` that :func:`json.loads` accepts by default.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from typing import cast

from sieval.core.models.dialect import OutputContractError
from sieval.core.types import JSONValue


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def reject_duplicate_keys(
    pairs: list[tuple[str, JSONValue]],
) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_loads(payload: str, path: str) -> JSONValue:
    try:
        return cast(
            JSONValue,
            json.loads(
                payload,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_json_constant,
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OutputContractError(f"{path} is not strict JSON") from exc
