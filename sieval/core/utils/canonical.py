"""Canonical JSON, the text every identity digest and call key hashes.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime


def _default(value: object) -> object:
    """Encode the non-JSON values a raw sample or a model default may hold."""
    if isinstance(value, (set, frozenset)):
        return sorted(canonical_json(v) for v in value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, bytes):
        return {"__bytes__": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        return dict(value)
    tolist = getattr(value, "tolist", None)  # numpy arrays and scalars
    if callable(tolist):
        return tolist()
    tobytes = getattr(value, "tobytes", None)  # PIL images
    if callable(tobytes):
        raw = tobytes()
        if isinstance(raw, bytes):
            return {"__bytes__": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"cannot digest a {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Canonical JSON: sorted keys, no whitespace, ASCII only.

    NaN and the infinities are written as the bare tokens ``NaN``,
    ``Infinity`` and ``-Infinity``. That is not strict JSON, but this text is
    only ever hashed: refusing them would leave a sample holding a missing
    float with no digest, and spelling them as strings would make them equal
    to the string ``"NaN"``.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=True,
        default=_default,
    )
