"""Canonical JSON encoding shared by identity digests and call keys.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import hashlib
import json
from datetime import UTC, datetime
from types import MappingProxyType

import numpy as np
import pytest

from sieval.core.utils.canonical import canonical_json


def test_encodes_values_a_raw_sample_may_hold():
    raw = {
        "ids": (1, 2),
        "tags": {"y", "x"},
        "blob": b"\x00\x01",
        "score": np.float32(0.5),
        "vec": np.array([1, 2]),
        "when": datetime(2026, 1, 1, tzinfo=UTC),
    }
    decoded = json.loads(canonical_json(raw))
    assert decoded["ids"] == [1, 2]
    assert decoded["tags"] == ['"x"', '"y"']
    assert decoded["blob"] == {"__bytes__": hashlib.sha256(b"\x00\x01").hexdigest()}
    assert decoded["score"] == 0.5
    assert decoded["vec"] == [1, 2]
    assert decoded["when"].startswith("2026-01-01")


def test_undigestable_value_raises():
    with pytest.raises(TypeError, match="cannot digest"):
        canonical_json({"f": object()})


def test_a_frozenset_and_a_mapping_encode_like_a_set_and_a_dict():
    assert canonical_json(frozenset({"b", "a"})) == canonical_json({"a", "b"})
    assert canonical_json(MappingProxyType({"b": 1, "a": 2})) == '{"a":2,"b":1}'


def test_tobytes_returning_non_bytes_is_refused():
    class _Weird:
        def tobytes(self):
            return "not bytes"

    with pytest.raises(TypeError, match="cannot digest a _Weird"):
        canonical_json(_Weird())
