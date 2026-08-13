"""JSON-shape primitives for the modules that build reconciliation records.

`capabilities`, `requirements` and `reconcile` all normalise user-supplied JSON
into the record shapes `reconcile()` consumes, so all three need the same
detach-and-validate step, and all three must agree on what "representable" means
-- a value one of them accepts and another rejects makes the record depend on
which module touched it first.

They held three private copies. Two were byte-identical; `reconcile`'s had
already drifted, rejecting the same values with different wording. That drift is
the argument for a single owner: this module exists because the rule is one
contract, not because the code happened to repeat.

`validate_nonempty_string` is here on weaker grounds -- three characters of
policy, and the three copies never diverged. It stays because `capabilities` and
`requirements` already import from this module, not as precedent for moving
every shared three-liner here.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import math
from collections.abc import Mapping

from sieval.core.types import JSONValue


def copy_json_value(value: object, path: str) -> JSONValue:
    """Deep-copy *value* into plain JSON, rejecting anything not representable.

    ``path`` is the dotted location reported in errors, so a rejection names
    the offending leaf rather than the whole record.
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
