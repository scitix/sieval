"""The identity boundary between an evaluation and the system it measures.

An evaluation subject may be a :class:`Model` called by SiEval, or an external
system that a task calls through its own client. Its request arguments must
include every stable subject detail whose change should invalidate inference.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import json
from collections.abc import Mapping
from typing import Literal, NotRequired, Protocol, TypedDict, runtime_checkable

from sieval.core.types import JSONValue
from sieval.core.utils.canonical import canonical_json

type SubjectMeta = Mapping[str, JSONValue]


class ServingFingerprint(TypedDict):
    """How one run served an evaluation subject, when serving is known.

    ``recipe`` is the serving plan of an engine launched by the run, without
    its checkpoint or host. ``stated`` carries caller-declared endpoint facts.
    ``group`` identifies the serving group, ``lifecycles`` records launched
    model lifecycles, and ``context`` records the served context window.
    """

    engine: str
    engine_version: str | None
    launch: Literal["in_run", "external"]
    recipe: dict[str, JSONValue] | None
    stated: NotRequired[dict[str, JSONValue]]
    group: NotRequired[str]
    lifecycles: NotRequired[dict[str, dict[str, str]]]
    context: NotRequired[dict[str, JSONValue]]


@runtime_checkable
class EvalSubject(Protocol):
    """Stable identity and request description for the subject under test."""

    def subject_meta(self) -> SubjectMeta:
        """Return JSON-compatible identity metadata, including a ``model`` name."""

    def call_key(self) -> str:
        """Return a plugin-defined stable key used to recognize recorded calls.

        :func:`subject_call_key` is an optional canonical implementation.
        """

    def request_args(self) -> JSONValue:
        """Return JSON-compatible request identity data.

        Changes here invalidate inference. A mapping is convenient when the
        subject has named arguments; other JSON values are also valid.
        """

    def serving(self) -> ServingFingerprint | None:
        """Return serving facts, or ``None`` when the subject is not a served model."""


def validate_eval_subject(subject: object) -> EvalSubject:
    """Validate the runtime contract before runner identity is recorded."""

    if not isinstance(subject, EvalSubject):
        raise TypeError("task.subject must implement EvalSubject")
    metadata = subject.subject_meta()
    if not isinstance(metadata, Mapping):
        raise TypeError("subject.subject_meta() must return a mapping")
    name = metadata.get("model")
    if not isinstance(name, str) or not name:
        raise ValueError("subject.subject_meta() must include a non-empty 'model' name")
    key = subject.call_key()
    if not isinstance(key, str) or not key:
        raise TypeError("subject.call_key() must return a non-empty string")
    request_args = subject.request_args()
    try:
        json.dumps(request_args, allow_nan=True, default=_mapping_json_value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "subject.request_args() must return a JSON-compatible value"
        ) from exc
    serving = subject.serving()
    if serving is not None and not isinstance(serving, Mapping):
        raise TypeError("subject.serving() must return a mapping or None")
    return subject


_CALL_KEY_EXCLUDED = frozenset({"api_base", "api_base_env", "provenance"})


def subject_call_key(meta: Mapping[str, object]) -> str:
    """Canonical subject call key, excluding endpoint and provenance details."""

    return canonical_json(
        {key: value for key, value in meta.items() if key not in _CALL_KEY_EXCLUDED}
    )


def _mapping_json_value(value: object) -> object:
    """Normalize a JSONValue Mapping implementation for validation."""

    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"cannot encode {type(value).__name__} as a subject JSON value")
