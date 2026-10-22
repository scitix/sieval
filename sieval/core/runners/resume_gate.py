"""Resume compatibility, keyed on the result schema version.

Pure, I/O-free: decides whether a task result directory may be resumed by the
running SiEval. The deciding number is ``meta.json`` ``result_schema_version``
(:data:`~sieval.core.tasks.run_meta.RESULT_SCHEMA_VERSION`), not the package
version. It versions the on-disk layout of a task's result directory that a
resume reads: shard records and their ``.idx`` files, ``meta.json`` and the
``report.json`` shape the loader reads. It is unrelated to ``identity.schema``
(``IDENTITY_SCHEMA``), which versions only the identity block inside
``meta.json``.

Whether a reused result is still valid is not decided here: the stage
identities decide that (:mod:`sieval.core.runners.stage_resume`). The package
version stays in ``meta.json`` ``version`` and ``runs[].sieval`` as provenance.

Bump ``RESULT_SCHEMA_VERSION`` when a release makes

* an incompatible change to that layout, one an older directory cannot be
  read under; or
* a change to core semantics the stage identities cannot see. The grader
  identity walks task and plugin code, not ``sieval.core`` or
  ``sieval.datasets``, so this number is the only guard against such a change.

0.7.0, 0.8.0 and 0.10.0 would each have bumped it. 0.11.0 does not: its
directories are laid out as 0.10.x's.

A directory written before the field existed maps from its recorded version:
0.10.0 and later, local builds included, is schema 1; anything older, an
unparseable version or ``0.0.0`` is refused.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import enum
from dataclasses import dataclass

from packaging.version import InvalidVersion, Version

from sieval.core.tasks.run_meta import RESULT_SCHEMA_VERSION

_ZERO = Version("0.0.0")

#: The first release that wrote each result schema version, for directories
#: that predate the ``result_schema_version`` field.
_FIRST_RELEASE: dict[int, Version] = {1: Version("0.10.0")}


class ResumeAction(enum.Enum):
    """Outcome of a resume compatibility check."""

    EXACT = "exact"
    COMPATIBLE = "compatible"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class ResumeVerdict:
    """A resume decision; ``reason`` is populated only for ``REJECT``."""

    action: ResumeAction
    reason: str = ""


class ResumeVersionError(RuntimeError):
    """Raised when a resume crosses an incompatible result schema version."""


def schema_from_version(v_run: str) -> int | None:
    """The result schema version a directory without the field was written in.

    ``None`` when *v_run* is unparseable, ``0.0.0``, or older than the first
    release of every known schema version.
    """
    try:
        parsed = Version(v_run)
    except InvalidVersion:
        return None
    if parsed == _ZERO:
        return None
    known = [s for s, first in _FIRST_RELEASE.items() if parsed >= first]
    return max(known) if known else None


def resume_schema_verdict(
    recorded_schema: object,
    v_run: str,
    v_cur: str,
    current_schema: int = RESULT_SCHEMA_VERSION,
) -> ResumeVerdict:
    """Decide whether a directory may resume under the running SiEval.

    *recorded_schema* is ``meta.json`` ``result_schema_version`` as read, or
    ``None`` when absent; *v_run* is the recorded ``version``, *v_cur* the
    running one.

    1. recorded schema absent      -> map it from *v_run*; refuse if unmappable
    2. recorded schema not an int  -> REJECT
    3. schema differs from current -> REJECT
    4. same schema, same version   -> EXACT
    5. same schema                 -> COMPATIBLE (dev and local builds included)
    """
    if recorded_schema is None:
        recorded_schema = schema_from_version(v_run)
        if recorded_schema is None:
            return ResumeVerdict(
                ResumeAction.REJECT,
                f"no result_schema_version, and version {v_run} predates "
                f"result schema version 1 ({_FIRST_RELEASE[1]}) or is not a "
                "release version",
            )
    elif isinstance(recorded_schema, bool) or not isinstance(recorded_schema, int):
        return ResumeVerdict(
            ResumeAction.REJECT,
            f"result_schema_version {recorded_schema!r} is not an integer",
        )

    if recorded_schema != current_schema:
        return ResumeVerdict(
            ResumeAction.REJECT,
            f"result schema version {recorded_schema} differs from {current_schema}",
        )
    if v_run == v_cur:
        return ResumeVerdict(ResumeAction.EXACT)
    return ResumeVerdict(ResumeAction.COMPATIBLE)


def format_reject_message(
    recorded_schema: object,
    v_run: str,
    v_cur: str,
    reason: str,
    current_schema: int = RESULT_SCHEMA_VERSION,
) -> str:
    """Build the operator-facing 'Resume aborted' message for a rejected resume."""
    recorded = "absent" if recorded_schema is None else repr(recorded_schema)
    return (
        "Resume aborted: the result directory's schema is incompatible with "
        "this sieval.\n"
        f"  persisted (meta.json): result_schema_version {recorded}, "
        f"sieval {v_run}\n"
        f"  current:               result_schema_version {current_schema}, "
        f"sieval {v_cur}\n"
        f"  reason: {reason}\n"
        "Either:\n"
        "  1. Remove the result_dir and start fresh\n"
        "  2. Reinstall a sieval that writes the persisted result schema version"
    )
