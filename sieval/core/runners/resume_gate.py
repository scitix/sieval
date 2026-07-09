"""Resume version-compatibility decision logic.

Pure, I/O-free: decides whether a run persisted under one sieval version
may be resumed under another. The go/no-go ladder is exact-match-first, then
rejects unparseable / unknown / dev-local mismatches and incompatible version
series; otherwise the resume is compatible and recorded via per-record
provenance.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import enum
from dataclasses import dataclass

from packaging.version import InvalidVersion, Version

_ZERO = Version("0.0.0")


class ResumeAction(enum.Enum):
    """Outcome of a resume version check."""

    EXACT = "exact"
    COMPATIBLE = "compatible"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class ResumeVerdict:
    """A resume decision; ``reason`` is populated only for ``REJECT``."""

    action: ResumeAction
    reason: str = ""


class ResumeVersionError(RuntimeError):
    """Raised when a resume crosses an incompatible sieval version boundary."""


def _compat_key(v: Version) -> tuple[int, ...]:
    """Return the break-axis key: major post-1.0, (major, minor) under 1.0."""
    if v.major >= 1:
        return (v.major,)
    return (v.major, v.minor)


def resume_version_verdict(v_run: str, v_cur: str) -> ResumeVerdict:
    """Decide whether a run created under ``v_run`` may resume under ``v_cur``.

    1. exact string match                          -> EXACT
    2. unparseable, or either side is 0.0.0         -> REJECT (fail-closed)
    3. either side is a dev/local build             -> REJECT (unpinnable)
    4. incompatible break-axis (compat_key differs) -> REJECT
    5. otherwise (same series, non-exact)           -> COMPATIBLE
    """
    # Rule 1 short-circuits before parsing, so an identical pair always resumes
    # (even a "0.0.0" or dev/local string — you are resuming your own build).
    if v_run == v_cur:
        return ResumeVerdict(ResumeAction.EXACT)

    try:
        parsed_run = Version(v_run)
        parsed_cur = Version(v_cur)
    except InvalidVersion:
        return ResumeVerdict(ResumeAction.REJECT, "version string is unparseable")

    if parsed_run == _ZERO or parsed_cur == _ZERO:
        return ResumeVerdict(ResumeAction.REJECT, "version is unknown (0.0.0)")

    # Reject unpinnable builds only (dev/local: same tag, different code).
    # Pinnable pre-/post-releases fall through to the series check below.
    if (
        parsed_run.local is not None
        or parsed_run.is_devrelease
        or parsed_cur.local is not None
        or parsed_cur.is_devrelease
    ):
        return ResumeVerdict(
            ResumeAction.REJECT,
            "development/local build cannot be matched non-exactly",
        )

    if _compat_key(parsed_run) != _compat_key(parsed_cur):
        return ResumeVerdict(ResumeAction.REJECT, "incompatible version series")

    return ResumeVerdict(ResumeAction.COMPATIBLE)


def format_reject_message(v_run: str, v_cur: str, reason: str) -> str:
    """Build the operator-facing 'Resume aborted' message for a rejected resume."""
    return (
        "Resume aborted: sieval version is incompatible with the persisted run.\n"
        f"  persisted (meta.json): {v_run}\n"
        f"  current:               {v_cur}\n"
        f"  reason: {reason}\n"
        "Either:\n"
        "  1. Remove the result_dir and start fresh\n"
        "  2. Reinstall sieval matching the persisted version series"
    )
