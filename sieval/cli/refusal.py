"""Deliberate refusals: a stable exit code and one machine-readable line.

A refusal is a run SiEval stops on purpose before evaluating anything, where
retrying the same command on the same node refuses again. A scheduler reacts to
it differently from a crash, so it is told apart by exit code, and, for callers
that only see logs, by one stderr line::

    SIEVAL_REFUSAL {"v": 1, "kind": "resume", "reason": "data", "task": "t", ...}

=========  ====  ==========================================================
kind       exit  reasons
=========  ====  ==========================================================
resume     3     ``schema`` (result schema version), ``data``, ``inference``
hardware   4     ``no_gpu``, ``mismatch`` (not the declared GPUs)
=========  ====  ==========================================================

Crashes exit 1 and interruptions 130 / 143, as before. A failure that mixes a
refusal with anything else, or refusals of two kinds, exits 1; the line is
still printed for each refusal. ``task`` is present only for a per-task
refusal, and ``message`` is the first line of the human message, capped at
:data:`MESSAGE_LIMIT` characters. Changing a kind, reason or code bumps
:data:`REFUSAL_LINE_VERSION`.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import sys
from dataclasses import dataclass
from typing import Literal

from sieval.core.runners import StageResumeError
from sieval.core.runners.resume_gate import ResumeVersionError
from sieval.infer.introspect import LaunchGpuError

#: Version of the ``SIEVAL_REFUSAL`` line's fields and of the exit codes.
REFUSAL_LINE_VERSION = 1
REFUSAL_PREFIX = "SIEVAL_REFUSAL"
#: Characters of the human message the line carries.
MESSAGE_LIMIT = 200

RefusalKind = Literal["resume", "hardware"]
EXIT_CODES: dict[RefusalKind, int] = {"resume": 3, "hardware": 4}


@dataclass(frozen=True, slots=True)
class Refusal:
    kind: RefusalKind
    reason: str
    task: str | None
    message: str

    def line(self) -> str:
        """The ``SIEVAL_REFUSAL`` line, without a trailing newline."""
        first = self.message.strip().split("\n", 1)[0]
        if len(first) > MESSAGE_LIMIT:
            first = first[: MESSAGE_LIMIT - 1] + "…"
        payload: dict[str, object] = {
            "v": REFUSAL_LINE_VERSION,
            "kind": self.kind,
            "reason": self.reason,
        }
        if self.task is not None:
            payload["task"] = self.task
        payload["message"] = first
        return f"{REFUSAL_PREFIX} {json.dumps(payload, ensure_ascii=False)}"


def _classify(exc: BaseException) -> Refusal | None:
    if isinstance(exc, StageResumeError):
        return Refusal("resume", exc.stage, exc.task, str(exc))
    if isinstance(exc, ResumeVersionError):
        return Refusal("resume", "schema", exc.task, str(exc))
    if isinstance(exc, LaunchGpuError):
        return Refusal("hardware", exc.reason, None, str(exc))
    return None


def _leaves(exc: BaseException) -> list[BaseException]:
    """*exc*, or every non-group exception nested in it."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaves(sub)]
    return [exc]


def classify_refusals(exc: BaseException) -> tuple[list[Refusal], int | None]:
    """The refusals in *exc*, and the exit code they decide.

    The code is ``None`` unless every leaf of *exc* (a task batch fails as an
    ``ExceptionGroup``) is a refusal of one kind.
    """
    leaves = _leaves(exc)
    found = [r for r in map(_classify, leaves) if r is not None]
    kinds = {r.kind for r in found}
    if len(found) == len(leaves) and len(kinds) == 1:
        return found, EXIT_CODES[kinds.pop()]
    return found, None


def report_refusals(exc: BaseException) -> int | None:
    """Print one ``SIEVAL_REFUSAL`` line per refusal in *exc* to stderr.

    Returns the exit code :func:`classify_refusals` decides.
    """
    refusals, code = classify_refusals(exc)
    # Raw, not through loguru: a log format would put text before the token.
    for refusal in refusals:
        sys.stderr.write(refusal.line() + "\n")
    sys.stderr.flush()
    return code
