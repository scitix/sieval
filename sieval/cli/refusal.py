"""Deliberate refusals: exit code 3 (resume), 4 (hardware) or 5 (environment), plus one
``SIEVAL_REFUSAL {json}`` stderr line each. The contract, documented under
"Exit codes" in ``docs/guide/configuration.md``, is versioned by
:data:`REFUSAL_LINE_VERSION`.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import sys
from dataclasses import dataclass
from typing import Literal

from sieval.core.runners import StageResumeError
from sieval.core.runners.resume_gate import ResumeVersionError
from sieval.core.runners.stage_resume import ResumeStage
from sieval.core.tasks.python_requires import MissingPythonRequirementsError
from sieval.infer.introspect import LaunchGpuError, LaunchGpuReason

#: Bumped when a field, kind, reason or exit code changes.
REFUSAL_LINE_VERSION = 2
REFUSAL_PREFIX = "SIEVAL_REFUSAL"
MESSAGE_LIMIT = 200

RefusalKind = Literal["resume", "hardware", "environment"]
EXIT_CODES: dict[RefusalKind, int] = {"resume": 3, "hardware": 4, "environment": 5}


@dataclass(frozen=True, slots=True)
class Refusal:
    kind: RefusalKind
    reason: ResumeStage | Literal["schema", "python_requires"] | LaunchGpuReason
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


def _classify(exc: BaseException) -> list[Refusal]:
    if isinstance(exc, StageResumeError):
        return [Refusal("resume", exc.stage, exc.task, str(exc))]
    if isinstance(exc, ResumeVersionError):
        return [Refusal("resume", "schema", exc.task, str(exc))]
    if isinstance(exc, LaunchGpuError):
        return [Refusal("hardware", exc.reason, None, str(exc))]
    if isinstance(exc, MissingPythonRequirementsError):
        # One line per task, so a caller can tell which tasks to fix.
        return [
            Refusal(
                "environment",
                "python_requires",
                task,
                "Missing Python packages: " + "; ".join(map(str, unmet)),
            )
            for task, unmet in exc.unmet.items()
        ]
    return []


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
    per_leaf = [_classify(leaf) for leaf in _leaves(exc)]
    found = [r for refusals in per_leaf for r in refusals]
    kinds = {r.kind for r in found}
    if all(per_leaf) and len(kinds) == 1:
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
