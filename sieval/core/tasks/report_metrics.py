"""The metric names a task's ``report.json`` can carry, known before it runs.

A platform checking a criterion such as ``score: {min: 0}`` needs to know
whether ``score`` is a metric the task reports at all; finding out from a
finished run's missing key is too late. :class:`ReportMetrics` says so per task:
``names`` it may write, and ``patterns`` for keys built from the data, such as
``score_{category}``, where a group absent from a run is simply not written.

"May write" is the whole claim. A declared name can still be absent from one
run (a key is omitted, never zeroed, when it cannot be computed); a name that
is neither declared nor matched is one the task never writes.

Only numeric keys are metrics: ``score_key``, ``denominator_policy``,
``ci95_units`` and the ``*_ci95`` intervals are not.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Literal

#: Keys the runner adds to a generative task's report beside the task's own.
RUNNER_GEN_METRICS = frozenset({"n_truncated", "n_scored_rollouts"})

#: What :func:`~sieval.core.tasks.metrics.sampling_report` can write: the
#: rollout block, the budget, and its population count. ``maj@k`` and
#: ``self_consistency`` only with votes.
SAMPLING_METRICS = frozenset(
    {"pass@1", "avg@n", "pass@k", "pass^k", "n", "k", "n_short", "n_problems"}
)
SAMPLING_VOTE_METRICS = frozenset({"maj@k", "self_consistency"})

#: What :func:`~sieval.core.tasks.metrics.health_metrics` writes.
HEALTH_METRICS = frozenset({"n_unextracted"})

_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


@cache
def _pattern_regex(pattern: str) -> re.Pattern[str]:
    """``score_{category}`` as a regex: each placeholder is one or more characters.

    A pattern needs a literal part, or it matches every name and checks nothing.
    """
    parts = _PLACEHOLDER.split(pattern)
    if len(parts) < 2 or not any(parts):
        raise ValueError(
            f"report metric pattern {pattern!r} needs a literal part and at least "
            "one {placeholder}, such as 'score_{category}'"
        )
    return re.compile(".+".join(re.escape(part) for part in parts))


@dataclass(frozen=True)
class ReportMetrics:
    """The numeric keys one task's report may write.

    *patterns* name keys built from the data, a ``{placeholder}`` standing for
    the varying part: ``score_{category}`` matches ``score_stem``.
    """

    names: frozenset[str]
    patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", frozenset(self.names))
        object.__setattr__(self, "patterns", tuple(dict.fromkeys(self.patterns)))
        if bad := sorted(n for n in self.names if not isinstance(n, str) or not n):
            raise ValueError(f"report metric names are non-empty strings: {bad!r}")
        for pattern in self.patterns:
            _pattern_regex(pattern)

    def match(self, name: str) -> Literal["name", "pattern"] | None:
        """``name`` when declared, ``pattern`` when only a pattern matches."""
        if name in self.names:
            return "name"
        if any(_pattern_regex(p).fullmatch(name) for p in self.patterns):
            return "pattern"
        return None

    def __or__(self, other: "ReportMetrics | Iterable[str]") -> "ReportMetrics":
        if isinstance(other, ReportMetrics):
            return ReportMetrics(
                self.names | other.names, (*self.patterns, *other.patterns)
            )
        return ReportMetrics(self.names | frozenset(other), self.patterns)

    def to_json(self) -> dict[str, list[str]]:
        """``{"names": [...], "patterns": [...]}``, names sorted."""
        return {"names": sorted(self.names), "patterns": list(self.patterns)}


def report_metrics(*names: str, patterns: Iterable[str] = ()) -> ReportMetrics:
    """:class:`ReportMetrics` from names written out."""
    return ReportMetrics(frozenset(names), tuple(patterns))


def _patched(owner: type) -> bool:
    """Whether *owner*'s ``report`` was defined outside *owner*'s module."""
    report = vars(owner)["report"]
    function = getattr(report, "__func__", report)
    return getattr(function, "__module__", owner.__module__) != owner.__module__


def class_path(cls: type) -> str:
    """``module.QualName``: how a table names a class."""
    return f"{cls.__module__}.{cls.__qualname__}"


def declared_report_metrics(
    task_cls: type, table: Mapping[str, ReportMetrics] | None = None
) -> ReportMetrics | None:
    """The declaration that describes *task_cls*'s report, or ``None``.

    A class declares by its own ``report_metrics`` attribute, or by an entry in
    *table* under :func:`class_path`. The nearest declaring class in the MRO
    answers. When a class nearer than it defines ``report``, that report may
    write keys the declaration never saw, and the answer is ``None``, unknown;
    so is an attribute set to ``None``.

    A ``report`` defined in another module than its class is a patch
    (``setattr(Task, "report", report)``), which no declaration on the class
    saw: unknown too.

    The runner's own keys (:data:`RUNNER_GEN_METRICS`) are added for a
    generative task.
    """
    table = table or {}
    mro = task_cls.__mro__
    report_at = next((i for i, k in enumerate(mro) if "report" in vars(k)), None)
    for depth, klass in enumerate(mro):
        if "report_metrics" in vars(klass):
            found = vars(klass)["report_metrics"]
        elif class_path(klass) in table:
            found = table[class_path(klass)]
        else:
            continue
        if found is None or report_at is None or report_at < depth:
            return None
        if _patched(mro[report_at]):
            return None
        if not isinstance(found, ReportMetrics):
            raise TypeError(
                f"{class_path(klass)}.report_metrics is a ReportMetrics or None, "
                f"not {type(found).__name__}"
            )
        if "gen" in getattr(task_cls, "tags", ()):
            found = found | RUNNER_GEN_METRICS
        return found
    return None
